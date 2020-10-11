import torch
import torch.nn.functional as F
from torch import nn
from detectron2.layers import get_norm
from torch.autograd import Function
import fvcore.nn.weight_init as weight_init


class RBPFun(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad, max_steps=20, norm_cap=True):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]
        if norm_cap:
            normsv = []
            normsg = []
            normg = torch.norm(neumann_g_prev)
            normsg.append(normg.data.item())
            normsv.append(normg.data.item())
        if truncate_iter <= 0:
            raise RuntimeError('Bug.')
            normv = 1.
            steps = 0.  # TODO: Add to TB
            while normv > 1e-9 and steps < max_steps:
                neumann_v = torch.autograd.grad(
                    last_state,
                    state_2nd_last,
                    grad_outputs=neumann_v_prev,
                    retain_graph=True,
                    allow_unused=True)
                neumann_g = neumann_g_prev + neumann_v[0]
                normv = torch.norm(neumann_v[0])
                steps += 1
                neumann_v_prev = neumann_v
                neumann_g_prev = neumann_g
        else:
            final_neum = None
            for ii in range(truncate_iter):
                neumann_v = torch.autograd.grad(
                    last_state,
                    state_2nd_last,
                    grad_outputs=neumann_v_prev,
                    retain_graph=True,
                    allow_unused=True)
                neumann_g = neumann_g_prev + neumann_v[0]
                if norm_cap:
                    normv = torch.norm(neumann_v[0])
                    normg = torch.norm(neumann_g)
                    if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                        normsg.append(normg.data.item())
                        normsv.append(normv.data.item())
                        neumann_g = neumann_g_prev
                        final_neum = neumann_g_prev
                    normsg.append(normg.data.item())
                    normsv.append(normv.data.item())
                neumann_v_prev = neumann_v
                neumann_g_prev = neumann_g
            if final_neum is not None:
                return (None, final_neum, None, None, None, None)
        return (None, neumann_g, None, None, None, None)


def CBP_penalty(
        last_state,
        prev_state,
        tau=0.95,  # Changed 2/25/20 from 0.9
        compute_hessian=True,
        pen_type='l1'):
    """Handles RBP grads in the forward pass."""
    """Compute the constrained RBP penalty."""
    norm_1_vect = torch.ones_like(last_state)
    norm_1_vect.requires_grad = False
    vj_prod = torch.autograd.grad(
        last_state,
        prev_state,
        grad_outputs=[norm_1_vect],
        retain_graph=True,
        create_graph=compute_hessian,
        allow_unused=True)[0]
    vj_penalty = (vj_prod - tau).clamp(0) ** 2  # Squared to emphasize outliers
    return vj_penalty.sum()  # Save memory with the mean


class hConvGRUCellv2(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=3,
            r=4,
            gate_fan_in=False,
            init=nn.init.orthogonal_,  # nn.init.xavier_uniform_
            grad_method='bptt',
            norm='GN'):
        super(hConvGRUCellv2, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.gate_fan_in = gate_fan_in

        if self.gala:
            self.u0_channel_gate_0 = nn.Conv2d(
                input_size, input_size // r, 1)
            self.u0_channel_gate_1 = nn.Conv2d(
                input_size // r, input_size, 1, bias=False)
            self.u0_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    input_size // r,
                    input_size,
                    kernel_size,
                    kernel_size))
            self.u0_spatial_bias = nn.Parameter(
                torch.empty((input_size // r, 1, 1)))
            self.u0_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    input_size // r,
                    kernel_size,
                    kernel_size))
            self.u0_combine_bias = nn.Parameter(
                torch.empty((input_size, 1, 1)))
        else:
            self.u0_gate = nn.Conv2d(input_size, input_size, 1)
        self.u1_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 1)
        # self.fan_out = nn.Conv2d(hidden_size, input_size, 3)
        self.fan_out_kernel = nn.Parameter(
            torch.empty(
                input_size,
                hidden_size,
                kernel_size,
                kernel_size))
        self.fan_out_bias = nn.Parameter(
            torch.empty((input_size, 1, 1)))
        if self.gate_fan_in:
            self.fan_in_gate = nn.Conv2d(input_size, hidden_size * hidden_size, 3)
            self.gate_fan_in_shape = [1, 1, hidden_size, hidden_size]
        self.w_gate_inh = nn.Parameter(
            torch.empty(input_size, input_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, input_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((input_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((input_size, 1, 1)))
        # self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        # self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if norm == "":
            raise NotImplementedError(norm)

        # Norm is harcoded to group norm
        bn_sizes = [input_size, hidden_size]
        self.bn = nn.ModuleList(
            [get_norm(norm, bn_sizes[i]) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)
        # init(self.fan_out.weight)
        init(self.fan_out_kernel)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)
        nn.init.constant_(self.fan_out_bias, 0.)
        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 0.)
        # nn.init.constant_(self.w, 0.5)
        # nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            init(self.u0_channel_gate_0.weight)
            init(self.u0_channel_gate_1.weight)
            init(self.u0_spatial_gate_0)
            init(self.u0_spatial_gate_1)
            init(self.u1_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u0_spatial_bias)
            nn.init.uniform_(self.u0_combine_bias, 1, init_timesteps - 1)
            self.u0_combine_bias.data.log()
            self.u1_gate.bias.data = -nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1).log()  # hack
        else:
            nn.init.xavier_uniform_(self.u0_gate.weight)
            nn.init.uniform_(self.u0_gate.bias.data, 1, init_timesteps - 1)
            self.u0_gate.bias.data.log()
            self.u1_gate.bias.data = -nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1).log()  # hack

    def forward(self, input_, h_, timestep=0):
        # Learn the fan-out gate. Keep low-D h_ in memory.
        # I gates are high dim, Inh is high->low, Exc is low->low
        # fan_out_kernel = self.hyper_gate(F.adaptive_avg_pool2d(h_, (1, 1)))
        # fan_out_h = F.conv2d(h_, fan_out_kernel, padding=self.padding)
        # fan_out_kernel = self.hyper_gate(F.adaptive_avg_pool2d(h_, (1, 1)))
        # fan_out_h_ = self.fan_out(h_)
        fan_out_h_ = F.conv2d(h_, self.fan_out_kernel, padding=self.padding) + self.fan_out_bias
        if self.gala:
            global_0 = F.softplus(self.u0_channel_gate_0(fan_out_h_))
            global_1 = self.u0_channel_gate_1(global_0)
            local_0 = F.softplus(
                F.conv2d(
                    fan_out_h_,
                    self.u0_spatial_gate_0,
                    padding=self.padding) + self.u0_spatial_bias)
            local_1 = F.conv2d(
                local_0,
                self.u0_spatial_gate_1,
                padding=self.padding)
            gate_act = global_1 * local_1 + self.u0_combine_bias
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u0_gate(fan_out_h_))
        c0_t = self.bn[0](
            F.conv2d(
                fan_out_h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        # inhibition = F.softplus(  # F.softplus(input_) moved outside
        inhibition = F.softplus(input_ - F.softplus(c0_t * (self.alpha * fan_out_h_ + self.mu)))
        g1_t = torch.sigmoid(self.u1_gate(torch.cat((inhibition, h_), 1)))
        if self.gate_fan_in:
            # Gate the excitation kernel
            raise NotImplementedError('Doesnt work.')
            fan_in_mask = F.sigmoid(F.adaptive_avg_pool2d(self.fan_in_gate(inhibition), (1, 1)))
            groups = inhibition.shape[0]
            fan_in_mask = torch.reshape(fan_in_mask, [groups] + self.gate_fan_in_shape)
            w_gate_exc = self.w_gate_exc.unsqueeze(0) * fan_in_mask
            h_t = F.softplus(self.bn[1](
                F.conv2d(
                    inhibition,
                    self.w_gate_exc,
                    padding=self.padding,
                    groups=groups)))
        else:
            h_t = F.softplus(self.bn[1](
                F.conv2d(
                    inhibition,
                    self.w_gate_exc,
                    padding=self.padding)))
        # h_t = F.softplus(
        #     self.kappa * (
        #         inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        return op


class hConvExplGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=5,
            r=4,
            explain_weight=False,
            grad_method='bptt',
            version='post',
            init=nn.init.orthogonal_,
            norm='GN'):
        super(hConvExplGRUCell, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.version = version
        self.gala = gala
        self.explain_weight = explain_weight

        if self.gala:
            self.u0_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u0_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u0_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    hidden_size // 2,
                    hidden_size,
                    kernel_size,
                    kernel_size))
            self.u0_spatial_bias = nn.Parameter(
                torch.empty((hidden_size // 2, 1, 1)))
            self.u0_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    hidden_size // 2,
                    kernel_size,
                    kernel_size))
            self.u0_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
            if self.explain_weight:
                self.explain_gate = nn.Conv2d(
                    hidden_size, hidden_size, 1)
        else:
            self.u0_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            init(self.u0_channel_gate_0.weight)
            init(self.u0_channel_gate_1.weight)
            init(self.u0_spatial_gate_0)
            init(self.u0_spatial_gate_1)
            init(self.u1_gate.weight)
            if self.explain_weight:
                init(self.explain_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u0_spatial_bias)
            nn.init.uniform_(self.u0_combine_bias, 1, init_timesteps - 1)
            self.u0_combine_bias.data.log()
            self.u1_gate.bias.data = -self.u0_combine_bias.data.squeeze()
        else:
            init(self.u0_gate.weight)
            nn.init.uniform_(self.u0_gate.bias.data, 1, init_timesteps - 1)
            self.u0_gate.bias.data.log()
            self.u1_gate.bias.data = -self.u0_gate.bias.data

    def forward(self, input_, h_, e_, timestep=0):
        if self.gala:
            if self.explain_weight:
                explained = self.explain_gate(e_)  # noqa Flip and project
            else:
                explained = torch.log(e_ / (1 - e_))  # noqa Flip and convert to logits
            if self.version == 'pre':
                h_exp = h_ * explained
                global_0 = F.softplus(self.u0_channel_gate_0(h_exp))
                local_0 = F.softplus(
                    F.conv2d(
                        h_exp,
                        self.u0_spatial_gate_0,
                        padding=self.padding) + self.u0_spatial_bias)
            elif self.version == 'post':
                global_0 = F.softplus(self.u0_channel_gate_0(h_))
                local_0 = F.softplus(
                    F.conv2d(
                        h_,
                        self.u0_spatial_gate_0,
                        padding=self.padding) + self.u0_spatial_bias)
            else:
                raise NotImplementedError(self.version)
            global_1 = self.u0_channel_gate_1(global_0)
            local_1 = F.conv2d(
                local_0,
                self.u0_spatial_gate_1,
                padding=self.padding)
            if self.version == 'pre':
                gate_act = global_1 * local_1 + self.u0_combine_bias
            elif self.version == 'post':
                gate_act = global_1 * local_1 - F.softplus(explained) + self.u0_combine_bias  # noqa
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u0_gate(h_))
        c0_t = self.bn[0](
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        inhibition = F.softplus(
            input_ - F.softplus(c0_t * (self.alpha * h_ + self.mu)))
        g1_t = torch.sigmoid(self.u1_gate(inhibition))
        excitation = self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding))
        h_t = F.softplus(
            self.kappa * (
                inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        if self.gala:
            e_ = e_ + g1_t  # Accumulate sigmoid scores
            return op, e_
        else:
            return op


class hConvGRUExtraSMCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=3,
            less_softplus=False,
            r=4,
            nl='softmax',
            # init=nn.init.orthogonal_,
            init=nn.init.kaiming_normal_,
            grad_method='bptt',
            norm='GN'):
        super(hConvGRUExtraSMCell, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.nl = nl
        self.less_softplus = less_softplus
        if self.gala:
            self.u0_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u0_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u0_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    hidden_size // 2,
                    hidden_size,
                    kernel_size,
                    kernel_size))
            self.u0_spatial_bias = nn.Parameter(
                torch.empty((hidden_size // 2, 1, 1)))
            self.u0_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    hidden_size // 2,
                    kernel_size,
                    kernel_size))
            self.u0_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u0_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        if not self.less_softplus:
            self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
            self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        # Norm is harcoded to group norm
        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        if not self.less_softplus:
            nn.init.constant_(self.w, 0.5)
            nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            init(self.u0_channel_gate_0.weight)
            init(self.u0_channel_gate_1.weight)
            init(self.u0_spatial_gate_0)
            init(self.u0_spatial_gate_1)
            init(self.u1_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u0_spatial_bias)
            nn.init.uniform_(self.u0_combine_bias, 1, init_timesteps - 1)
            self.u0_combine_bias.data.log()
            self.u1_gate.bias.data = -self.u0_combine_bias.data.squeeze()
        else:
            nn.init.xavier_uniform_(self.u0_gate.weight)
            nn.init.uniform_(self.u0_gate.bias.data, 1, init_timesteps - 1)
            self.u0_gate.bias.data.log()
            self.u1_gate.bias.data = -self.u0_gate.bias.data

    def forward(self, input_, h_, timestep=0):
        if self.nl == 'softplus':
            nl = F.softplus
        elif self.nl == 'tanh':
            nl = torch.tanh
        elif self.nl == 'relu':
            nl = F.relu
        else:
            raise NotImplementedError("Nonlinearity set to: {}".format(self.nl))
        if self.gala:
            global_0 = nl(self.u0_channel_gate_0(h_))
            global_1 = self.u0_channel_gate_1(global_0)
            local_0 = nl(
                F.conv2d(
                    h_,
                    self.u0_spatial_gate_0,
                    padding=self.padding) + self.u0_spatial_bias)
            local_1 = F.conv2d(
                local_0,
                self.u0_spatial_gate_1,
                padding=self.padding)
            gate_act = global_1 * local_1 + self.u0_combine_bias
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u0_gate(h_))
        c0_t = self.bn[0](
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        if self.less_softplus:
            inhibition = nl(  # F.softplus(input_) moved outside
                input_ - c0_t * (self.alpha * h_ + self.mu))
        else:
            inhibition = nl(  # F.softplus(input_) moved outside
                nl(input_) - nl(c0_t * (self.alpha * h_ + self.mu)))
        g1_t = torch.sigmoid(self.u1_gate(inhibition))
        excitation = nl(self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding)))
        if self.less_softplus:
            h_t = excitation
        else:
            h_t = nl(
                self.kappa * (
                    inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        op = F.softplus(op)  # Alekh trick
        return op


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=3,
            less_softplus=False,
            r=4,
            nl='softmax',
            # init=nn.init.orthogonal_,
            init=nn.init.kaiming_normal_,
            grad_method='bptt',
            norm='GN'):
        super(hConvGRUCell, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.nl = nl
        self.less_softplus = less_softplus
        if self.gala:
            self.u0_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u0_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u0_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    hidden_size // 2,
                    hidden_size,
                    kernel_size,
                    kernel_size))
            self.u0_spatial_bias = nn.Parameter(
                torch.empty((hidden_size // 2, 1, 1)))
            self.u0_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    hidden_size // 2,
                    kernel_size,
                    kernel_size))
            self.u0_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u0_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        if not self.less_softplus:
            self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
            # self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        # Norm is harcoded to group norm
        if self.batchnorm:
            self.bn = nn.ModuleList(
                [get_norm(norm, hidden_size) for i in range(2)])
            for bn in self.bn:
                nn.init.constant_(bn.weight, 0.1)

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        if not self.less_softplus:
            nn.init.constant_(self.w, 0.5)
            # nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            init(self.u0_channel_gate_0.weight)
            init(self.u0_channel_gate_1.weight)
            init(self.u0_spatial_gate_0)
            init(self.u0_spatial_gate_1)
            init(self.u1_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u0_spatial_bias)
            nn.init.uniform_(self.u0_combine_bias, 1, init_timesteps - 1)
            self.u0_combine_bias.data.log()
            self.u1_gate.bias.data = -self.u0_combine_bias.data.squeeze()
        else:
            nn.init.xavier_uniform_(self.u0_gate.weight)
            nn.init.uniform_(self.u0_gate.bias.data, 1, init_timesteps - 1)
            self.u0_gate.bias.data.log()
            self.u1_gate.bias.data = -self.u0_gate.bias.data

    def forward(self, input_, h_, timestep=0):
        if self.nl == 'softplus':
            nl = F.softplus
        elif self.nl == 'tanh':
            nl = torch.tanh
        elif self.nl == 'sigmoid':
            nl = torch.sigmoid
        elif self.nl == 'relu':
            nl = F.relu
        else:
            raise NotImplementedError("Nonlinearity set to: {}".format(self.nl))
        if self.gala:
            global_0 = nl(self.u0_channel_gate_0(h_))
            global_1 = self.u0_channel_gate_1(global_0)
            local_0 = nl(
                F.conv2d(
                    h_,
                    self.u0_spatial_gate_0,
                    padding=self.padding) + self.u0_spatial_bias)
            local_1 = F.conv2d(
                local_0,
                self.u0_spatial_gate_1,
                padding=self.padding)
            gate_act = global_1 * local_1 + self.u0_combine_bias
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u0_gate(h_))
        c0_t = F.conv2d(
            h_ * g1_t,
            self.w_gate_inh,
            padding=self.padding)
        if self.batchnorm:
            c0_t = self.bn[0](c0_t)
        if self.less_softplus:
            inhibition = nl(  # F.softplus(input_) moved outside
                input_ - c0_t * (self.alpha * h_ + self.mu))
        else:
            inhibition = nl(  # F.softplus(input_) moved outside
                input_ - c0_t * (self.alpha * h_ + self.mu))
        g1_t = torch.sigmoid(self.u1_gate(inhibition))
        c1_t = F.conv2d(
            inhibition,
            self.w_gate_exc,
            padding=self.padding)
        if self.batchnorm:
            c1_t = self.bn[1](c1_t)
        excitation = nl(c1_t)
        if self.less_softplus:
            h_t = excitation
        else:
            h_t = inhibition + nl(self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        return op


class hConvGRUCellOld(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=5,
            r=4,
            grad_method='bptt',
            norm='GN'):
        super(hConvGRUCellOld, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala

        if self.gala:
            self.u1_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u1_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u1_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    hidden_size // 2,
                    hidden_size,
                    kernel_size,
                    kernel_size))
            self.u1_spatial_bias = nn.Parameter(
                torch.empty((hidden_size // 2, 1, 1)))
            self.u1_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    hidden_size // 2,
                    kernel_size,
                    kernel_size))
            self.u1_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            nn.init.xavier_uniform_(self.u1_channel_gate_0.weight)
            nn.init.xavier_uniform_(self.u1_channel_gate_1.weight)
            nn.init.xavier_uniform_(self.u1_spatial_gate_0)
            nn.init.xavier_uniform_(self.u1_spatial_gate_1)
            nn.init.xavier_uniform_(self.u2_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u1_spatial_bias)
            nn.init.uniform_(self.u1_combine_bias, 1, init_timesteps - 1)
            self.u1_combine_bias.data.log()
            self.u2_gate.bias.data = -self.u1_combine_bias.data.squeeze()
        else:
            nn.init.xavier_uniform_(self.u1_gate.weight)
            nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1)
            self.u1_gate.bias.data.log()
            self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, input_, h_, timestep=0):
        if self.gala:
            global_0 = F.softplus(self.u1_channel_gate_0(h_))
            global_1 = self.u1_channel_gate_1(global_0)
            local_0 = F.softplus(
                F.conv2d(
                    h_,
                    self.u1_spatial_gate_0,
                    padding=self.padding) + self.u1_spatial_bias)
            local_1 = F.conv2d(
                local_0,
                self.u1_spatial_gate_1,
                padding=self.padding)
            gate_act = global_1 * local_1 + self.u1_combine_bias
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u1_gate(h_))
        c0_t = self.bn[0](
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        inhibition = F.softplus(
            input_ - F.softplus(c0_t * (self.alpha * h_ + self.mu)))
        g1_t = torch.sigmoid(self.u2_gate(inhibition))
        excitation = self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding))
        h_t = F.softplus(
            self.kappa * (
                inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        return op


class tdConvGRUCellOld(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(
            self,
            fan_in,
            td_fan_in,
            diff_fan_in,
            kernel_size,
            gala=False,
            batchnorm=True,
            timesteps=1,
            init=nn.init.orthogonal_,
            grad_method='bptt',
            norm='GN'):
        super(tdConvGRUCellOld, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.remap_0 = nn.Conv2d(td_fan_in, diff_fan_in, 1)
        self.remap_1 = nn.Conv2d(diff_fan_in, fan_in, 1)

        self.u1_gate = nn.Conv2d(fan_in, fan_in, 1)
        self.u2_gate = nn.Conv2d(fan_in, fan_in, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))

        # self.alpha = nn.Parameter(torch.empty((fan_in, 1, 1)))
        # self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))
        # self.w = nn.Parameter(torch.empty((fan_in, 1, 1)))
        # self.kappa = nn.Parameter(torch.empty((fan_in, 1, 1)))

        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(3)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        init(self.u1_gate.weight)
        init(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        # nn.init.constant_(self.alpha, 0.1)
        # nn.init.constant_(self.mu, 1)
        # nn.init.constant_(self.w, 0.5)
        # nn.init.constant_(self.kappa, 0.5)
        nn.init.uniform_(self.u1_gate.bias.data, 1, self.timesteps - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, lower_, higher_, timestep=0):
        prev_state2 = F.interpolate(
            higher_,
            scale_factor=2,
            mode="nearest")
        prev_state2 = F.softplus(self.remap_0(prev_state2))
        prev_state2 = self.bn[0](self.remap_1(prev_state2))
        g1_t = torch.sigmoid(self.u1_gate(prev_state2))
        c1_t = self.bn[1](
            F.conv2d(
                prev_state2 * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        inhibition = F.softplus(
            lower_ - F.softplus(c1_t * prev_state2))
        g2_t = torch.sigmoid(self.u2_gate(inhibition))
        excitation = F.softplus(self.bn[2](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding)))
        h2_t = excitation  # + self.w * inhibition * excitation
        # op = (1 - g2_t) * lower_ + g2_t * h2_t  # noqa Note: a previous version had higher_ in place of lower_
        op = (1 - g2_t) * lower_ + g2_t * h2_t  # noqa Note: a previous version had higher_ in place of lower_
        return op


class tdConvGRUCell(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(
            self,
            fan_in,
            td_fan_in,
            diff_fan_in,
            kernel_size,
            gala=False,
            batchnorm=True,
            timesteps=1,
            init=nn.init.orthogonal_,
            grad_method='bptt',
            norm='SyncBN'):
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.remap_0 = nn.Conv2d(td_fan_in, diff_fan_in, 1)
        self.remap_1 = nn.Conv2d(diff_fan_in, fan_in, 1)

        self.u1_gate = nn.Conv2d(fan_in, fan_in, 1)
        self.u2_gate = nn.Conv2d(fan_in, fan_in, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.w = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((fan_in, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        init(self.u1_gate.weight)
        init(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.kappa, 0.5)
        # if self.timesteps == 1:
        #     init_timesteps = 2
        # else:
        #     init_timesteps = self.timesteps
        nn.init.uniform_(self.u1_gate.bias.data, 1, self.timesteps - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, lower_, higher_, timestep=0):
        prev_state2 = F.interpolate(
            higher_,
            scale_factor=2,
            mode="nearest")
        prev_state2 = F.softplus(self.remap_0(prev_state2))
        prev_state2 = F.softplus(self.remap_1(prev_state2))
        g1_t = torch.sigmoid(self.u1_gate(prev_state2))
        c1_t = self.bn[0](
            F.conv2d(
                prev_state2 * g1_t,
                self.w_gate_inh,
                padding=self.padding))

        inhibition = F.softplus(
            lower_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid(self.u2_gate(inhibition))
        excitation = self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding))
        h2_t = F.softplus(
            self.kappa * (
                inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g2_t) * lower_ + g2_t * h2_t  # noqa Note: a previous version had higher_ in place of lower_
        return op
