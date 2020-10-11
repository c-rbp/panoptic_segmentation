import torch
import torch.nn.functional as F
from torch import nn
from detectron2.layers import get_norm
from torch.autograd import Function


class RBPFun(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(
                last_state,
                state_2nd_last,
                grad_outputs=neumann_v_prev,
                retain_graph=True,
                allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            normsv.append(normv.data.item())
            normsg.append(normg.data.item())
        return (None, neumann_g, None, None, None, None)


class CBPForward(object):
    """Handles RBP grads in the forward pass."""
    def __init__(self, mu=0.9, compute_hessian=True, pen_type='l1'):
        """Set defaults for CBP."""
        self.mu = mu
        self.compute_hessian = compute_hessian
        self.pen_type = pen_type  # Not used right now

    def __call__(self, last_state, prev_state):
        """Compute the constrained RBP penalty."""
        norm_1_vect = torch.ones_like(last_state)
        norm_1_vect.requires_grad = False
        vj_prod = torch.autograd.grad(
            last_state,
            prev_state,
            grad_outputs=[norm_1_vect],
            retain_graph=True,
            create_graph=self.compute_hessian,
            allow_unused=True)[0]
        vj_penalty = (vj_prod - self.mu).clamp(0) ** 2
        return vj_penalty.mean()  # Save memory with the mean


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
            spatial_kernel=5,
            r=4,
            grad_method='bptt',
            norm='SyncBN'):
        super(hConvGRUCell, self).__init__()
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
            self.u1_spatial_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, spatial_kernel)
            self.u1_spatial_gate_1 = nn.Conv2d(
                hidden_size // r, 1, spatial_kernel, bias=False)
            self.u1_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
            nn.init.xavier_uniform_(self.u1_gate.weight)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.mu, 1)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            nn.init.uniform_(self.u1_combine_bias, 1, init_timesteps - 1)
            self.u1_combine_bias.data.log()
        else:
            nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1)
            self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, input_, h_, timestep=0):
        if self.gala:
            global_0 = F.softplus(self.u1_channel_gate_0(h_))
            global_1 = self.u1_channel_gate_1(global_0)
            local_0 = F.softplus(self.u1_spatial_gate_0(h_))
            local_1 = self.u1_spatial_gate_1(local_0)
            g1_t = F.softplus(global_1 * local_1 + self.u1_combine_bias)
        else:
            g1_t = torch.sigmoid(self.u1_gate(h_))
        c1_t = self.bn[0](
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        next_state1 = F.softplus(
            input_ - F.softplus(c1_t * (self.alpha * h_ + self.mu)))
        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv2d(
                next_state1,
                self.w_gate_exc,
                padding=self.padding))
        h_ = (1 - g2_t) * h_ + g2_t * h2_t
        return h_


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
            batchnorm=True,
            timesteps=8,
            grad_method='bptt',
            norm='SyncBN'):
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.remap_0 = nn.Conv2d(td_fan_in, diff_fan_in, 1)
        self.remap_1 = nn.Conv2d(diff_fan_in, fan_in, 1)

        self.u1_gate = nn.Conv2d(fan_in, fan_in, 1)
        self.u2_gate = nn.Conv2d(fan_in, fan_in, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)

        nn.init.xavier_uniform_(self.u1_gate.weight)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.mu, 1)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1)
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

        next_state1 = F.softplus(
            lower_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv2d(
                next_state1,
                self.w_gate_exc,
                padding=self.padding))

        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t
        return prev_state2
