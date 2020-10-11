# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ResNetBlockBase",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="SyncBN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenSyncBN", "SyncBN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For SyncBN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last SyncBN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="SyncBN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.

    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="SyncBN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenSyncBN", "SyncBN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8, grad_method='bptt', norm='SyncBN'):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method

        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])
        #     [
        #         nn.BatchNorm2d(
        #             hidden_size,
        #             eps=1e-03,
        #             affine=True,
        #             track_running_stats=False) for i in range(3)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)

        nn.init.xavier_uniform_(self.u1_gate.weight)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.kappa, 0.5)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.mu, 1)
        nn.init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, input_, prev_state2, timestep=0):

        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))
        c1_t = self.bn[0](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))

        next_state1 = F.softplus(input_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        h2_t = self.bn[1](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))

        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        return prev_state2


class tdConvGRUCell(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(self, fan_in, td_fan_in, diff_fan_in, kernel_size, batchnorm=True, timesteps=8, grad_method='bptt', norm='SyncBN'):
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.resize = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True)  # TODO: play with align_corners
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
        self.kappa = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.w = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))

        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(2)])
        #     [
        #         nn.BatchNorm2d(
        #             hidden_size,
        #             eps=1e-03,
        #             affine=True,
        #             track_running_stats=False) for i in range(3)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)

        nn.init.xavier_uniform_(self.u1_gate.weight)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.kappa, 0.5)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.mu, 1)
        nn.init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, lower_, higher_, timestep=0):
        prev_state2 = self.resize(higher_)
        prev_state2 = F.softplus(self.remap_0(prev_state2))
        prev_state2 = self.remap_1(prev_state2)

        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))
        c1_t = self.bn[0](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))

        next_state1 = F.softplus(lower_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        h2_t = self.bn[1](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))

        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t


        return prev_state2


class ResNet(Backbone):
    def __init__(
            self,
            stem,
            stages,
            norm='SyncBN',
            num_classes=None,
            out_features=None,
            rec_kernel_size=3,
            rec2_kernel_size=1,
            timesteps=3,
            recurrent_bn=True,
            grad_method='bptt'):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes
        self.rec_kernel_size = rec_kernel_size
        self.rec2_kernel_size = rec2_kernel_size
        self.recurrent_bn = recurrent_bn
        self.grad_method = grad_method
        self.timesteps = timesteps
        self.hidden_states = {}
        self.norm = norm

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []

        # Bottom-up pass
        horizontal_layers = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

            # Now add hGRU
            fan_in = blocks[-1].out_channels
            if i == (len(stages) - 1):
                ks = self.rec2_kernel_size
            else:
                ks = self.rec_kernel_size
            recurrent = hConvGRUCell(
                input_size=fan_in,
                hidden_size=fan_in,
                kernel_size=ks,
                batchnorm=self.recurrent_bn,
                timesteps=1,
                norm=self.norm,
                grad_method=self.grad_method)
            horizontal_name = "horizontal{}".format(i + 2)
            self.add_module(horizontal_name, recurrent)
            self.stages_and_names.append((recurrent, horizontal_name))
            horizontal_layers += [[horizontal_name, fan_in]]
            self._out_feature_strides[
                horizontal_name] = self._out_feature_strides[name]
            self._out_feature_channels[
                horizontal_name] = self._out_feature_channels[name]

        # Top-down pass
        for idx, (name, fan_in) in zip(
                np.arange(len(horizontal_layers))[::-1][:-1],
                horizontal_layers[::-1][1:]):
            strip_name = name.strip('horizontal')
            td_fan_in = horizontal_layers[idx][1]
            diff_fan_in = fan_in + ((td_fan_in - fan_in) // 2)
            recurrent = tdConvGRUCell(
                fan_in=fan_in,
                td_fan_in=td_fan_in,
                diff_fan_in=diff_fan_in,
                kernel_size=self.rec2_kernel_size,
                batchnorm=self.recurrent_bn,
                timesteps=1,
                norm=self.norm,
                grad_method=self.grad_method)
            td_name = "topdown{}".format(strip_name)
            self.add_module(td_name, recurrent)
            self.stages_and_names.append((recurrent, td_name))
            self._out_feature_strides[
                td_name] = self._out_feature_strides[name]
            self._out_feature_channels[
                td_name] = self._out_feature_channels[name]

        # Add readout
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        """4 Horizontals, 3 TDs, 6 remapping layers."""
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        # Do res 2 first and only once (0th element of
        # self.stages_and_names)
        res2 = self.stages_and_names[0][0](x)
        if "res2" in self._out_features:
            outputs["res2"] = res2

        # Do res 2 first and only once
        res2 = self.stages_and_names[0][0](x)
        for n in range(self.timesteps):
            x = res2
            for stage, name in self.stages_and_names[1:]:
                # print(name)
                if 'horizontal' in name:
                    if n == 0:
                        # Init the Hidden state
                        self.hidden_states[name] = torch.zeros_like(x)
                    x = stage(
                        input_=x,
                        prev_state2=self.hidden_states[name])
                    self.hidden_states[name] = x
                    if name in self._out_features:
                        outputs[name] = x
                elif 'topdown' in name:
                    layer_id = int(name.strip('topdown'))
                    lower_name = 'horizontal{}'.format(layer_id)
                    higher_name = 'horizontal{}'.format(layer_id + 1)
                    lower_activity = self.hidden_states[lower_name]
                    higher_activity = self.hidden_states[higher_name]
                    lower_activity = stage(
                        lower_=lower_activity,
                        higher_=higher_activity)
                    self.hidden_states[lower_name] = lower_activity
                    if name in self._out_features:
                        outputs[name] = lower_activity
                elif 'res' in name:
                    x = stage(x)
                    if name in self._out_features:
                        outputs[name] = x
                else:
                    raise NotImplementedError(name)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_resnet_gnbn_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    # out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    out_stage_idx = [{"topdown2": 2, "topdown3": 3, "topdown4": 4, "horizontal5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, norm=norm)

