# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from ..backbone.rnns import hConvGRUCell, hConvGRUExtraSMCell, RBPFun, CBP_penalty


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    if isinstance(pred_mask_logits, dict):
        timesteps = pred_mask_logits['timesteps']
        # print('Timesteps: {}'.format(timesteps))
        pred_mask_logits = pred_mask_logits['preds']
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


@ROI_MASK_HEAD_REGISTRY.register()
class Recurrentv1MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Recurrentv1MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        rnn_norm          = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.output_norm  = cfg.MODEL.ROI_MASK_HEAD.OUTPUTNORM
        self.timesteps    = cfg.MODEL.ROI_MASK_HEAD.TIMESTEPS
        norm              = cfg.MODEL.ROI_MASK_HEAD.NORM
        gala              = cfg.MODEL.ROI_MASK_HEAD.GALA
        nl                = cfg.MODEL.ROI_MASK_HEAD.NL
        self.grad_method  = cfg.MODEL.ROI_MASK_HEAD.GRAD
        self.xi           = cfg.MODEL.ROI_MASK_HEAD.XI
        self.tau          = cfg.MODEL.ROI_MASK_HEAD.TAU
        self.neumann_iterations = 15
        # fmt: on
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = hConvGRUCell(
            input_size=num_mask_classes,
            hidden_size=num_mask_classes,
            kernel_size=3,
            batchnorm=False,
            timesteps=self.timesteps,
            norm=rnn_norm,
            gala=gala,
            less_softplus=True,
            nl=nl,
            grad_method=self.grad_method)
        self.add_module("mask_fcn_rnn0", self.predictor)
        # self.horizontal_norm = get_norm(self.output_norm, num_mask_classes)
        self.deconv = ConvTranspose2d(
            conv_dims,  #  if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.preproc = Conv2d(conv_dims, num_mask_classes, kernel_size=3, stride=1, padding=0)

        weight_init.c2_msra_fill(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.preproc.weight, std=0.001)
        if self.preproc.bias is not None:
            nn.init.constant_(self.preproc.bias, 0)

    def readout(self, x):
        """Apply the readout head."""
        if len(self.output_norm):
            x = self.horizontal_norm(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)

    def forward(self, x, max_timesteps=-10, eps=-10, test_type='per-batch'):
        #print(test_type, self.timesteps)
        # x = self.horizontal_norm(x)
        x = self.deconv(x)
        x = self.preproc(x)
        if self.grad_method == 'bptt':
            for n in range(self.timesteps):
                if n == 0:
                    # Init the Hidden state
                    hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                hidden_state = self.predictor(
                    input_=x,  # Changed all act funs to softplus
                    h_=hidden_state)
            return hidden_state
        elif self.grad_method == 'cbp' or self.grad_method == 'rbp':
            penalty = 0.
            if self.training:
                timesteps = self.timesteps
                with torch.no_grad():
                    for n in range(timesteps - 1):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                        hidden_state = self.predictor(
                            input_=x,
                            h_=hidden_state)
                prev_state = hidden_state.clone().detach().requires_grad_()  # noqa
                last_state = self.predictor(
                    input_=x,  # Changed all act funs to softplus
                    h_=prev_state)
                x = RBPFun.apply(
                    prev_state,
                    last_state,
                    None,
                    None,
                    None,
                    self.neumann_iterations)
                penalty = penalty + CBP_penalty(
                    prev_state=prev_state,
                    last_state=last_state,
                    tau=self.tau,
                    compute_hessian=(self.grad_method.lower() == 'cbp'))
                preds = x
            elif test_type == 'per-batch':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)  # .requires_grad_()  # noqa
                        hidden_state = self.predictor(
                            input_=x,
                            h_=hidden_state)
                        #print(n, torch.norm(hidden_state))
                preds = hidden_state
            elif test_type == 'per-instance':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                mask = torch.ones_like(x)
                diffs = []
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)
                        hidden_state = self.predictor(
                            input_=x,
                            h_=hidden_state)
                        if n == 0 and eps > 0:
                            old_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            final_h = hidden_state
                        elif n > 0 and eps > 0:
                            new_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            diff = old_h - new_h
                            diff = diff.view(hidden_state.shape[0], -1).abs().mean(1)
                            diff_thresh = (diff > eps).float()
                            # if len(diffs):
                            #     diff_thresh = torch.min(diff_thresh, (diffs[-1] > diff).float())  # Stop if we're oscillating
                            # diffs.append(diff)
                            old_h = new_h

                            update_mask = (mask * diff_thresh.reshape(-1, 1, 1, 1)).float()
                            # print(n, mask.sum(),  torch.min(update_mask, mask).sum())
                            mask = torch.min(update_mask, mask)
                            final_h = final_h * (1 - update_mask) + hidden_state * update_mask
                            if not mask.sum():
                                break
                        else:
                            final_h = hidden_state
                preds = final_h
            outputs = {'preds': preds, 'vj_penalty': penalty * self.xi, 'timesteps': n + 2}
            return outputs
        else:
            raise NotImplementedError(self.grad_method)




@ROI_MASK_HEAD_REGISTRY.register()
class RecurrentSPMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(RecurrentSPMaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        rnn_norm          = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.output_norm  = cfg.MODEL.ROI_MASK_HEAD.OUTPUTNORM
        self.timesteps    = cfg.MODEL.ROI_MASK_HEAD.TIMESTEPS
        norm              = cfg.MODEL.ROI_MASK_HEAD.NORM
        gala              = cfg.MODEL.ROI_MASK_HEAD.GALA
        nl                = cfg.MODEL.ROI_MASK_HEAD.NL
        self.grad_method  = cfg.MODEL.ROI_MASK_HEAD.GRAD
        self.xi           = cfg.MODEL.ROI_MASK_HEAD.XI
        self.tau          = cfg.MODEL.ROI_MASK_HEAD.TAU
        self.neumann_iterations = 15
        # fmt: on

        self.recurrent = hConvGRUExtraSMCell(
            input_size=conv_dims,
            hidden_size=conv_dims,
            kernel_size=3,
            batchnorm=True,
            timesteps=self.timesteps,
            norm=rnn_norm,
            gala=gala,
            less_softplus=True,
            nl=nl,
            grad_method=self.grad_method)
        self.add_module("mask_fcn_rnn", self.recurrent)
        self.horizontal_norm = get_norm(self.output_norm, conv_dims)
        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        weight_init.c2_msra_fill(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def readout(self, x):
        """Apply the readout head."""
        if len(self.output_norm):
            x = self.horizontal_norm(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)

    def forward(self, x, max_timesteps=-10, eps=-10, test_type='per-batch'):
        #print(test_type, self.timesteps)
        if self.grad_method == 'bptt':
            for n in range(self.timesteps):
                if n == 0:
                    # Init the Hidden state
                    hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                hidden_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=hidden_state)
            preds = self.readout(hidden_state)
            return preds
        elif self.grad_method == 'cbp' or self.grad_method == 'rbp':
            penalty = 0.
            if self.training:
                timesteps = self.timesteps
                with torch.no_grad():
                    for n in range(timesteps - 1):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                prev_state = hidden_state.clone().detach().requires_grad_()  # noqa
                last_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=prev_state)
                x = RBPFun.apply(
                    prev_state,
                    last_state,
                    None,
                    None,
                    None,
                    self.neumann_iterations)
                penalty = penalty + CBP_penalty(
                    prev_state=prev_state,
                    last_state=last_state,
                    tau=self.tau,
                    compute_hessian=(self.grad_method.lower() == 'cbp'))
                preds = self.readout(x)
            elif test_type == 'per-batch':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)  # .requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        #print(n, torch.norm(hidden_state))
                # import pdb;pdb.set_trace()
                preds = self.readout(hidden_state)
            elif test_type == 'per-instance':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                mask = torch.ones_like(x)
                diffs = []
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        if n == 0 and eps > 0:
                            old_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            final_h = hidden_state
                        elif n > 0 and eps > 0:
                            new_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            diff = old_h - new_h
                            diff = diff.view(hidden_state.shape[0], -1).abs().mean(1)
                            diff_thresh = (diff > eps).float()
                            # if len(diffs):
                            #     diff_thresh = torch.min(diff_thresh, (diffs[-1] > diff).float())  # Stop if we're oscillating
                            # diffs.append(diff)
                            old_h = new_h

                            update_mask = (mask * diff_thresh.reshape(-1, 1, 1, 1)).float()
                            # print(n, mask.sum(),  torch.min(update_mask, mask).sum())
                            mask = torch.min(update_mask, mask)
                            final_h = final_h * (1 - update_mask) + hidden_state * update_mask
                            if not mask.sum():
                                break
                        else:
                            final_h = hidden_state
                preds = self.readout(final_h)
            outputs = {'preds': preds, 'vj_penalty': penalty * self.xi, 'timesteps': n + 2}
            return outputs
        else:
            raise NotImplementedError(self.grad_method)


@ROI_MASK_HEAD_REGISTRY.register()
class Recurrent5MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Recurrent5MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        rnn_norm          = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.output_norm  = cfg.MODEL.ROI_MASK_HEAD.OUTPUTNORM
        self.timesteps    = cfg.MODEL.ROI_MASK_HEAD.TIMESTEPS
        norm              = cfg.MODEL.ROI_MASK_HEAD.NORM
        gala              = cfg.MODEL.ROI_MASK_HEAD.GALA
        nl                = cfg.MODEL.ROI_MASK_HEAD.NL
        self.grad_method  = cfg.MODEL.ROI_MASK_HEAD.GRAD
        self.xi           = cfg.MODEL.ROI_MASK_HEAD.XI
        self.tau          = cfg.MODEL.ROI_MASK_HEAD.TAU
        self.neumann_iterations = 15
        # fmt: on

        self.recurrent = hConvGRUCell(
            input_size=conv_dims,
            hidden_size=conv_dims,
            kernel_size=5,
            batchnorm=True,
            timesteps=self.timesteps,
            norm=rnn_norm,
            gala=gala,
            less_softplus=True,
            nl=nl,
            grad_method=self.grad_method)
        self.add_module("mask_fcn_rnn", self.recurrent)
        self.horizontal_norm = get_norm(self.output_norm, conv_dims)
        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        weight_init.c2_msra_fill(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def readout(self, x):
        """Apply the readout head."""
        if len(self.output_norm):
            x = self.horizontal_norm(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)

    def forward(self, x, max_timesteps=-10, eps=-10, test_type='per-batch'):
        #print(test_type, self.timesteps)
        if self.grad_method == 'bptt':
            for n in range(self.timesteps):
                if n == 0:
                    # Init the Hidden state
                    hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                hidden_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=hidden_state)
            preds = self.readout(hidden_state)
            return preds
        elif self.grad_method == 'cbp' or self.grad_method == 'rbp':
            penalty = 0.
            if self.training:
                timesteps = self.timesteps
                with torch.no_grad():
                    for n in range(timesteps - 1):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                prev_state = hidden_state.clone().detach().requires_grad_()  # noqa
                last_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=prev_state)
                x = RBPFun.apply(
                    prev_state,
                    last_state,
                    None,
                    None,
                    None,
                    self.neumann_iterations)
                penalty = penalty + CBP_penalty(
                    prev_state=prev_state,
                    last_state=last_state,
                    tau=self.tau,
                    compute_hessian=(self.grad_method.lower() == 'cbp'))
                preds = self.readout(x)
            elif test_type == 'per-batch':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)  # .requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        #print(n, torch.norm(hidden_state))
                # import pdb;pdb.set_trace()
                preds = self.readout(hidden_state)
            elif test_type == 'per-instance':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                mask = torch.ones_like(x)
                diffs = []
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        if n == 0 and eps > 0:
                            old_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            final_h = hidden_state
                        elif n > 0 and eps > 0:
                            new_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            diff = old_h - new_h
                            diff = diff.view(hidden_state.shape[0], -1).abs().mean(1)
                            diff_thresh = (diff > eps).float()
                            # if len(diffs):
                            #     diff_thresh = torch.min(diff_thresh, (diffs[-1] > diff).float())  # Stop if we're oscillating
                            # diffs.append(diff)
                            old_h = new_h

                            update_mask = (mask * diff_thresh.reshape(-1, 1, 1, 1)).float()
                            # print(n, mask.sum(),  torch.min(update_mask, mask).sum())
                            mask = torch.min(update_mask, mask)
                            final_h = final_h * (1 - update_mask) + hidden_state * update_mask
                            if not mask.sum():
                                break
                        else:
                            final_h = hidden_state
                preds = self.readout(final_h)
            outputs = {'preds': preds, 'vj_penalty': penalty * self.xi, 'timesteps': n + 2}
            return outputs
        else:
            raise NotImplementedError(self.grad_method)


@ROI_MASK_HEAD_REGISTRY.register()
class RecurrentMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(RecurrentMaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        rnn_norm          = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.output_norm  = cfg.MODEL.ROI_MASK_HEAD.OUTPUTNORM
        self.timesteps    = cfg.MODEL.ROI_MASK_HEAD.TIMESTEPS
        norm              = cfg.MODEL.ROI_MASK_HEAD.NORM
        gala              = cfg.MODEL.ROI_MASK_HEAD.GALA
        nl                = cfg.MODEL.ROI_MASK_HEAD.NL
        self.grad_method  = cfg.MODEL.ROI_MASK_HEAD.GRAD
        self.xi           = cfg.MODEL.ROI_MASK_HEAD.XI
        self.tau          = cfg.MODEL.ROI_MASK_HEAD.TAU
        self.less_softplus      = cfg.MODEL.ROI_MASK_HEAD.LESS_SOFTPLUS
        self.neumann_iterations = 15
        # fmt: on

        self.recurrent = hConvGRUCell(
            input_size=conv_dims,
            hidden_size=conv_dims,
            kernel_size=3,
            batchnorm=True,
            timesteps=self.timesteps,
            norm=rnn_norm,
            gala=gala,
            less_softplus=self.less_softplus,
            nl=nl,
            grad_method=self.grad_method)
        self.add_module("mask_fcn_rnn", self.recurrent)
        self.horizontal_norm = get_norm(self.output_norm, conv_dims)
        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        weight_init.c2_msra_fill(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def readout(self, x):
        """Apply the readout head."""
        if len(self.output_norm):
            x = self.horizontal_norm(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)

    def forward(self, x, max_timesteps=-10, eps=-10, test_type='per-batch'):
        #print(test_type, self.timesteps)
        if self.grad_method == 'bptt':
            for n in range(self.timesteps):
                if n == 0:
                    # Init the Hidden state
                    hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                hidden_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=hidden_state)
            preds = self.readout(hidden_state)
            return preds
        elif self.grad_method == 'cbp' or self.grad_method == 'rbp':
            penalty = 0.
            if self.training:
                timesteps = self.timesteps
                with torch.no_grad():
                    for n in range(timesteps - 1):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x).requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                prev_state = hidden_state.clone().detach().requires_grad_()  # noqa
                last_state = self.recurrent(
                    input_=x,  # Changed all act funs to softplus
                    h_=prev_state)
                x = RBPFun.apply(
                    prev_state,
                    last_state,
                    None,
                    None,
                    None,
                    self.neumann_iterations)
                penalty = penalty + CBP_penalty(
                    prev_state=prev_state,
                    last_state=last_state,
                    tau=self.tau,
                    compute_hessian=(self.grad_method.lower() == 'cbp'))
                preds = self.readout(x)
            elif test_type == 'per-batch':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)  # .requires_grad_()  # noqa
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        #print(n, torch.norm(hidden_state))
                # import pdb;pdb.set_trace()
                preds = self.readout(hidden_state)
            elif test_type == 'per-instance':
                # self.horizontal_norm.training = True
                # self.horizontal_norm.train()
                timesteps = max(self.timesteps, max_timesteps)
                mask = torch.ones_like(x)
                diffs = []
                with torch.no_grad():
                    for n in range(timesteps):
                        if n == 0:
                            # Init the Hidden state
                            hidden_state = torch.zeros_like(x)
                        hidden_state = self.recurrent(
                            input_=x,
                            h_=hidden_state)
                        if n == 0 and eps > 0:
                            old_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            final_h = hidden_state
                        elif n > 0 and eps > 0:
                            new_h = hidden_state  # .view(hidden_state.shape[0], -1).abs().mean(1)  # self.readout(hidden_state)
                            diff = old_h - new_h
                            diff = diff.view(hidden_state.shape[0], -1).abs().mean(1)
                            diff_thresh = (diff > eps).float()
                            # if len(diffs):
                            #     diff_thresh = torch.min(diff_thresh, (diffs[-1] > diff).float())  # Stop if we're oscillating
                            # diffs.append(diff)
                            old_h = new_h
                            
                            update_mask = (mask * diff_thresh.reshape(-1, 1, 1, 1)).float()
                            # print(n, mask.sum(),  torch.min(update_mask, mask).sum())
                            mask = torch.min(update_mask, mask)
                            final_h = final_h * (1 - update_mask) + hidden_state * update_mask
                            if not mask.sum():
                                break
                        else:
                            final_h = hidden_state
                preds = self.readout(final_h)
            outputs = {'preds': preds, 'vj_penalty': penalty * self.xi, 'timesteps': n + 2}
            return outputs
        else:
            raise NotImplementedError(self.grad_method)
       


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
