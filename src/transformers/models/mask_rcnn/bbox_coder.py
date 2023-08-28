import torch
from torch import nn
import numpy as np


def bbox2delta(proposals, ground_truth, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    """Compute deltas of proposals w.r.t. ground truth.

    We usually compute the deltas of x, y, width, height of proposals w.r.t ground truth bboxes to get regression
    target. This is the inverse function of [`delta2bbox`].

    Args:
        proposals (`torch.Tensor`):
            Boxes to be transformed, shape (N, ..., 4) with N = number of proposals.
        ground_truth (`torch.Tensor`):
            Ground truth bboxes to be used as base, shape (N, ..., 4) with N = number of boxes.
        means (`Sequence[float]`, *optional*, defaults to `(0.0, 0.0, 0.0, 0.0)`):
            Denormalizing means for delta coordinates
        stds (`Sequence[float]`, *optional*, defaults to `(1.0, 1.0, 1.0, 1.0)`):
            Denormalizing standard deviation for delta coordinates

    Returns:
       `torch.Tensor`: deltas with shape (N, 4), where columns represent delta_x, delta_y, delta_width, delta_height.
    """
    if proposals.size() != ground_truth.size():
        raise ValueError("Should have as many proposals as there are ground truths")

    proposals = proposals.float()
    ground_truth = ground_truth.float()

    # predicted boxes
    predicted_x = (proposals[..., 0] + proposals[..., 2]) * 0.5
    predicted_y = (proposals[..., 1] + proposals[..., 3]) * 0.5
    predicted_width = proposals[..., 2] - proposals[..., 0]
    predicted_height = proposals[..., 3] - proposals[..., 1]

    # ground truth boxes
    ground_truth_x = (ground_truth[..., 0] + ground_truth[..., 2]) * 0.5
    ground_truth_y = (ground_truth[..., 1] + ground_truth[..., 3]) * 0.5
    ground_truth_width = ground_truth[..., 2] - ground_truth[..., 0]
    ground_truth_height = ground_truth[..., 3] - ground_truth[..., 1]

    delta_x = (ground_truth_x - predicted_x) / predicted_width
    delta_y = (ground_truth_y - predicted_y) / predicted_height
    delta_width = torch.log(ground_truth_width / predicted_width)
    delta_height = torch.log(ground_truth_height / predicted_height)
    deltas = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(
    rois,
    deltas,
    means=(0.0, 0.0, 0.0, 0.0),
    stds=(1.0, 1.0, 1.0, 1.0),
    max_shape=None,
    wh_ratio_clip=16 / 1000,
    clip_border=True,
    add_ctr_clamp=False,
    ctr_clamp=32,
):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are network outputs used to shift/scale
    those boxes. This is the inverse function of [`bbox2delta`].

    Args:
        rois (`torch.Tensor`):
            Boxes to be transformed. Has shape (N, 4) with N = num_base_anchors * width * height, when rois is a grid
            of anchors.
        deltas (`torch.Tensor`):
            Encoded offsets relative to each roi. Has shape (N, num_classes * 4) or (N, 4) with N = num_base_anchors *
            width * height. Offset encoding follows https://arxiv.org/abs/1311.2524.
        means (`Sequence[float]`, *optional*, defaults to `(0., 0., 0., 0.)`):
            Denormalizing means for delta coordinates.
        stds (`Sequence[float]`, *optional*, defaults to `(1., 1., 1., 1.)`):
            Denormalizing standard deviation for delta coordinates.
        max_shape (`Tuple[int, int]`, *optional*):
            Maximum bounds for boxes, specifies (H, W). Default None.
        wh_ratio_clip (`float`, *optional*, defaults to 16 / 1000):
            Maximum aspect ratio for boxes.
        clip_border (`bool`, *optional*, defaults to `True`):
            Whether to clip the objects outside the border of the image.
        add_ctr_clamp (`bool`, *optional*, defaults to `False`):
            Whether to add center clamp. When set to True, the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor. Only used by YOLOF.
        ctr_clamp (`int`, *optional*, defaults to 32):
            The maximum pixel shift to clamp. Only used by YOLOF.

    Returns:
        `torch.Tensor`: Boxes with shape (N, num_classes * 4) or (N, 4), where 4 represent top_left_x, top_left_y,
        bottom_right_x, bottom_right_y.
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    delta_x_y = denorm_deltas[:, :2]
    delta_width_height = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    predicted_x_y = (rois_[:, :2] + rois_[:, 2:]) * 0.5
    predicted_width_height = rois_[:, 2:] - rois_[:, :2]

    dxy_wh = predicted_width_height * delta_x_y

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        delta_width_height = torch.clamp(delta_width_height, max=max_ratio)
    else:
        delta_width_height = delta_width_height.clamp(min=-max_ratio, max=max_ratio)

    ground_truth_x_y = predicted_x_y + dxy_wh
    ground_truth_width_height = predicted_width_height * delta_width_height.exp()
    top_left_x_y = ground_truth_x_y - (ground_truth_width_height * 0.5)
    bottom_right_x_y = ground_truth_x_y + (ground_truth_width_height * 0.5)
    bboxes = torch.cat([top_left_x_y, bottom_right_x_y], dim=-1)
    if clip_border and max_shape is not None:
        max_shape = max_shape[-2:]
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


class MaskRCNNDeltaXYWHBBoxCoder(nn.Module):
    """Delta XYWH bounding box coder.

    Following the practice in [R-CNN](https://arxiv.org/abs/1311.2524), this coder encodes a bounding box (x1, y1, x2,
    y2) into deltas (delta_x, delta_y, delta_width, delta_height) and decodes delta (delta_x, delta_y, delta_width,
    delta_height) back to original bounding box (x1, y1, x2, y2). This corresponds to (top_left_x, top_left_y,
    bottom_right_x, bottom_right_y).

    Args:
        target_means (`Sequence[float]`):
            Denormalizing means of target for delta coordinates.
        target_stds (`Sequence[float]`):
            Denormalizing standard deviation of target for delta coordinates.
        clip_border (`bool`, *optional*, defaults to `True`):
            Whether to clip the objects outside the border of the image.
        add_ctr_clamp (`bool`, *optional*, defaults to `False`):
            Whether to add center clamp, when added, the predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF.
        ctr_clamp (`int`, *optional*, defaults to 32):
            The maximum pixel shift to clamp. Only used by YOLOF.
    """

    def __init__(
        self,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        clip_border=True,
        add_ctr_clamp=False,
        ctr_clamp=32,
    ):
        super().__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to transform the `bboxes` into the `gt_bboxes`.

        Args:
            bboxes (`torch.Tensor`):
                Source boxes, e.g., object proposals.
            gt_bboxes (`torch.Tensor`):
                Target of the transformation, e.g., ground-truth boxes.

        Returns:
            `torch.Tensor`: Box transformation deltas
        """

        if bboxes.size(0) != gt_bboxes.size(0):
            raise ValueError("bboxes and gt_bboxes should have same batch size")
        if not (bboxes.size(-1) == gt_bboxes.size(-1) == 4):
            raise ValueError("bboxes and gt_bboxes should have 4 elements in last dimension")
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (`torch.Tensor`):
                Basic boxes. Shape (batch_size, N, 4) or (N, 4) with N = number of boxes.
            pred_bboxes (`torch.Tensor`):
                Encoded offsets with respect to each roi. Has shape (batch_size, N, num_classes * 4) or (batch_size, N,
                4) or (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H when rois is a grid of anchors.
                Offset encoding follows [1]_.
            max_shape (`Sequence[int]` or `torch.Tensor` or `Sequence[Sequence[int]]`, *optional*):
                Maximum bounds for boxes, specifies (H, W, C) or (H, W). If `bboxes` shape is (B, N, 4), then the
                `max_shape` should be a Sequence[Sequence[int]] and the length of `max_shape` should also be B.
            wh_ratio_clip (`float`, *optional*, defaults to 16 / 1000):
                The allowed ratio between width and height.

        Returns:
            `torch.Tensor`: Decoded boxes.
        """

        if pred_bboxes.size(0) != bboxes.size(0):
            raise ValueError("pred_bboxes and bboxes should have the same first dimension")
        if pred_bboxes.ndim == 3:
            if pred_bboxes.size(1) != bboxes.size(1):
                raise ValueError("pred_bboxes and bboxes should have the same second dimension")

        if pred_bboxes.ndim == 2:
            # single image decode
            decoded_bboxes = delta2bbox(
                bboxes,
                pred_bboxes,
                self.means,
                self.stds,
                max_shape,
                wh_ratio_clip,
                self.clip_border,
                self.add_ctr_clamp,
                self.ctr_clamp,
            )
        else:
            raise ValueError("Predicted boxes should have 2 dimensions")

        return decoded_bboxes