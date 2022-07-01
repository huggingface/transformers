# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torchvision
from torch.nn.modules.utils import _pair


def crop_and_resize(masks, bboxes, out_shape, inds, device="cpu", interpolation="bilinear", binarize=True):
    """See [`BaseInstanceMasks.crop_and_resize`].

    Source:
    https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/core/mask/structures.py#L333.

    """
    if len(masks) == 0:
        empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
        return empty_masks

    # convert bboxes to tensor
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).to(device=device)
    if isinstance(inds, np.ndarray):
        inds = torch.from_numpy(inds).to(device=device)

    num_bbox = bboxes.shape[0]
    fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
    rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_bbox > 0:
        gt_masks_th = torch.from_numpy(masks).to(device).index_select(0, inds).to(dtype=rois.dtype)
        targets = torchvision.ops.roi_align(
            gt_masks_th[:, None, :, :], rois, output_size=out_shape, spatial_scale=1.0, sampling_ratio=0, aligned=True
        ).squeeze(1)
        if binarize:
            resized_masks = (targets >= 0.5).cpu().numpy()
        else:
            resized_masks = targets.cpu().numpy()
    else:
        resized_masks = []
    return resized_masks


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv >>> import mmdet >>> from mmdet.core.mask import BitmapMasks >>> from
        mmdet.core.mask.mask_target import * >>> H, W = 17, 18 >>> cfg = mmcv.Config({'mask_size': (13, 14)}) >>> rng =
        np.random.RandomState(0) >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image >>>
        pos_proposals_list = [ >>> torch.Tensor([ >>> [ 7.2425, 5.5929, 13.9414, 14.9541], >>> [ 7.3241, 3.6170,
        16.3850, 15.3102], >>> ]), >>> torch.Tensor([ >>> [ 4.8448, 6.4010, 7.0314, 9.7681], >>> [ 5.9790, 2.6989,
        7.4416, 4.8580], >>> [ 0.0000, 0.0000, 0.1398, 9.8232], >>> ]), >>> ] >>> # Corresponding class index for each
        proposal for each image >>> pos_assigned_gt_inds_list = [ >>> torch.LongTensor([7, 0]), >>>
        torch.LongTensor([5, 4, 1]), >>> ] >>> # Ground truth mask for each true object for each image >>>
        gt_masks_list = [ >>> BitmapMasks(rng.rand(8, H, W), height=H, width=W), >>> BitmapMasks(rng.rand(6, H, W),
        height=H, width=W), >>> ] >>> mask_targets = mask_target( >>> pos_proposals_list, pos_assigned_gt_inds_list,
        >>> gt_masks_list, cfg) >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.

    Example:
        >>> import mmcv >>> import mmdet >>> from mmdet.core.mask import BitmapMasks >>> from
        mmdet.core.mask.mask_target import * # NOQA >>> H, W = 32, 32 >>> cfg = mmcv.Config({'mask_size': (7, 11)}) >>>
        rng = np.random.RandomState(0) >>> # Masks for each ground truth box (relative to the image) >>> gt_masks_data
        = rng.rand(3, H, W) >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W) >>> # Predicted positive boxes
        in one image >>> pos_proposals = torch.FloatTensor([ >>> [ 16.2, 5.5, 19.9, 20.9], >>> [ 17.3, 13.6, 19.3,
        19.3], >>> [ 14.8, 16.4, 17.0, 23.7], >>> [ 0.0, 0.0, 16.0, 16.0], >>> [ 4.0, 0.0, 20.0, 16.0], >>> ]) >>> #
        For each predicted proposal, its assignment to a gt mask >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2,
        1, 1]) >>> mask_targets = mask_target_single( >>> pos_proposals, pos_assigned_gt_inds, gt_masks, cfg) >>>
        assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    device = pos_proposals.device
    mask_size = _pair(cfg["mask_size"])
    binarize = not cfg.get("soft_mask_target", False)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        print("Gt_masks shape:", gt_masks.shape)
        # TODO verify this replacement is correct
        # maxh, maxw = gt_masks.height, gt_masks.width
        maxh, maxw = tuple(gt_masks.shape[1:])
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = crop_and_resize(
            gt_masks.numpy(), proposals_np, mask_size, device=device, inds=pos_assigned_gt_inds, binarize=binarize
        )

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0,) + mask_size)

    return mask_targets
