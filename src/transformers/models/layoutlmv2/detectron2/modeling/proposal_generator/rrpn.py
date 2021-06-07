# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
from typing import Dict, List
import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.memory import retry_if_cuda_oom

from ..box_regression import Box2BoxTransformRotated
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn import RPN

logger = logging.getLogger(__name__)


def find_top_rrpn_proposals(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 5).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size(float): minimum proposal box side length in pixels (absolute units wrt
            input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 5

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = RotatedBoxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = (boxes[keep], scores_per_img[keep], level_ids[keep])

        keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


@PROPOSAL_GENERATOR_REGISTRY.register()
class RRPN(RPN):
    """
    Rotated Region Proposal Network described in :paper:`RRPN`.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.anchor_boundary_thresh >= 0:
            raise NotImplementedError(
                "anchor_boundary_thresh is a legacy option not implemented for RRPN."
            )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["box2box_transform"] = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        return ret

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[RotatedBoxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[RotatedBoxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across feature maps. Label values are in {-1, 0, 1},
                with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            list[Tensor]:
                i-th element is a Nx5 tensor, where N is the total number of anchors across
                feature maps.  The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as 1.
        """
        anchors = RotatedBoxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            """
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou_rotated)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.no_grad()
    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rrpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training,
        )
