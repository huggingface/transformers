# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict, List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList

from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN
from .semantic_seg import build_sem_seg_head

__all__ = ["PanopticFPN"]


@META_ARCH_REGISTRY.register()
class PanopticFPN(GeneralizedRCNN):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        combine_overlap_thresh: float = 0.5,
        combine_stuff_area_thresh: float = 4096,
        combine_instances_score_thresh: float = 0.5,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold

        Other arguments are the same as :class:`GeneralizedRCNN`.
        """
        super().__init__(**kwargs)
        self.sem_seg_head = sem_seg_head
        # options when combining instance & semantic outputs
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "combine_overlap_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH,
                "combine_stuff_area_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT,
                "combine_instances_score_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH,  # noqa
            }
        )
        ret["sem_seg_head"] = build_sem_seg_head(cfg, ret["backbone"].output_shape())
        logger = logging.getLogger(__name__)
        if not cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED:
            logger.warning(
                "PANOPTIC_FPN.COMBINED.ENABLED is no longer used. "
                " model.inference(do_postprocess=) should be used to toggle postprocessing."
            )
        if cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT != 1.0:
            w = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT
            logger.warning(
                "PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT should be replaced by weights on each ROI head."
            )

            def update_weight(x):
                if isinstance(x, dict):
                    return {k: v * w for k, v in x.items()}
                else:
                    return x * w

            roi_heads = ret["roi_heads"]
            roi_heads.box_predictor.loss_weight = update_weight(roi_heads.box_predictor.loss_weight)
            roi_heads.mask_head.loss_weight = update_weight(roi_heads.mask_head.loss_weight)
        return ret

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert "sem_seg" in batched_inputs[0]
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
        ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = sem_seg_losses
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)
        proposals, _ = self.proposal_generator(images, features, None)
        detector_results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info
