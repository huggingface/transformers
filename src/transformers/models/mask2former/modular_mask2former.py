# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn

from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...utils import (
    TensorType,
    logging,
    requires_backends,
)
from ...utils.import_utils import requires
from ..maskformer.image_processing_maskformer import MaskFormerImageProcessor
from ..maskformer.image_processing_pil_maskformer import MaskFormerImageProcessorPil
from .image_processing_mask2former import (
    compute_segments,
    convert_segmentation_to_rle,
    remove_low_and_no_objects,
)


logger = logging.get_logger(__name__)


class Mask2FormerImageProcessor(MaskFormerImageProcessor):
    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
        return_segmentation_scores: bool = False,
        return_traceable_outputs: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_segmentation_scores (`bool`, *optional*, defaults to `False`):
                Whether to return segmentation scores alongside the segmentation map. When `True`, each element of
                the returned list is a [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation`
                (class IDs, shape `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`).
            return_traceable_outputs (`bool`, *optional*, defaults to `False`):
                If set to `True`, a tuple of tensors is returned instead of a list, see the returns section below.
                All target sizes must be equal in that case.

        Returns:
            `list[torch.Tensor]` or `list[SemanticSegmentationPostProcessorOutput]`: When
            `return_segmentation_scores=False` (default), a list of length `batch_size` where each item is a
            segmentation map of shape `(height, width)` with class IDs. When `return_segmentation_scores=True`,
            a list of [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation` (class IDs, shape
            `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`). In both cases,
            `(height, width)` corresponds to the target size (if `target_sizes` is specified).

            When `return_traceable_outputs=True`, a tuple `(semantic_map,)` of shape `(batch_size, height, width)`,
            extended with the segmentation scores of shape `(batch_size, num_classes, height, width)` if
            `return_segmentation_scores=True`. The maps are already resized to the target size, so they can be used
            as is; passing `segmentation_scores` to
            [`~Mask2FormerImageProcessor.build_semantic_segmentation_outputs`] outside of the traced code gives the
            same output as `return_traceable_outputs=False`.
        """
        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        batch_size = class_queries_logits.shape[0]

        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if return_traceable_outputs and len(set(target_sizes)) != 1:
                raise ValueError("All target sizes must be identical when `return_traceable_outputs=True`.")

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width), without the null class
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
        segmentation_scores = masks_classes.transpose(1, 2) @ masks_probs.flatten(start_dim=2)
        segmentation_scores = segmentation_scores.unflatten(dim=-1, sizes=masks_probs.shape[-2:])

        # Logits are resized before the argmax only in the traceable branch,
        # `build_semantic_segmentation_outputs` resizes them otherwise
        if return_traceable_outputs:
            if target_sizes is not None:
                segmentation_scores = torch.nn.functional.interpolate(
                    segmentation_scores, size=target_sizes[0], mode="bilinear", align_corners=False
                )
            semantic_map = segmentation_scores.argmax(dim=1)
            return (semantic_map, segmentation_scores) if return_segmentation_scores else (semantic_map,)

        return self.build_semantic_segmentation_outputs(
            segmentation_scores=segmentation_scores,
            target_sizes=target_sizes,
            return_segmentation_scores=return_segmentation_scores,
        )

    def build_semantic_segmentation_outputs(
        self,
        segmentation_scores: torch.Tensor,
        target_sizes: list[tuple[int, int]] | None = None,
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Builds semantic segmentation maps from the tensors returned by
        `post_process_semantic_segmentation(..., return_traceable_outputs=True)`. See
        [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] for the arguments and the returned values.
        `target_sizes` must be left to `None` if the logits were already resized by
        `post_process_semantic_segmentation`.
        """
        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

        if target_sizes is None or len(set(target_sizes)) == 1:
            if target_sizes is not None:
                segmentation_scores = torch.nn.functional.interpolate(
                    segmentation_scores, size=target_sizes[0], mode="bilinear", align_corners=False
                )
            semantic_map = segmentation_scores.argmax(dim=1)
        else:
            segmentation_scores = [
                torch.nn.functional.interpolate(
                    image_scores.unsqueeze(dim=0), size=target_size, mode="bilinear", align_corners=False
                )[0]
                for image_scores, target_size in zip(segmentation_scores, target_sizes)
            ]
            semantic_map = [image_scores.argmax(dim=0) for image_scores in segmentation_scores]

        if not return_segmentation_scores:
            return list(semantic_map)

        return [
            SemanticSegmentationPostProcessorOutput(
                data={"segmentation": segmentation, "segmentation_scores": image_scores}
            )
            for segmentation, image_scores in zip(semantic_map, segmentation_scores)
        ]

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
        return_binary_maps: bool | None = False,
        return_traceable_outputs: bool = False,
    ) -> list[dict]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into instance segmentation predictions.
        Only supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
        to `True` to get the correct segmentation result.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.
            return_binary_maps (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps
                (one per detected instance).
            return_traceable_outputs (`bool`, *optional*, defaults to `False`):
                If set to `True`, a tuple of tensors with static shapes is returned instead of a list of
                dictionaries, see the returns section below. All target sizes must be equal in that case.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id`, or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`, or a tensor of shape `(num_instances, height, width)` if return_binary_maps is set to `True`.
              Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.

            When `return_traceable_outputs=True`, a tuple `(masks, scores, classes, keep_instances)` of tensors with a
            static `num_queries` dimension: `masks` of shape `(batch_size, num_queries, height, width)` with binary
            instance masks, `scores` and `classes` of shape `(batch_size, num_queries)`, and `keep_instances` of shape
            `(batch_size, num_queries)`, `True` for the instances that are above `threshold` and have a non-empty mask.
            Everything up to that tuple is traceable, so passing it to
            [`~Mask2FormerImageProcessor.build_instance_segmentation_outputs`] outside of the traced code gives the
            same output as `return_traceable_outputs=False`.
        """
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        batch_size = class_queries_logits.shape[0]
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if return_traceable_outputs and len(set(target_sizes)) != 1:
                raise ValueError("All target sizes must be identical when `return_traceable_outputs=True`.")

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )
        height, width = masks_queries_logits.shape[-2:]

        # Remove the null class `[..., :-1]` and keep the `num_queries` highest scoring (query, class) pairs
        scores = torch.nn.functional.softmax(class_queries_logits, dim=-1)[..., :-1]
        scores, topk_indices = scores.flatten(1, 2).topk(num_queries, dim=-1, sorted=False)
        classes = topk_indices % num_classes
        query_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")

        mask_logits = masks_queries_logits.gather(
            dim=1, index=query_indices[..., None, None].expand(-1, -1, height, width)
        )
        masks = (mask_logits > 0).to(mask_logits.dtype)

        # Calculate average mask prob
        mask_scores = (mask_logits.sigmoid().flatten(2) * masks.flatten(2)).sum(-1) / (masks.flatten(2).sum(-1) + 1e-6)
        scores = scores * mask_scores

        if return_traceable_outputs and target_sizes is not None:
            masks = torch.nn.functional.interpolate(masks, size=target_sizes[0], mode="nearest")

        # Discard instances with a low score or an empty mask
        keep_instances = (scores >= threshold) & masks.flatten(2).any(dim=-1)

        if return_traceable_outputs:
            return masks, scores, classes, keep_instances

        return self.build_instance_segmentation_outputs(
            masks=masks,
            scores=scores,
            classes=classes,
            keep_instances=keep_instances,
            target_sizes=target_sizes,
            return_coco_annotation=return_coco_annotation,
            return_binary_maps=return_binary_maps,
        )

    def build_instance_segmentation_outputs(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        keep_instances: torch.Tensor,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
        return_binary_maps: bool | None = False,
    ) -> list[dict]:
        """
        Builds instance segmentation predictions from the tensors returned by
        `post_process_instance_segmentation(..., return_traceable_outputs=True)`. See
        [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] for the arguments and the returned values.
        `target_sizes` must be left to `None` if the masks were already resized by
        `post_process_instance_segmentation`.
        """
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

        results: list[dict[str, TensorType]] = []
        for idx, (image_masks, image_scores, image_classes, image_keep) in enumerate(
            zip(masks, scores, classes, keep_instances)
        ):
            image_masks = image_masks[image_keep]
            image_scores = image_scores[image_keep]
            image_classes = image_classes[image_keep]

            # Resizing is done after filtering, interpolating all masks to the target size is expensive
            if target_sizes is not None and image_masks.shape[0] != 0:
                image_masks = torch.nn.functional.interpolate(
                    image_masks.unsqueeze(dim=0), size=target_sizes[idx], mode="nearest"
                )[0]
                # Masks can become empty when downsampled
                non_empty_masks = image_masks.flatten(1).any(dim=-1)
                image_masks = image_masks[non_empty_masks]
                image_scores = image_scores[non_empty_masks]
                image_classes = image_classes[non_empty_masks]

            height, width = target_sizes[idx] if target_sizes is not None else image_masks.shape[-2:]
            segmentation = torch.zeros((height, width)) - 1
            segments = []
            for segment_id in range(image_masks.shape[0]):
                segmentation[image_masks[segment_id] == 1] = segment_id
                segments.append(
                    {
                        "id": segment_id,
                        "label_id": image_classes[segment_id].item(),
                        "was_fused": False,
                        "score": round(image_scores[segment_id].item(), 6),
                    }
                )

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instance maps
            if return_binary_maps and image_masks.shape[0] != 0:
                segmentation = image_masks

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
        return_traceable_outputs: bool = False,
    ) -> list[dict]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentationOutput`]):
                The outputs from [`Mask2FormerForUniversalSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.
            return_traceable_outputs (`bool`, *optional*, defaults to `False`):
                If set to `True`, a tuple of tensors with static shapes is returned instead of a list of
                dictionaries, see the returns section below. All target sizes must be equal in that case and
                `label_ids_to_fuse` is unused.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.

            When `return_traceable_outputs=True`, a tuple `(masks, scores, classes, keep_instances)` of tensors with a
            static `num_queries` dimension: `masks` of shape `(batch_size, num_queries, height, width)` with mask
            probabilities, `scores` and `classes` of shape `(batch_size, num_queries)`, and `keep_instances` of shape
            `(batch_size, num_queries)`, `True` for the instances that are above `threshold` and do not predict the
            null class.
            Everything up to that tuple is traceable, so passing it to
            [`~Mask2FormerImageProcessor.build_panoptic_segmentation_outputs`] outside of the traced code gives the
            same output as `return_traceable_outputs=False`.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_classes = class_queries_logits.shape[-1] - 1

        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if return_traceable_outputs and len(set(target_sizes)) != 1:
                raise ValueError("All target sizes must be identical when `return_traceable_outputs=True`.")

        # Scale back to preprocessed image size - (384, 384) for all models
        masks = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        ).sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        scores, classes = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Masks are resized before filtering only in the traceable branch, `build_panoptic_segmentation_outputs`
        # resizes the remaining masks otherwise
        if return_traceable_outputs and target_sizes is not None:
            masks = torch.nn.functional.interpolate(masks, size=target_sizes[0], mode="bilinear", align_corners=False)

        # Discard instances with a low score or predicting the null class
        keep_instances = classes.ne(num_classes) & (scores > threshold)

        if return_traceable_outputs:
            return masks, scores, classes, keep_instances

        return self.build_panoptic_segmentation_outputs(
            masks=masks,
            scores=scores,
            classes=classes,
            keep_instances=keep_instances,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            label_ids_to_fuse=label_ids_to_fuse,
            target_sizes=target_sizes,
        )

    def build_panoptic_segmentation_outputs(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        keep_instances: torch.Tensor,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """
        Builds panoptic segmentation predictions from the tensors returned by
        `post_process_panoptic_segmentation(..., return_traceable_outputs=True)`. See
        [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] for the arguments and the returned values.
        `target_sizes` must be left to `None` if the masks were already resized by
        `post_process_panoptic_segmentation`.
        """
        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        if target_sizes is not None:
            if isinstance(target_sizes, (torch.Tensor, np.ndarray)):
                target_sizes = target_sizes.tolist()
            target_sizes = [tuple(size) for size in target_sizes]

        results: list[dict[str, TensorType]] = []
        for idx, (image_masks, image_scores, image_classes, image_keep) in enumerate(
            zip(masks, scores, classes, keep_instances)
        ):
            image_masks = image_masks[image_keep]
            image_scores = image_scores[image_keep]
            image_classes = image_classes[image_keep]
            target_size = target_sizes[idx] if target_sizes is not None else None

            # No mask found
            if image_masks.shape[0] <= 0:
                height, width = target_size if target_size is not None else image_masks.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item, resizing the masks here keeps the
            # interpolation off the queries that were filtered out
            segmentation, segments = compute_segments(
                mask_probs=image_masks,
                pred_scores=image_scores,
                pred_labels=image_classes,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


@requires(backends=("torch",))
class Mask2FormerImageProcessorPil(MaskFormerImageProcessorPil):
    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_segmentation_scores (`bool`, *optional*, defaults to `False`):
                Whether to return segmentation scores alongside the segmentation map. When `True`, each element of
                the returned list is a [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation`
                (class IDs, shape `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`).

        Returns:
            `list[torch.Tensor]` or `list[SemanticSegmentationPostProcessorOutput]`: When
            `return_segmentation_scores=False` (default), a list of length `batch_size` where each item is a
            segmentation map of shape `(height, width)` with class IDs. When `return_segmentation_scores=True`,
            a list of [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation` (class IDs, shape
            `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`). In both cases,
            `(height, width)` corresponds to the target size (if `target_sizes` is specified).
        """
        requires_backends(self, ["torch"])
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(
                    SemanticSegmentationPostProcessorOutput(
                        data={"segmentation": semantic_map, "segmentation_scores": resized_logits[0]}
                    )
                )
        else:
            semantic_map = segmentation.argmax(dim=1)
            semantic_segmentation = [
                SemanticSegmentationPostProcessorOutput(
                    data={"segmentation": semantic_map[i], "segmentation_scores": segmentation[i]}
                )
                for i in range(batch_size)
            ]

        if not return_segmentation_scores:
            semantic_segmentation = [item.segmentation for item in semantic_segmentation]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
        return_binary_maps: bool | None = False,
    ) -> list[dict]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into instance segmentation predictions.
        Only supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
        to `True` to get the correct segmentation result.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.
            return_binary_maps (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps
                (one per detected instance).
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id`, or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`, or a tensor of shape `(num_instances, height, width)` if return_binary_maps is set to `True`.
              Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        requires_backends(self, ["torch"])
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        device = masks_queries_logits.device
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        # Loop over items in batch size
        results: list[dict[str, TensorType]] = []

        for i in range(class_queries_logits.shape[0]):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]

            scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            segmentation = torch.zeros((384, 384)) - 1
            if target_sizes is not None:
                segmentation = torch.zeros(target_sizes[i]) - 1
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(0), size=target_sizes[i], mode="nearest"
                )[0]

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "was_fused": False,
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instance maps
            if return_binary_maps and len(instance_maps) != 0:
                segmentation = torch.stack(instance_maps, dim=0)

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentationOutput`]):
                The outputs from [`Mask2FormerForUniversalSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        requires_backends(self, ["torch"])
        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: list[dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["Mask2FormerImageProcessor", "Mask2FormerImageProcessorPil"]
