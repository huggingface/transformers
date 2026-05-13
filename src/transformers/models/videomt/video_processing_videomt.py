# Copyright 2026 The HuggingFace Inc. team.
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
"""Video processor class for Videomt."""

from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import is_torch_available, requires_backends
from ...video_processing_utils import BaseVideoProcessor


if is_torch_available():
    import torch
    import torch.nn.functional as F


def check_segment_validity(
    mask_labels: "torch.Tensor",
    mask_probs: "torch.Tensor",
    query_idx: int,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> tuple[bool, "torch.Tensor"]:
    """
    Checks whether a predicted query produces a valid panoptic segment.

    Args:
        mask_labels (`torch.Tensor`):
            Tensor of shape `(height, width)` containing the winning query index for each pixel.
        mask_probs (`torch.Tensor`):
            Tensor of shape `(num_queries, height, width)` containing per-query mask probabilities.
        query_idx (`int`):
            Index of the query to validate.
        mask_threshold (`float`, *optional*, defaults to 0.5):
            Threshold used to binarize the query mask probabilities.
        overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
            Minimum overlap ratio required between the assigned query area and the original query mask area.

    Returns:
        `tuple[bool, torch.Tensor]`: A tuple containing whether the segment is valid and the final boolean mask for
        that segment.
    """
    query_mask = mask_labels == query_idx
    query_mask_area = query_mask.sum()

    original_mask = mask_probs[query_idx] >= mask_threshold
    original_area = original_mask.sum()

    final_mask = query_mask & original_mask
    final_mask_area = final_mask.sum()

    mask_exists = query_mask_area > 0 and original_area > 0 and final_mask_area > 0

    if mask_exists:
        area_ratio = query_mask_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, final_mask


def compute_segments(
    mask_probs: "torch.Tensor",
    pred_scores: "torch.Tensor",
    pred_labels: "torch.Tensor",
    label_ids_to_fuse: set[int] | None,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: tuple[int, int] | None = None,
) -> tuple["torch.Tensor", list[dict[str, int | float]]]:
    """
    Converts per-query mask predictions into a panoptic segmentation map.

    Args:
        mask_probs (`torch.Tensor`):
            Tensor of shape `(num_queries, height, width)` containing per-query mask logits.
        pred_scores (`torch.Tensor`):
            Tensor of shape `(num_queries,)` containing the confidence score of each predicted query.
        pred_labels (`torch.Tensor`):
            Tensor of shape `(num_queries,)` containing the predicted class ID of each query.
        label_ids_to_fuse (`set[int]`, *optional*):
            Label IDs that should be fused across disconnected regions.
        mask_threshold (`float`, *optional*, defaults to 0.5):
            Threshold used to binarize the query mask probabilities.
        overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
            Minimum overlap ratio required to keep a predicted segment.
        target_size (`tuple[int, int]`, *optional*):
            Final `(height, width)` of the segmentation map. If unset, uses the spatial size of `mask_probs`.

    Returns:
        `tuple[torch.Tensor, list[dict[str, int | float]]]`: The panoptic segmentation map and the metadata for each
        predicted segment.
    """
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device) - 1
    segments: list[dict] = []

    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    current_segment_id = 0
    stuff_memory_list: dict[int, int] = {}

    for query_idx in range(pred_labels.shape[0]):
        pred_class = pred_labels[query_idx].item()

        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, query_idx, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if label_ids_to_fuse and pred_class in label_ids_to_fuse:
            if pred_class in stuff_memory_list:
                segmentation[final_mask] = stuff_memory_list[pred_class]
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id

        segmentation[final_mask] = current_segment_id
        segment_score = round(pred_scores[query_idx].item(), 6)
        segments.append(
            {
                "id": current_segment_id,
                "label_id": pred_class,
                "score": segment_score,
            }
        )
        current_segment_id += 1
    return segmentation, segments


class VideomtVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 640, "width": 640}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    model_input_names = ["pixel_values_videos"]

    def _resize_mask_logits(
        self,
        masks_queries_logits: "torch.Tensor",
        target_sizes: list[tuple[int, int]],
    ) -> list["torch.Tensor"]:
        """Interpolates mask logits to each frame's original resolution."""
        resized = []
        for idx, original_size in enumerate(target_sizes):
            upsampled = F.interpolate(
                masks_queries_logits[idx][None, ...],
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )[0]
            resized.append(upsampled)
        return resized

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
    ) -> list["torch.Tensor"]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into semantic segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.

        Returns:
            `list[torch.Tensor]`: A list of tensors, each of shape `(height, width)`, where each value is the
            predicted class index for the corresponding pixel.
        """
        requires_backends(self, ["torch"])

        masks_queries_logits = outputs.masks_queries_logits  # [num_frames, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [num_frames, num_queries, num_classes+1]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.float().softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.float().sigmoid()

        segmentation_logits = torch.matmul(masks_classes.transpose(1, 2), masks_probs.flatten(2))
        segmentation_logits = segmentation_logits.reshape(
            masks_probs.shape[0], masks_classes.shape[-1], masks_probs.shape[-2], masks_probs.shape[-1]
        )

        output_logits = self._resize_mask_logits(segmentation_logits, target_sizes)

        return [logit.argmax(dim=0) for logit in output_logits]

    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into instance segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.
            threshold (`float`, *optional*, defaults to 0.5):
                Minimum combined score to keep an instance.

        Returns:
            `list[dict]`: A list of dicts (one per frame), each containing:
                - `"segmentation"` -- A `torch.Tensor` of shape `(height, width)` with instance IDs (or -1 for background).
                - `"segments_info"` -- A list of dicts with `"id"`, `"label_id"`, and `"score"` for each instance.
        """
        requires_backends(self, ["torch"])

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        mask_probs_batch = self._resize_mask_logits(masks_queries_logits, target_sizes)

        device = masks_queries_logits.device
        num_frames = class_queries_logits.shape[0]
        num_queries = class_queries_logits.shape[-2]

        results = []

        for frame_idx in range(num_frames):
            mask_pred = mask_probs_batch[frame_idx]
            mask_class = class_queries_logits[frame_idx]

            class_probs = mask_class.float().softmax(dim=-1)[..., :-1]
            scores, pred_classes = class_probs.max(-1)
            pred_masks = mask_pred > 0

            mask_probs = mask_pred.float().sigmoid()
            mask_scores = (mask_probs.flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores * mask_scores

            segmentation = torch.full(target_sizes[frame_idx], fill_value=-1, dtype=torch.long, device=device)

            segments = []
            current_segment_id = 0
            for query_idx in range(num_queries):
                score = pred_scores[query_idx].item()

                if torch.any(pred_masks[query_idx]) and score >= threshold:
                    segmentation[pred_masks[query_idx]] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[query_idx].item(),
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
    ) -> list[dict]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into panoptic segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.
            threshold (`float`, *optional*, defaults to 0.8):
                Minimum score to keep a predicted segment.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing mask probabilities.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                Overlap threshold to merge masks into a single segment.
            label_ids_to_fuse (`set[int]`, *optional*):
                Label IDs that should be fused across disconnected regions.

        Returns:
            `list[dict]`: A list of dicts (one per frame), each containing:
                - `"segmentation"` -- A `torch.Tensor` of shape `(height, width)` with segment IDs (or -1 for background).
                - `"segments_info"` -- A list of dicts with `"id"`, `"label_id"`, and `"score"` for each segment.
        """
        requires_backends(self, ["torch"])

        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits

        num_frames = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs_batch = self._resize_mask_logits(masks_queries_logits, target_sizes)
        pred_scores_batch, pred_labels_batch = class_queries_logits.float().softmax(dim=-1).max(-1)

        results: list = []

        for frame_idx in range(num_frames):
            mask_probs = mask_probs_batch[frame_idx]
            pred_scores = pred_scores_batch[frame_idx]
            pred_labels = pred_labels_batch[frame_idx]

            if not (mask_probs.shape[0] == pred_scores.shape[0] == pred_labels.shape[0]):
                raise ValueError("mask, scores and labels must have the same shape!")

            to_keep = pred_labels.ne(num_labels) & (pred_scores > threshold)
            mask_probs = mask_probs[to_keep]
            pred_scores = pred_scores[to_keep]
            pred_labels = pred_labels[to_keep]

            if mask_probs.shape[0] <= 0:
                height, width = target_sizes[frame_idx] if target_sizes is not None else mask_probs.shape[1:]
                segmentation = torch.full(
                    (height, width), fill_value=-1, dtype=torch.long, device=masks_queries_logits.device
                )
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            segmentation, segments = compute_segments(
                mask_probs=mask_probs,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                label_ids_to_fuse=label_ids_to_fuse,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_size=target_sizes[frame_idx] if target_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["VideomtVideoProcessor"]
