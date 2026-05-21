# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoModel

# The modular converter drops `from ..<sibling>.modeling_<sibling> import X` style
# imports (they're routed through the inheritance-inlining machinery). To keep the
# `Sam3VisionEncoderOutput` and `Sam31VisionEncoderOutput` runtime references in the
# generated file, we go through the package namespace instead of the modeling submodule.
from ..sam3 import Sam3VisionEncoderOutput
from ..sam3_1_tracker_video import Sam31VisionEncoderOutput
from ..sam3_video.configuration_sam3_video import Sam3VideoConfig
from ..sam3_video.modeling_sam3_video import (
    Sam3VideoInferenceSession,
    Sam3VideoModel,
    Sam3VideoPreTrainedModel,
    Sam3VideoSegmentationOutput,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31VideoConfig(Sam3VideoConfig):
    r"""
    detector_config (`dict` or `Sam3Config`, *optional*):
        Configuration for the SAM3 detector heads (heads-only, the vision encoder is shared with the
        tracker). If not provided, a default `Sam3Config` will be used.
    tracker_config (`dict` or `Sam31TrackerVideoConfig`, *optional*):
        Configuration for the SAM3.1 video tracker. This config owns the shared TriNeck vision encoder
        (`tracker_config.vision_config: Sam31VisionConfig`). If not provided, a default
        `Sam31TrackerVideoConfig` will be used.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    low_res_mask_size (`int`, *optional*, defaults to 288):
        Size (height and width) of the low-resolution mask outputs from the tracker before upsampling to video resolution.
    score_threshold_detection (`float`, *optional*, defaults to 0.5):
        Probability threshold for detection outputs - only keep detections above this threshold.
    det_nms_thresh (`float`, *optional*, defaults to 0.1):
        IoU threshold for detection NMS (Non-Maximum Suppression).
    assoc_iou_thresh (`float`, *optional*, defaults to 0.1):
        IoU threshold for detection-to-track matching. A detection is considered "matched" to a tracklet if
        it overlaps with the tracklet above this threshold. Often a loose threshold like 0.1.
    trk_assoc_iou_thresh (`float`, *optional*, defaults to 0.5):
        IoU threshold for detection-to-track matching, used to determine whether a masklet is "unmatched"
        by any detections. Often a stricter threshold like 0.5.
    new_det_thresh (`float`, *optional*, defaults to 0.7):
        Probability threshold for a detection to be added as a new object.
    recondition_on_trk_masks (`bool`, *optional*, defaults to `True`):
        Whether to use tracked masks (True) or detection masks (False) for reconditioning. Use True when tracked
        masks are higher quality and detector serves as validation signal to strengthen memory and prevent drift.
    hotstart_delay (`int`, *optional*, defaults to 15):
        Number of frames to buffer outputs during hotstart. We hold off the outputs for `hotstart_delay`
        frames and remove tracklets based on hotstart heuristics.
    hotstart_unmatch_thresh (`int`, *optional*, defaults to 8):
        Number of unmatched frames required to remove a tracklet during hotstart period.
    hotstart_dup_thresh (`int`, *optional*, defaults to 8):
        Number of overlapping frames required to remove a duplicate tracklet during hotstart period.
    suppress_unmatched_only_within_hotstart (`bool`, *optional*, defaults to `True`):
        Whether to suppress masks only within hotstart period. If False, we can suppress masks even if
        they start before hotstart period.
    init_trk_keep_alive (`int`, *optional*, defaults to 30):
        Initial keep-alive counter for new tracks.
    max_trk_keep_alive (`int`, *optional*, defaults to 30):
        Maximum keep-alive counter value. Tracks with matched detections get their counter increased up to this value.
    min_trk_keep_alive (`int`, *optional*, defaults to -1):
        Minimum keep-alive counter value. Tracks with unmatched detections get their counter decreased to this value.
    suppress_overlapping_based_on_recent_occlusion_threshold (`float`, *optional*, defaults to 0.7):
        Threshold for suppressing overlapping objects based on recent occlusion. Overlapping masks with
        IoU above this threshold are suppressed based on which was most recently occluded.
    decrease_trk_keep_alive_for_empty_masklets (`bool`, *optional*, defaults to `False`):
        Whether to decrease keep-alive counter for masklets with zero area in SAM2 prediction.
    fill_hole_area (`int`, *optional*, defaults to 16):
        Minimum area (in pixels) for filling holes in masks and removing small sprinkles.
    max_num_objects (`int`, *optional*, defaults to 10000):
        Maximum number of objects to track. Default 10000 effectively turns off this limit.
    recondition_every_nth_frame (`int`, *optional*, defaults to 16):
        Frequency of mask reconditioning (in frames). Set to 0 to disable reconditioning.
    high_conf_thresh (`float`, *optional*, defaults to 0.8):
        High confidence threshold for reconditioning. Only detections above this threshold can recondition tracklets.
    high_iou_thresh (`float`, *optional*, defaults to 0.8):
        High IoU threshold for reconditioning. Only detections with IoU above this threshold can recondition tracklets.

    Example:
    ```python
    >>> from transformers import Sam31VideoConfig, Sam31VideoModel

    >>> configuration = Sam31VideoConfig()
    >>> configuration.image_size = 1008
    >>> model = Sam31VideoModel(configuration)
    ```
    """

    model_type = "sam3_1_video"

    def __post_init__(self, **kwargs):
        if self.detector_config is None:
            self.detector_config = CONFIG_MAPPING["sam3"]()
            logger.info("detector_config is None. Initializing the Sam3Config with default values.")
        if isinstance(self.detector_config, dict):
            self.detector_config["model_type"] = self.detector_config.get("model_type", "sam3")
            self.detector_config = CONFIG_MAPPING[self.detector_config["model_type"]](**self.detector_config)

        if self.tracker_config is None:
            self.tracker_config = CONFIG_MAPPING["sam3_1_tracker_video"]()
            logger.info("tracker_config is None. Initializing the Sam31TrackerVideoConfig with default values.")
        if isinstance(self.tracker_config, dict):
            self.tracker_config["model_type"] = self.tracker_config.get("model_type", "sam3_1_tracker_video")
            self.tracker_config = CONFIG_MAPPING[self.tracker_config["model_type"]](**self.tracker_config)
        # Skip Sam3VideoConfig.__post_init__ (which would re-default to non-3.1 sub-configs)
        # and call straight into PreTrainedConfig.
        PreTrainedConfig.__post_init__(self, **kwargs)

    @property
    def image_size(self):
        """Image size for the video model.

        Sourced from the tracker's vision config, which owns the canonical (shared) TriNeck encoder.
        """
        return self.tracker_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Recursively propagate the image size to detector and tracker configs."""
        self.detector_config.image_size = value
        self.tracker_config.image_size = value


class Sam31VideoInferenceSession(Sam3VideoInferenceSession):
    pass


@auto_docstring(custom_intro="Base class for the SAM3.1 Video model's output.")
@dataclass
class Sam31VideoSegmentationOutput(Sam3VideoSegmentationOutput):
    pass


class Sam31VideoPreTrainedModel(Sam3VideoPreTrainedModel):
    config_class = Sam31VideoConfig
    base_model_prefix = "sam3_1_video"


@auto_docstring
class Sam31VideoModel(Sam3VideoModel):
    r"""Full SAM 3.1 video PCS (phrase-conditioned segmentation) model.

    Composition compared with `Sam3VideoModel`:

      * `detector_model` is a `Sam3Model` instantiated with `remove_vision_encoder=True`.
        SAM3.1 reuses the same DETR / CLIP-text / dot-product-scoring detection heads as
        SAM3 unchanged; only the vision encoder differs. The detector is fed
        `vision_embeds` pre-computed from the shared TriNeck (see
        `_tri_neck_to_sam3_view`).
      * `tracker_model` is a `Sam31TrackerVideoModel` that **keeps** its TriNeck
        `vision_encoder` (i.e. `remove_vision_encoder=False`). The TriNeck lives once at
        `self.tracker_model.vision_encoder`, is run once per frame, and its three FPN
        streams (`sam3_fpn_*`, `interactive_fpn_*`, `propagation_fpn_*`) are routed to
        the detector + tracker.
      * There is **no** `self.tracker_neck` (the TriNeck already emits the tracker
        streams, so the SAM3-style second FPN is unnecessary).
    """

    config_class = Sam31VideoConfig

    def __init__(self, config: Sam31VideoConfig):
        Sam31VideoPreTrainedModel.__init__(self, config)
        self.config = config
        # Detector heads only: the TriNeck lives on the tracker (shared).
        self.detector_model = AutoModel.from_config(config.detector_config, remove_vision_encoder=True)
        # Tracker owns the shared TriNeck vision encoder (Sam31VisionModel).
        self.tracker_model = AutoModel.from_config(config.tracker_config)
        self.low_res_mask_size = config.low_res_mask_size
        self.score_threshold_detection = config.score_threshold_detection
        self.det_nms_thresh = config.det_nms_thresh
        self.assoc_iou_thresh = config.assoc_iou_thresh
        self.trk_assoc_iou_thresh = config.trk_assoc_iou_thresh
        self.new_det_thresh = config.new_det_thresh
        self.recondition_on_trk_masks = config.recondition_on_trk_masks
        # hotstart parameters
        self.hotstart_delay = config.hotstart_delay
        self.hotstart_unmatch_thresh = config.hotstart_unmatch_thresh
        self.hotstart_dup_thresh = config.hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = config.suppress_unmatched_only_within_hotstart
        self.init_trk_keep_alive = config.init_trk_keep_alive
        self.max_trk_keep_alive = config.max_trk_keep_alive
        self.min_trk_keep_alive = config.min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            config.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.decrease_trk_keep_alive_for_empty_masklets = config.decrease_trk_keep_alive_for_empty_masklets
        self.fill_hole_area = config.fill_hole_area
        self.eval()

        self.max_num_objects = config.max_num_objects
        self.recondition_every_nth_frame = config.recondition_every_nth_frame
        self.high_conf_thresh = config.high_conf_thresh
        self.high_iou_thresh = config.high_iou_thresh

        self.post_init()

    def _tri_neck_to_sam3_view(self, tri_neck_out: Sam31VisionEncoderOutput) -> Sam3VisionEncoderOutput:
        r"""Adapt a TriNeck `Sam31VisionEncoderOutput` into a `Sam3VisionEncoderOutput`
        that the SAM3 detector heads expect.

        The TriNeck deliberately blocks access to the SAM3-style flat `fpn_*` fields
        (they raise `AttributeError`) so detector code that hasn't been ported to the
        three-stream layout fails loudly. This helper builds a view on the `sam3_*`
        stream that exposes those fields, so `Sam3Model.forward(vision_embeds=...)` can
        consume it unchanged.

        SAM3 image vs SAM3.1 multiplex have different FPN cardinalities:

        * SAM3 image uses `scale_factors=[4.0, 2.0, 1.0, 0.5]` (4 levels).
          `Sam3Model.forward` does `fpn_hidden_states[:-1]` (the implicit equivalent of
          Meta's `scalp=1`), giving the 3 levels `[288, 144, 72]` consumed by the
          geometry encoder, the DETR encoder (last level), and the mask decoder.
        * SAM3.1 TriNeck uses `scale_factors=[4.0, 2.0, 1.0]` (3 levels). The Meta
          `SAM3VLBackboneTri` is built with `scalp=0`, so all 3 levels reach the
          detector. To reuse `Sam3Model.forward` unchanged, we append a dummy zero
          tensor at the end so the existing `[:-1]` strip leaves `[288, 144, 72]`
          intact. The dummy is dropped before it touches any module, so it never
          affects numerics.
        """
        sam3_feats = list(tri_neck_out.sam3_fpn_hidden_states)
        sam3_pos = list(tri_neck_out.sam3_fpn_position_encoding)
        # Append a dummy level only when the TriNeck emits exactly 3 levels (SAM3.1).
        # If a future config returns 4 levels we should not pad.
        if len(sam3_feats) == 3:
            last_feat = sam3_feats[-1]
            sam3_feats.append(torch.zeros_like(last_feat))
            sam3_pos.append(torch.zeros_like(sam3_pos[-1]))
        return Sam3VisionEncoderOutput(
            last_hidden_state=tri_neck_out.last_hidden_state,
            fpn_hidden_states=tuple(sam3_feats),
            fpn_position_encoding=tuple(sam3_pos),
            hidden_states=tri_neck_out.hidden_states,
            attentions=tri_neck_out.attentions,
        )

    def get_vision_features_for_tracker(self, vision_embeds: Sam31VisionEncoderOutput):
        r"""Project the TriNeck interactive + propagation streams into the flattened
        per-stream FPN lists the tracker pipeline expects, and return the interactive
        stream (used by the inherited SAM3-video orchestration for SAM-click conditioning
        and memory encoding).

        Differences vs. `Sam3VideoModel.get_vision_features_for_tracker`:

          * No `tracker_neck` is run. The TriNeck already produced both streams in
            `vision_embeds`; we just route them through the tracker's
            `_tri_neck_to_tracker_fpn` adapter, which handles the per-stream
            `mask_decoder.conv_s0/s1` (interactive) and `propagation_mask_decoder.conv_s0/s1`
            (propagation) projections and the HWxNxC flattening.
        """
        # Interactive stream: used by the inherited memory-encoder / SAM-click path
        # (Sam3VideoModel.run_memory_encoder etc.) via the returned values below.
        interactive_feats, interactive_pos = self.tracker_model._tri_neck_to_tracker_fpn(vision_embeds, "interactive")
        return interactive_feats, interactive_pos

    def _cache_tracker_vision_features(
        self,
        inference_session: Sam31VideoInferenceSession,
        frame_idx: int,
        vision_embeds: Sam31VisionEncoderOutput,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        r"""Compute the propagation + interactive FPN streams from the cached TriNeck
        output and write them into the session cache under the keys the tracker reads in
        `Sam31TrackerVideoModel._prepare_vision_features`. Returns the interactive
        stream so the inherited SAM3 video orchestration (which only knows about a single
        stream) keeps working.
        """
        propagation_feats, propagation_pos = self.tracker_model._tri_neck_to_tracker_fpn(vision_embeds, "propagation")
        interactive_feats, interactive_pos = self.tracker_model._tri_neck_to_tracker_fpn(vision_embeds, "interactive")
        inference_session.cache.cache_vision_features(
            frame_idx,
            {
                "vision_feats": propagation_feats,
                "vision_pos_embeds": propagation_pos,
                "interactive_vision_feats": interactive_feats,
                "interactive_vision_pos_embeds": interactive_pos,
            },
        )
        return interactive_feats, interactive_pos

    def _det_track_one_frame(
        self,
        inference_session: Sam31VideoInferenceSession,
        frame_idx: int,
        reverse: bool,
        streaming: bool = False,
    ):
        r"""SAM3.1 variant of `Sam3VideoModel._det_track_one_frame`.

        Only the vision-encoder call and the shared-feature wiring differ; the detection,
        propagation, planning, execution, and output-build steps are unchanged from the
        SAM3 video pipeline.
        """
        pixel_values = inference_session.get_frame(frame_idx).unsqueeze(0)

        # Run the shared TriNeck once: produces sam3 (detector), interactive (cond/click),
        # and propagation (memory) streams in a single forward pass.
        tri_neck_out: Sam31VisionEncoderOutput = self.tracker_model.vision_encoder(
            pixel_values,
            need_sam3_out=True,
            need_interactive_out=True,
            need_propagation_out=True,
            return_dict=True,
        )
        # Adapter: expose the sam3 stream as a plain Sam3VisionEncoderOutput for the
        # heads-only detector.
        vision_embeds = self._tri_neck_to_sam3_view(tri_neck_out)

        # Step 1: run detection for all prompts (efficiently reusing vision embeddings)
        all_detections = self.run_detection(
            inference_session=inference_session,
            vision_embeds=vision_embeds,
        )
        det_out, det_idx_to_prompt_id = self._merge_detections_from_prompts(all_detections, inference_session)

        # Step 1b: cache the tracker FPN streams so the tracker propagation /
        # memory-encoder paths reuse them instead of re-running the vision encoder.
        self._cache_tracker_vision_features(inference_session, frame_idx, tri_neck_out)

        # Step 2: propagate masklets for one frame (no new memory encoding yet).
        tracker_low_res_masks_global, tracker_obj_scores_global = self.run_tracker_propagation(
            inference_session=inference_session, frame_idx=frame_idx, reverse=reverse
        )

        # Step 3: planning phase (hotstart / NMS / det<->trk association / reconditioning).
        tracker_update_plan, tracker_metadata_new = self.run_tracker_update_planning_phase(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            det_idx_to_prompt_id=det_idx_to_prompt_id,
            streaming=streaming,
        )

        # Step 4: execution phase (memory encoding for new conditioning frames).
        self.run_tracker_update_execution_phase(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_update_plan=tracker_update_plan,
        )

        # Step 5: build outputs for this frame.
        reconditioned_obj_ids = tracker_update_plan["reconditioned_obj_ids"]
        obj_id_to_mask = self.build_outputs(
            inference_session=inference_session,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = inference_session.obj_ids
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )

        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            tracker_obj_scores_global,
        )


__all__ = [
    "Sam31VideoConfig",
    "Sam31VideoModel",
    "Sam31VideoPreTrainedModel",
    "Sam31VideoInferenceSession",
    "Sam31VideoSegmentationOutput",
]
