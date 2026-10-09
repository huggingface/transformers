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
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

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
    fill_holes_in_mask_scores,
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
        IoM threshold for detection NMS (the metric is IoM because `det_nms_use_iom=True`).
    det_nms_use_iom (`bool`, *optional*, defaults to `True`):
        SAM3.1 multiplex tracking uses IoM (Intersection over Minimum) instead of IoU for NMS,
        which is more aggressive at removing nested duplicates.
    assoc_iou_thresh (`float`, *optional*, defaults to 0.1):
        IoU/IoM threshold for detection-to-track matching. A detection is considered "matched" to a
        tracklet if it overlaps with the tracklet above this threshold. Often a loose threshold like
        0.1.
    trk_assoc_iou_thresh (`float`, *optional*, defaults to 0.5):
        IoU/IoM threshold for detection-to-track matching, used to determine whether a masklet is
        "unmatched" by any detections. Often a stricter threshold like 0.5.
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
    suppress_unmatched_only_within_hotstart (`bool`, *optional*, defaults to `False`):
        Whether to suppress masks only within hotstart period. If False, we can suppress masks even if
        they start before hotstart period. SAM3.1 multiplex disables this gate so that stale tracks
        (e.g. detections that lose their object) keep getting suppressed across the whole video instead
        of accumulating into ghost tracks.
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
    fill_hole_area (`int`, *optional*, defaults to 0):
        Minimum area (in pixels) for filling holes in masks and removing small sprinkles.
        SAM3.1 multiplex uses 0 (disabled) in Meta's reference config.
    suppress_det_close_to_boundary (`bool`, *optional*, defaults to `True`):
        Suppress detections whose box center is within 2.5% of the image border.
    max_num_objects (`int`, *optional*, defaults to 10000):
        Maximum number of objects to track. Default 10000 effectively turns off this limit.
    recondition_every_nth_frame (`int`, *optional*, defaults to 16):
        Frequency of mask reconditioning (in frames). Set to 0 to disable reconditioning.
    high_conf_thresh (`float`, *optional*, defaults to 0.8):
        High confidence threshold for reconditioning. Only detections above this threshold can recondition tracklets.
    high_iou_thresh (`float`, *optional*, defaults to 0.8):
        High IoU threshold for reconditioning. Ignored when `use_iom_recondition=True`; in that case
        `iom_thresh_recondition` is used instead.
    use_iom_recondition (`bool`, *optional*, defaults to `True`):
        SAM3.1 multiplex tracking uses IoM instead of IoU as the detection-to-track association
        metric, which is more robust to partial occlusions and shrinking tracked masks.
    iom_thresh_recondition (`float`, *optional*, defaults to 0.5):
        IoM threshold for reconditioning. Lower than IoU's 0.8 default because IoM scores are
        intrinsically larger when one mask is nested inside the other.
    masklet_confirmation_enable (`bool`, *optional*, defaults to `True`):
        Require consecutive detection-track matches before publishing a masklet.
    masklet_confirmation_consecutive_det_thresh (`int`, *optional*, defaults to 3):
        Consecutive matched frames required to confirm a masklet.

    Example:
    ```python
    >>> from transformers import Sam31VideoConfig, Sam31VideoModel

    >>> configuration = Sam31VideoConfig()
    >>> configuration.image_size = 1008
    >>> model = Sam31VideoModel(configuration)
    ```
    """

    model_type = "sam3_1_video"

    # Overrides for SAM3.1 multiplex tracking, matching Meta's `_create_multiplex_pcs_video_predictor`
    # in `facebook_sam3/sam3/model_builder.py`:
    #   * IoM (Intersection over Minimum) for both NMS and detection-track
    #     association/reconditioning. IoM is more aggressive than IoU at suppressing nested
    #     duplicates and more permissive at associating a partially-occluded / shrinking tracked
    #     mask with its current detection, which is what SAM3.1's multiplex tracker is tuned for.
    #   * `suppress_unmatched_only_within_hotstart=False` so stale tracks keep getting suppressed
    #     across the entire video instead of being held forever once hotstart ends.
    #
    # NOTE: Meta also lowers `score_threshold_detection` to 0.4 and `new_det_thresh` to 0.65 for
    # this predictor, but the HF detection-score distribution is slightly compressed compared to
    # Meta's (residual text-feature norm and bf16/fp32 differences), so adopting those thresholds
    # admits noticeably more false-positive tracks (mean HF precision drops by ~0.07 on the
    # `foot.mp4 / "shoe"` benchmark). We therefore keep the SAM3 defaults (0.5 / 0.7) here; the
    # values can still be overridden through the config for users who want exact Meta parity.
    det_nms_use_iom: bool = True
    use_iom_recondition: bool = True
    iom_thresh_recondition: float = 0.5
    suppress_unmatched_only_within_hotstart: bool = False
    suppress_det_close_to_boundary: bool = True
    fill_hole_area: int = 0
    masklet_confirmation_enable: bool = True
    score_threshold_detection: float = 0.4
    new_det_thresh: float = 0.65

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
    r"""
    object_ids (`list[int]`, *optional*):
        List of object IDs being tracked in the current frame.
    obj_id_to_mask (`dict[int, torch.FloatTensor]`, *optional*):
        Dictionary mapping object IDs to their predicted low-resolution masks.
    obj_id_to_score (`dict[int, float]`, *optional*):
        Dictionary mapping object IDs to their detection scores.
    obj_id_to_tracker_score (`dict[int, float]`, *optional*):
        Dictionary mapping object IDs to their tracker scores for the current frame.
    removed_obj_ids (`set[int]`, *optional*):
        Set of object IDs that have been removed (e.g., via hotstart heuristics).
    suppressed_obj_ids (`set[int]`, *optional*):
        Set of object IDs that have been suppressed in the current frame.
    unconfirmed_obj_ids (`list[int]`, *optional*):
        Object IDs that are tracked but not yet confirmed by masklet confirmation heuristics.
    frame_idx (`int`, *optional*):
        The frame index of the video.
    """

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
        self.det_nms_use_iom = config.det_nms_use_iom
        self.suppress_det_close_to_boundary = config.suppress_det_close_to_boundary
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
        self.use_iom_recondition = config.use_iom_recondition
        self.iom_thresh_recondition = config.iom_thresh_recondition
        self.masklet_confirmation_enable = config.masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = config.masklet_confirmation_consecutive_det_thresh

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

    def _refresh_recondition_object_pointer(
        self,
        inference_session: Sam31VideoInferenceSession,
        frame_idx: int,
        obj_idx: int,
        recond_mask_high_res: torch.Tensor,
    ) -> None:
        """Re-derive the object pointer for a reconditioned frame from the detector mask.

        Meta's `recondition_masks_in_existing_state` (`video_tracking_multiplex.py:3267`)
        runs `_use_mask_as_output(mask_inputs=new_masks)` for every reconditioned object
        and writes the resulting `obj_ptr` into the cond-frame `prev_output["obj_ptr"]`
        slot before re-encoding the spatial memory. The HF planning-phase recondition only
        replaces `high_res_masks` and re-encodes memory — so the stored
        `object_pointer` still reflects the (drifting) tracker prediction, and the
        memory-attention object-pointer stream keeps pulling the model toward the old
        location even though the spatial memory was corrected. This helper closes that
        gap by running `Sam31TrackerVideoModel._use_mask_as_output` with the detector
        mask as input on the interactive feature pyramid and overwriting
        `object_pointer`, `pred_masks`, and `object_score_logits` in the stored
        cond/non_cond entry for `frame_idx`.

        Mirrors Meta's per-object path (one call per reconditioned obj_idx) since
        HF's `_use_mask_as_output` is not multiplex-aware.
        """
        output_dict = inference_session.output_dict_per_obj[obj_idx]
        storage_key = (
            "cond_frame_outputs"
            if frame_idx in output_dict["cond_frame_outputs"]
            else ("non_cond_frame_outputs" if frame_idx in output_dict["non_cond_frame_outputs"] else None)
        )
        if storage_key is None:
            return

        current_out = output_dict[storage_key][frame_idx]
        if "object_pointer" not in current_out:
            return

        tracker = self.tracker_model
        (
            _propagation_feats,
            _propagation_pos,
            interactive_vision_feats,
            _interactive_vision_pos,
        ) = tracker._prepare_vision_features(inference_session, frame_idx, batch_size=1)

        interactive_high_res = None
        if len(interactive_vision_feats) > 1:
            interactive_high_res = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(interactive_vision_feats[:-1], tracker.backbone_feature_sizes[:-1])
            ]

        pix_feat = (
            interactive_vision_feats[-1]
            .permute(1, 2, 0)
            .view(-1, tracker.hidden_dim, *tracker.backbone_feature_sizes[-1])
        )
        no_mem_embed = tracker.no_memory_embedding.to(dtype=pix_feat.dtype, device=pix_feat.device)
        pix_feat = pix_feat + no_mem_embed.view(1, -1, 1, 1)

        # Pass the bilinearly-upsampled detector LOGIT mask through. Binarising
        # (matching Meta's `_recondition_masklets > 0` branch) helps deer slightly but
        # regresses foot/shoe by ~0.04 IoU; keeping the logit signal yields the best
        # combined improvement vs no-refresh baseline.
        mask_inputs = recond_mask_high_res.float()
        while mask_inputs.dim() < 4:
            mask_inputs = mask_inputs.unsqueeze(0)
        if mask_inputs.dtype != pix_feat.dtype:
            mask_inputs = mask_inputs.to(pix_feat.dtype)

        sam_output = tracker._use_mask_as_output(pix_feat, interactive_high_res, mask_inputs)

        target_device = current_out["object_pointer"].device
        target_dtype = current_out["object_pointer"].dtype
        current_out["object_pointer"] = sam_output.object_pointer.to(
            device=target_device, dtype=target_dtype
        )
        if "pred_masks" in current_out and current_out["pred_masks"] is not None:
            current_out["pred_masks"] = sam_output.pred_masks.to(
                device=current_out["pred_masks"].device,
                dtype=current_out["pred_masks"].dtype,
            )
        if "object_score_logits" in current_out and current_out["object_score_logits"] is not None:
            current_out["object_score_logits"] = sam_output.object_score_logits.to(
                device=current_out["object_score_logits"].device,
                dtype=current_out["object_score_logits"].dtype,
            )

    def _tracker_update_memories(
        self,
        inference_session: Sam31VideoInferenceSession,
        frame_idx: int,
        low_res_masks: torch.Tensor,
        reconditioned_masks: dict[int, torch.Tensor] | None = None,
    ) -> None:
        r"""SAM3.1 multiplex memory encoding for the full PCS video pipeline.

        Why this needs to override `Sam3VideoModel._tracker_update_memories`
        ------------------------------------------------------------------
        The inherited SAM3 implementation batches per-object masks as
        `(num_objects, 1, H, W)` and feeds them directly into
        `tracker_model._encode_new_memory(...)` via `run_memory_encoder(...)`. That works
        for the SAM3 video tracker (which encodes one mask per object), but is silently
        **incorrect** for the SAM3.1 multiplex tracker:

          * `Sam31TrackerVideoModel._encode_new_memory` expects bucketed multiplex input
            shaped `(num_buckets, multiplex_count, H, W)` (one channel per multiplex slot)
            followed by a parallel `(num_buckets, multiplex_count, 1, 1)`
            `conditioning_slots_mask` indicating which slots host conditioning objects on
            this frame. The mask downsampler's first conv was trained against
            `multiplex_count * mask_downsampler_input_channel_multiplier` channels (16 mask
            channels + 16 cond indicator channels for the default config).
          * When the inherited path passes `(num_objects, 1, H, W)`,
            `_encode_new_memory`'s shape-adapt branch zero-pads the missing channels (one
            real mask + 31 zero channels per object) and runs the encoder, producing
            memory features that have no relation to Meta's multiplex memory.
          * In addition, the inherited path doesn't store
            `propagation_image_features` / `propagation_image_pos_enc` next to the
            memory tensors — the propagation memory cross-attention path in
            `Sam31TrackerVideoModel._prepare_memory_conditioned_features_batched_for_propagation`
            then has to fall back to the per-frame vision-feature cache.

        Both issues are resolved by routing the per-object masks/scores through the
        tracker's own `_batch_encode_memories`, which already implements Meta's bucketed
        multiplex encoding (and writes the propagation FPN snapshot per object). We keep
        the SAM3-video heuristics that run before encoding (recondition override,
        per-prompt non-overlapping suppression, area-derived score logits) so the
        detector/tracker association behaviour stays unchanged.

        """
        if len(inference_session.obj_ids) == 0:
            return

        if reconditioned_masks is None:
            reconditioned_masks = {}

        # Match the multiplex memory encoder's input grid: SAM3.1 trains the multiplex
        # `_encode_new_memory` on masks at `image_size` (1008x1008 for the default config)
        # — that's the shape of `current_out["high_res_masks"]` the standalone PVS tracker
        # feeds into `_batch_encode_memories`. Interpolating up to `image_size` here means
        # `_encode_new_memory` will NOT hit its internal `shape != mask_mem_size` branch
        # (which would otherwise add a `bilinear + antialias` downsample to 1008, then the
        # `mask_downsampler` would upsample back to its 1152x1152 `interpol_size`).
        # Avoiding that round-trip restores the same single-resolution flow Meta runs
        # (where suppression is applied at the same resolution the memory encoder sees)
        # while staying compatible with HF's `_encode_new_memory`.
        target_h = target_w = self.tracker_model.image_size
        high_res_masks = nn.functional.interpolate(
            low_res_masks.unsqueeze(1).float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Recondition override: detector mask replaces tracker mask for high-confidence
        # IoU-matched objects at memory-encode time.
        #
        # Promote the reconditioned frame entry from `non_cond_frame_outputs` to
        # `cond_frame_outputs` BEFORE memory encoding so the new memory features land
        # in `cond_frame_outputs` (see `_batch_encode_memories`, which writes to whichever
        # storage_key already holds `frame_idx`). This mirrors Meta's
        # `_recondition_masklets`, which calls `add_new_masks(reconditioning=True)` —
        # that path always writes the recondition output to `cond_frame_outputs`
        # (`storage_key = "cond_frame_outputs" if is_cond else ...`) and removes it
        # from `non_cond_frame_outputs`. Without this promotion, recondition memory
        # features get evicted from the rolling `non_cond` window after
        # `num_maskmem - 1 = 6` frames and the model has no permanent anchors against
        # drift over long propagations (catastrophic past frames 60-100).
        #
        # Keep `is_mask_from_pts_per_obj=[False]*num_objects` below: Meta's
        # `_tracker_update_memories` always passes `is_mask_from_pts=False`. Promotion
        # (storage key) and is_mask_from_pts (memory-channel conditioning indicator)
        # are independent — Meta does both: promote AND `is_mask_from_pts=False`.
        for obj_idx, recond_mask in reconditioned_masks.items():
            recond = recond_mask.float()
            while recond.dim() < 4:
                recond = recond.unsqueeze(0)
            if recond.shape[-2:] != (target_h, target_w):
                recond = nn.functional.interpolate(
                    recond, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            high_res_masks[obj_idx] = recond.squeeze(0)

            output_dict = inference_session.output_dict_per_obj[obj_idx]
            # Mirror Meta's `recondition_masks_in_existing_state`: refresh object_pointer
            # / pred_masks / object_score_logits from the detector mask via
            # `_use_mask_as_output` before promoting the entry to `cond_frame_outputs`.
            # See `_refresh_recondition_object_pointer` for full rationale.
            self._refresh_recondition_object_pointer(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idx=obj_idx,
                recond_mask_high_res=recond.squeeze(0),
            )
            if frame_idx in output_dict["non_cond_frame_outputs"]:
                current_out = output_dict["non_cond_frame_outputs"].pop(frame_idx)
                output_dict["cond_frame_outputs"][frame_idx] = current_out

        prompt_ids_for_objects = [
            inference_session.obj_id_to_prompt_id[obj_id] for obj_id in inference_session.obj_ids
        ]
        high_res_masks = self._suppress_object_pw_area_shrinkage(high_res_masks, prompt_ids_for_objects)
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        num_objects = len(inference_session.obj_ids)
        objects_needing_memory_encoding = list(range(num_objects))
        high_res_masks_for_memory = [high_res_masks[i : i + 1] for i in range(num_objects)]
        object_score_logits_for_memory = [object_score_logits[i : i + 1] for i in range(num_objects)]
        # PCS planning-phase memory encoding runs before new detections are added to the
        # session and only for propagated masklets. Meta always uses
        # `is_mask_from_pts=False` here (see `Sam3MultiplexVideoBase._tracker_update_memories`).
        # New objects receive click/mask conditioning memory in `run_tracker_update_execution_phase`.
        is_mask_from_pts_per_obj = [False] * num_objects

        self.tracker_model._batch_encode_memories(
            inference_session=inference_session,
            frame_idx=frame_idx,
            objects_needing_memory_encoding=objects_needing_memory_encoding,
            high_res_masks_for_memory=high_res_masks_for_memory,
            object_score_logits_for_memory=object_score_logits_for_memory,
            is_mask_from_pts_per_obj=is_mask_from_pts_per_obj,
        )
        self.tracker_model._prune_stale_tracker_outputs(inference_session, frame_idx)

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
        # SAM3.1-specific multiplex bucketed memory encoding happens upstream of this
        # call: `Sam3VideoModel.run_tracker_update_planning_phase` invokes
        # `_tracker_update_memories`, which we override above to route through
        # `Sam31TrackerVideoModel._batch_encode_memories` (and which writes the
        # `propagation_image_features` snapshot the multiplex memory cross-attention
        # path looks up in `_get_stored_or_cached_propagation_image_features`).
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

    def _prepare_recondition_masks(
        self,
        inference_session: Sam31VideoInferenceSession,
        frame_idx: int,
        det_out: dict[str, torch.Tensor],
        trk_masks: torch.Tensor,
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        tracker_obj_scores_global: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], set[int]]:
        r"""Meta `_recondition_masklets` low-res merge + memory-encode overrides for PCS.

        Updates `trk_masks` in-place (binary agreement with detector, hole fill) and
        returns high-res detector masks for `_tracker_update_memories`. Does not call
        `add_new_masks(reconditioning=True)` here — that path marks frames as
        click-conditioning and breaks multiplex propagation memory.
        """
        reconditioned_masks: dict[int, torch.Tensor] = {}
        reconditioned_obj_ids: set[int] = set()
        if len(trk_id_to_max_iou_high_conf_det) == 0:
            return reconditioned_masks, reconditioned_obj_ids

        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            obj_idx = inference_session.obj_id_to_idx(trk_obj_id)
            if tracker_obj_scores_global[obj_idx].sigmoid() <= self.high_conf_thresh:
                continue

            new_mask = det_out["mask"][det_idx : det_idx + 1]
            old_mask = trk_masks[obj_idx : obj_idx + 1]
            if new_mask.shape[-2:] != old_mask.shape[-2:]:
                new_mask = F.interpolate(
                    new_mask.unsqueeze(1),
                    size=old_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            binary_agreement = (new_mask > 0) == (old_mask > 0)
            merged = torch.where(binary_agreement, old_mask, new_mask)
            merged = fill_holes_in_mask_scores(
                merged.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            ).squeeze(1)
            trk_masks[obj_idx] = merged.squeeze(0)

            # Pass the detector low-res LOGIT mask straight through (not binarised).
            # `_tracker_update_memories` bilinearly upsamples this to `image_size` and
            # then `_batch_encode_memories` feeds the result into `_encode_new_memory`,
            # which expects the same low-res-logits-upsampled-via-bilinear input shape
            # that the standalone PVS tracker produces. Binarising before bilinear
            # interpolation collapses the boundary into a smooth [0,1] gradient that
            # the encoder (trained on mask-logit input) doesn't see in non-recondition
            # frames — empirically losing ~0.5 IoU on the deer 60+ regime.
            det_for_mem = det_out["mask"][det_idx : det_idx + 1].unsqueeze(1)
            reconditioned_masks[obj_idx] = det_for_mem
            reconditioned_obj_ids.add(trk_obj_id)

        return reconditioned_masks, reconditioned_obj_ids

    def build_outputs(
        self,
        inference_session: Sam31VideoInferenceSession,
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_update_plan: dict,
        reconditioned_obj_ids: set | None = None,
    ):
        """Meta-parity build_outputs for SAM3.1 multiplex video.

        Identical to `Sam3VideoModel.build_outputs` except for Part 3: we do not
        overwrite the reconditioned objects' masks with the raw detector logits.
        `_prepare_recondition_masks` already wrote Meta's agreement-preserving merge
        (`torch.where((det>0)==(trk>0), trk, det)` + `fill_holes_in_mask_scores`) into
        `tracker_low_res_masks_global[obj_idx]` (see Meta `_recondition_masklets`
        `facebook_sam3/sam3/model/sam3_multiplex_base.py:925-938`). Overriding here
        with the raw detector mask discards that merge and degrades the output (~+0.04
        IoU on the deer 30-79 / 60+ regime). Part 1 picks the merged mask up via
        `tracker_low_res_masks_global`.
        """
        new_det_out_inds: list[int] = tracker_update_plan["new_det_out_inds"]
        new_det_obj_ids: list[int] = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask: dict[int, torch.Tensor] = {}

        existing_masklet_obj_ids = inference_session.obj_ids
        for obj_id, mask in zip(existing_masklet_obj_ids, tracker_low_res_masks_global):
            obj_id_to_mask[int(obj_id)] = mask.unsqueeze(0)

        if len(new_det_out_inds) > 0:
            new_det_out_inds_t = torch.tensor(
                new_det_out_inds, dtype=torch.long, device=det_out["mask"].device
            )
            new_det_low_res_masks = det_out["mask"][new_det_out_inds_t]
            new_det_low_res_masks = fill_holes_in_mask_scores(
                new_det_low_res_masks.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            ).squeeze(1)
            for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
                obj_id_to_mask[int(obj_id)] = mask.unsqueeze(0)

        return obj_id_to_mask


__all__ = [
    "Sam31VideoConfig",
    "Sam31VideoModel",
    "Sam31VideoPreTrainedModel",
    "Sam31VideoInferenceSession",
    "Sam31VideoSegmentationOutput",
]
