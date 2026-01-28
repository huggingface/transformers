# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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


from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from transformers.models.sam3.modeling_sam3 import Sam3VisionNeck

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, is_kernels_available, logging
from ..auto import AutoModel
from .configuration_sam3_video import Sam3VideoConfig


if is_kernels_available():
    from ...integrations.hub_kernels import get_kernel

logger = logging.get_logger(__name__)

cv_utils_kernel = None  # None = not attempted, False = failed, kernel object = success


def _load_cv_utils_kernel_once():
    """Load cv_utils_kernel once on first use."""
    global cv_utils_kernel
    if cv_utils_kernel is not None:
        return  # Already attempted loading (successfully or not)

    if not is_kernels_available():
        logger.warning_once(
            "kernels library is not installed. NMS post-processing, hole filling, and sprinkle removal will be skipped. "
            "Install it with `pip install kernels` for better mask quality."
        )
        cv_utils_kernel = False
        return

    try:
        cv_utils_kernel = get_kernel("kernels-community/cv-utils")
    except Exception as e:
        logger.warning_once(
            f"Failed to load cv_utils kernel (your torch/cuda setup may not be supported): {e}. "
            "NMS post-processing, hole filling, and sprinkle removal will be skipped."
        )
        cv_utils_kernel = False


class Sam3VideoInferenceCache:
    """Cache for vision features and model constants."""

    def __init__(
        self,
        inference_device: torch.device | str = "cpu",
        inference_state_device: torch.device | str = "cpu",
        max_vision_features_cache_size: int = 1,
    ):
        self.inference_device = inference_device
        self.inference_state_device = inference_state_device
        self.max_vision_features_cache_size = max_vision_features_cache_size

        self._vision_features = {}

    def cache_vision_features(self, frame_idx: int, features: dict):
        """Cache vision features with automatic device management."""
        cached = {}
        if len(self._vision_features) >= self.max_vision_features_cache_size:
            # remove the oldest frame
            self._vision_features.pop(min(self._vision_features.keys()))

        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                cached[key] = value.to(self.inference_state_device, non_blocking=True)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                cached[key] = [v.to(self.inference_state_device, non_blocking=True) for v in value]
            else:
                cached[key] = value
        self._vision_features[frame_idx] = cached

    def get_vision_features(self, frame_idx: int) -> dict | None:
        """Get cached vision features, automatically moved to inference device."""
        if frame_idx not in self._vision_features:
            return None

        cached = self._vision_features[frame_idx]
        moved = {}
        for key, value in cached.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.inference_device, non_blocking=True)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                moved[key] = [v.to(self.inference_device, non_blocking=True) for v in value]
            else:
                moved[key] = value
        return moved

    def clear_all(self):
        """Clear all cached data."""
        self._vision_features.clear()


class Sam3VideoInferenceSession:
    r"""
    Manages video inference session parameters, state and cache.

    Args:
        video (`torch.FloatTensor`, *optional*):
            The video to process. No need to provide when streaming.
        video_height (`int`, *optional*):
            The height of the video.
        video_width (`int`, *optional*):
            The width of the video.
        inference_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to use for inference.
        inference_state_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to store the inference state on.
        video_storage_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to store the video on.
        dtype (`torch.dtype`, *optional*, defaults to `"float32"`):
            The dtype to use for the video.
        max_vision_features_cache_size (`int`, *optional*, defaults to 1):
            The maximum number of vision features to cache.
    """

    def __init__(
        self,
        video: torch.FloatTensor | None = None,
        video_height: int | None = None,
        video_width: int | None = None,
        inference_device: torch.device | str = "cpu",
        inference_state_device: torch.device | str = "cpu",
        video_storage_device: torch.device | str = "cpu",
        dtype: torch.dtype | str = "float32",
        max_vision_features_cache_size: int = 1,
    ):
        # store as a dictionary to avoid double memory allocation with torch.cat when adding new frames
        self.processed_frames = (
            dict(enumerate(video.to(video_storage_device, dtype=dtype))) if video is not None else None
        )
        self.video_height = video_height
        self.video_width = video_width

        self.inference_device = inference_device
        self.inference_state_device = inference_state_device
        self.video_storage_device = video_storage_device
        self.dtype = dtype
        self.max_vision_features_cache_size = max_vision_features_cache_size

        # Cache for computed features
        self.cache = Sam3VideoInferenceCache(
            inference_device=self.inference_device,
            inference_state_device=self.inference_state_device,
            max_vision_features_cache_size=self.max_vision_features_cache_size,
        )

        # Persistent object tracking state
        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        self.mask_inputs_per_obj = {}
        self.point_inputs_per_obj = {}

        # Persistent model outputs/history
        self.output_dict_per_obj = {}
        self.frames_tracked_per_obj = {}

        # Multi-prompt support
        self.prompts = {}  # prompt_id -> prompt_text
        self.prompt_input_ids = {}  # prompt_id -> input_ids
        self.prompt_embeddings = {}  # prompt_id -> text embeddings
        self.prompt_attention_masks = {}  # prompt_id -> attention_mask
        self.obj_id_to_prompt_id = {}  # obj_id -> prompt_id (assigned at detection time)

        # Tracking metadata for detection-tracking fusion
        self.obj_id_to_score = {}  # Detection scores per object
        self.obj_id_to_tracker_score_frame_wise = defaultdict(dict)  # Frame-wise tracker scores
        self.obj_id_to_last_occluded = {}  # Last occlusion frame per object
        self.max_obj_id = -1  # Maximum object ID assigned so far (-1 means no object has been assigned yet)

        # Hotstart metadata
        self.obj_first_frame_idx = {}  # First frame index per object
        self.unmatched_frame_inds = defaultdict(list)  # Unmatched frame indices per object
        self.overlap_pair_to_frame_inds = defaultdict(list)  # Overlap tracking for duplicate detection
        self.trk_keep_alive = {}  # Keep-alive counters per object
        self.removed_obj_ids = set()  # Set of removed object IDs
        self.suppressed_obj_ids = defaultdict(set)  # Suppressed object IDs per frame
        self.hotstart_removed_obj_ids = set()  # Set of removed object IDs during hotstart

        # Output buffering for hotstart delay
        self.output_buffer = []

    @property
    def num_frames(self) -> int | None:
        """Number of frames in the video."""
        return len(self.processed_frames) if self.processed_frames is not None else None

    def add_prompt(self, prompt_text: str) -> int:
        """
        Add a text prompt to the session and return its unique ID.
        If the prompt already exists, returns the existing ID.
        """
        for prompt_id, text in self.prompts.items():
            if text == prompt_text:
                return prompt_id

        prompt_id = len(self.prompts)
        self.prompts[prompt_id] = prompt_text
        return prompt_id

    # Object management
    def obj_id_to_idx(self, obj_id: int) -> int:
        """Map object ID to index, creating new entry if needed."""
        if obj_id not in self._obj_id_to_idx:
            obj_idx = len(self._obj_id_to_idx)
            self._obj_id_to_idx[obj_id] = obj_idx
            self._obj_idx_to_id[obj_idx] = obj_id
            self.obj_ids.append(obj_id)

            self.mask_inputs_per_obj[obj_idx] = {}
            self.point_inputs_per_obj[obj_idx] = {}
            self.output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            self.frames_tracked_per_obj[obj_idx] = {}
        return self._obj_id_to_idx[obj_id]

    # Video Inference specific functions
    def obj_idx_to_id(self, obj_idx: int) -> int:
        """Map model-side object index to client-side object id."""
        return self._obj_idx_to_id[obj_idx]

    def get_obj_num(self) -> int:
        """Get the total number of unique object ids received so far in this session."""
        return len(self._obj_idx_to_id)

    def add_mask_inputs(self, obj_idx: int, frame_idx: int, inputs: torch.Tensor):
        """Add mask inputs with automatic device placement."""
        self.mask_inputs_per_obj[obj_idx][frame_idx] = inputs.to(
            self.inference_device, dtype=self.dtype, non_blocking=True
        )

    def remove_mask_inputs(self, obj_idx: int, frame_idx: int):
        """Remove mask inputs."""
        self.mask_inputs_per_obj[obj_idx].pop(frame_idx, None)

    def remove_object(self, obj_id: int, strict: bool = False):
        """
        Remove an object from the inference session. This would remove the object from
        all frames in the video.

        Args:
            obj_id (`int`): The object ID to remove.
            strict (`bool`, *optional*, defaults to `False`): Whether to raise an error if the object doesn't exist.
        """
        old_obj_idx_to_rm = self._obj_id_to_idx.get(obj_id, None)
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. All existing object ids: {self.obj_ids}."
            )

        # Clean up prompt mapping
        self.obj_id_to_prompt_id.pop(obj_id, None)

        # If this is the only remaining object id, we simply reset the state.
        if len(self._obj_id_to_idx) == 1:
            self.reset_inference_session()
            return

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = self.obj_ids
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        self._obj_id_to_idx = dict(zip(new_obj_ids, new_obj_inds))
        self._obj_idx_to_id = dict(zip(new_obj_inds, new_obj_ids))
        self.obj_ids = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(self.point_inputs_per_obj)
        _map_keys(self.mask_inputs_per_obj)
        _map_keys(self.output_dict_per_obj)
        _map_keys(self.frames_tracked_per_obj)

    # Output management with smart device placement
    def store_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: str | None = None,
        output_value: torch.Tensor | dict | None = None,
        is_conditioning_frame: bool = True,
    ):
        """
        Store output with smart device management.
        If output_key is None, the output is stored as a dictionary.

        Args:
            obj_idx (int): The index of the object.
            frame_idx (int): The index of the frame.
            output_key (Optional[str]): The key of the output. If None, the output is stored as a dictionary.
            output_value (Optional[Union[torch.Tensor, dict]]): The value of the output.
            is_conditioning_frame (bool): Whether the output is for a conditioning frame.
        """
        storage_key = "cond_frame_outputs" if is_conditioning_frame else "non_cond_frame_outputs"

        if output_key is None and isinstance(output_value, dict):
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx] = {}
            for key, value in output_value.items():
                self.store_output(obj_idx, frame_idx, key, value, is_conditioning_frame)
            return

        # Device placement: small tensors stay on inference device, large ones go to inference state device
        if output_key in ["object_pointer", "object_score_logits"]:  # Small tensors
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value
        elif isinstance(output_value, torch.Tensor):  # Large tensors like masks, features
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value.to(
                self.inference_state_device, non_blocking=True
            )
        else:
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value

    def get_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: str,
        is_conditioning_frame: bool = True,
    ):
        """
        Get output with smart device management.

        Args:
            obj_idx (int): The index of the object.
            frame_idx (int): The index of the frame.
            output_key (str): The key of the output.
            is_conditioning_frame (bool): Whether the output is for a conditioning frame.
        """
        storage_key = "cond_frame_outputs" if is_conditioning_frame else "non_cond_frame_outputs"
        out = self.output_dict_per_obj[obj_idx][storage_key].get(frame_idx, None)
        # move to inference device if needed
        if out is None:
            return None
        value = out[output_key]
        if isinstance(value, torch.Tensor):
            value = value.to(self.inference_device, non_blocking=True)
        return value

    # Video frame management
    def add_new_frame(self, pixel_values: torch.Tensor, frame_idx: int | None = None) -> int:
        """Add new frame with automatic device placement."""
        pixel_values = pixel_values.to(self.video_storage_device, dtype=self.dtype, non_blocking=True)
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.squeeze(0)

        if frame_idx is None:
            frame_idx = len(self.processed_frames) if self.processed_frames is not None else 0

        if self.processed_frames is None:
            self.processed_frames = {frame_idx: pixel_values}
        else:
            self.processed_frames[frame_idx] = pixel_values

        return frame_idx

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """Get frame from video."""
        return self.processed_frames[frame_idx].to(self.inference_device, non_blocking=True)

    def reset_tracking_data(self):
        """Reset tracking data but keep cache."""
        self._obj_id_to_idx.clear()
        self._obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.output_dict_per_obj.clear()
        self.frames_tracked_per_obj.clear()
        # Note: cache and video data are preserved

        # Reset prompt mappings for objects (but keep prompts themselves)
        self.obj_id_to_prompt_id.clear()

    def reset_inference_session(self):
        """Reset tracking data and cache."""
        self._obj_id_to_idx.clear()
        self._obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.output_dict_per_obj.clear()
        self.frames_tracked_per_obj.clear()
        self.cache.clear_all()

        # Reset prompt mappings for objects (but keep prompts themselves)
        self.obj_id_to_prompt_id.clear()

    def reset_state(self):
        """Reset the inference session state."""
        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []
        self.output_dict_per_obj = {}
        self.frames_tracked_per_obj = {}

        # Reset detection-tracking fusion state
        self.obj_id_to_score = {}
        self.obj_id_to_tracker_score_frame_wise = defaultdict(dict)
        self.obj_id_to_last_occluded = {}
        self.max_obj_id = 0
        self.obj_first_frame_idx = {}
        self.unmatched_frame_inds = defaultdict(list)
        self.overlap_pair_to_frame_inds = defaultdict(list)
        self.trk_keep_alive = {}
        self.removed_obj_ids = set()
        self.suppressed_obj_ids = defaultdict(set)
        self.output_buffer = []

        # Reset multi-prompt state
        self.prompts.clear()
        self.prompt_input_ids.clear()
        self.prompt_embeddings.clear()
        self.prompt_attention_masks.clear()
        self.obj_id_to_prompt_id.clear()

        # Clear cache
        self.cache.clear_all()


@dataclass
@auto_docstring(custom_intro="Base class for the Sam3Video model's output.")
class Sam3VideoSegmentationOutput(ModelOutput):
    r"""
    object_ids (`list[int]`, *optional*):
        List of object IDs being tracked in the current frame.
    obj_id_to_mask (`dict[int, torch.FloatTensor]`, *optional*):
        Dictionary mapping object IDs to their predicted low-resolution masks.
        Each mask has shape `(1, H_low, W_low)`.
    obj_id_to_score (`dict[int, float]`, *optional*):
        Dictionary mapping object IDs to their detection scores.
    obj_id_to_tracker_score (`dict[int, float]`, *optional*):
        Dictionary mapping object IDs to their tracker scores for the current frame.
    removed_obj_ids (`set[int]`, *optional*):
        Set of object IDs that have been removed (e.g., via hotstart heuristics).
    suppressed_obj_ids (`set[int]`, *optional*):
        Set of object IDs that have been suppressed in the current frame.
    frame_idx (`int`, *optional*):
        The frame index of the video.
    """

    object_ids: list[int] | None = None
    obj_id_to_mask: dict[int, torch.FloatTensor] | None = None
    obj_id_to_score: dict[int, float] | None = None
    obj_id_to_tracker_score: dict[int, float] | None = None
    removed_obj_ids: set[int] | None = None
    suppressed_obj_ids: set[int] | None = None
    frame_idx: int | None = None


class Sam3VideoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Sam3VideoConfig
    base_model_prefix = "sam3_video"
    main_input_name = "pixel_values"
    input_modalities = ["video", "text"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@auto_docstring
class Sam3VideoModel(Sam3VideoPreTrainedModel):
    def __init__(self, config: Sam3VideoConfig):
        super().__init__(config)
        self.config = config
        self.detector_model = AutoModel.from_config(config.detector_config)
        self.tracker_model = AutoModel.from_config(config.tracker_config, remove_vision_encoder=True)
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

        # the maximum object number
        self.max_num_objects = config.max_num_objects
        self.recondition_every_nth_frame = config.recondition_every_nth_frame
        self.high_conf_thresh = config.high_conf_thresh
        self.high_iou_thresh = config.high_iou_thresh

        self.tracker_neck = Sam3VisionNeck(config.detector_config.vision_config)

        self.post_init()

    def get_vision_features_for_tracker(self, vision_embeds: torch.Tensor):
        hidden_states = vision_embeds.last_hidden_state
        batch_size = hidden_states.shape[0]
        height, width = self.tracker_model.prompt_encoder.image_embedding_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        fpn_hidden_states, fpn_position_encoding = self.tracker_neck(hidden_states_spatial)

        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        feature_maps = list(fpn_hidden_states[:-1])
        feature_maps[0] = self.tracker_model.mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.tracker_model.mask_decoder.conv_s1(feature_maps[1])

        # flatten NxCxHxW to HWxNxC
        feature_maps = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
        feature_maps_position_embeddings = [
            feature_map_position_embedding.flatten(2).permute(2, 0, 1)
            for feature_map_position_embedding in fpn_position_encoding[:-1]
        ]
        return feature_maps, feature_maps_position_embeddings

    def run_detection(
        self,
        inference_session: Sam3VideoInferenceSession,
        vision_embeds: torch.Tensor,
    ):
        """
        Run detection for all prompts efficiently by reusing vision embeddings.

        Args:
            inference_session: The inference session containing prompts and state
            vision_embeds: Pre-computed vision embeddings to reuse across prompts

        Returns:
            Dictionary mapping prompt_id to detection outputs
        """
        prompt_ids = list(inference_session.prompts.keys())
        if not prompt_ids:
            raise ValueError("No prompts available for detection. Please add prompts to the session first.")

        all_detections = {}

        for prompt_id in prompt_ids:
            # Get or compute text embeddings for this prompt
            if prompt_id not in inference_session.prompt_embeddings:
                text_embeds = self.detector_model.get_text_features(
                    input_ids=inference_session.prompt_input_ids[prompt_id],
                    attention_mask=inference_session.prompt_attention_masks[prompt_id],
                    return_dict=True,
                ).pooler_output
                inference_session.prompt_embeddings[prompt_id] = text_embeds
            else:
                text_embeds = inference_session.prompt_embeddings[prompt_id]

            # Run detector with cached vision features (efficient!)
            detector_outputs = self.detector_model(
                vision_embeds=vision_embeds,
                text_embeds=text_embeds,
                attention_mask=inference_session.prompt_attention_masks[prompt_id],
            )

            pred_logits = detector_outputs.pred_logits
            presence_logits = detector_outputs.presence_logits

            pred_probs = pred_logits.sigmoid()
            presence_scores = presence_logits.sigmoid()
            pred_probs = pred_probs * presence_scores

            run_nms = self.det_nms_thresh > 0.0
            if run_nms:
                keep = nms_masks(
                    pred_probs=pred_probs[0],
                    pred_masks=detector_outputs.pred_masks[0],
                    prob_threshold=self.score_threshold_detection,
                    iou_threshold=self.det_nms_thresh,
                )
                # Set suppressed detections' probabilities to 0
                pred_probs[0][~keep] = 0.0

            pred_boxes_xyxy = detector_outputs.pred_boxes
            pred_masks = detector_outputs.pred_masks
            # get the positive detection outputs above threshold
            pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)
            det_out = {
                "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
                "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
                "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
            }

            all_detections[prompt_id] = det_out

        return all_detections

    def run_tracker_propagation(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        reverse: bool,
    ):
        low_res_masks_list = []
        obj_scores_list = []
        if len(inference_session.obj_ids) > 0:
            # propagate one frame
            out = self.tracker_model(
                inference_session=inference_session,
                frame_idx=frame_idx,
                reverse=reverse,
                run_mem_encoder=False,
            )
            out_low_res_masks = out.pred_masks
            out_obj_scores = out.object_score_logits

            # only 1 frames should be propagated
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.low_res_mask_size
        if len(low_res_masks_list) > 0:
            low_res_masks = torch.cat(low_res_masks_list, dim=0)
            obj_scores = torch.cat(obj_scores_list, dim=0)

            # Apply hole filling to the masks
            low_res_masks = fill_holes_in_mask_scores(
                low_res_masks.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
            low_res_masks = low_res_masks.squeeze(1)
        else:
            low_res_masks = torch.zeros(0, H_mask, W_mask, device=self.device)
            obj_scores = torch.zeros(0, device=self.device)

        return low_res_masks, obj_scores

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores: Tensor,
        trk_masks: Tensor,
        trk_obj_ids: list[int],
        det_prompt_ids: torch.Tensor,
        trk_prompt_ids: torch.Tensor,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores: (N,) tensor of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) list of object IDs corresponding to trk_masks
          - det_prompt_ids: (N,) tensor of prompt IDs for each detection. Prevents cross-prompt
            associations by zeroing IoUs between detections and tracks from different prompts.
          - trk_prompt_ids: (M,) tensor of prompt IDs for each tracked object. Prevents cross-prompt
            associations by zeroing IoUs between detections and tracks from different prompts.

        Returns:
          - new_det_out_inds: list of new object indices among in FA detection outputs
          - unmatched_trk_obj_ids: list of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, list[int]]: mapping from FA detection indices
            to the list of matched tracklet object IDs
          - empty_trk_obj_ids: list of existing masklet object IDs with zero area in SAM2 prediction
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        trk_obj_ids_tensor = (
            torch.tensor(trk_obj_ids, dtype=torch.long, device=det_masks.device)
            if trk_obj_ids
            else torch.empty(0, dtype=torch.long, device=det_masks.device)
        )
        if trk_masks.size(0) == 0:
            # all detections are new
            new_det_out_inds = list(range(det_masks.size(0)))
            unmatched_trk_obj_ids = []
            empty_trk_obj_ids = []
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_out_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            # all previous tracklets are unmatched if they have a non-zero area
            new_det_out_inds = []
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2))  # (M,) tensor
            # Use tensor boolean indexing - elegant and avoids intermediate conversions
            unmatched_trk_obj_ids = trk_obj_ids_tensor[trk_is_nonempty].tolist()
            empty_trk_obj_ids = trk_obj_ids_tensor[~trk_is_nonempty].tolist()
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_out_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary, trk_masks_binary)  # (N, M) tensor

        # Prevent cross-prompt associations by zeroing IoUs between different prompt groups.
        prompt_match = det_prompt_ids.unsqueeze(1) == trk_prompt_ids.unsqueeze(0)
        ious = torch.where(prompt_match, ious, torch.zeros_like(ious))

        # trk_is_matched: for each track, True if matched to any detection above threshold
        trk_is_matched = (ious >= iou_threshold_trk).any(dim=0)  # (M,)
        # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2))  # (M,)
        trk_is_unmatched = trk_is_nonempty & ~trk_is_matched  # (M,)
        # Use tensor boolean indexing directly - no intermediate conversions
        unmatched_trk_obj_ids = trk_obj_ids_tensor[trk_is_unmatched].tolist()
        empty_trk_obj_ids = trk_obj_ids_tensor[~trk_is_nonempty].tolist()

        # For detections: allow many tracks to match to the same detection (many-to-one)
        # So, a detection is 'new' if it does not match any track above threshold
        det_matches_any_trk = (ious >= iou_threshold).any(dim=1)  # (N,)
        is_new_det = (det_scores >= new_det_thresh) & ~det_matches_any_trk  # (N,)
        new_det_out_inds = torch.where(is_new_det)[0].tolist()

        # Build detection-to-track mappings using tensor operations
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}  # trk id --> exactly one detection idx
        det_to_max_iou_trk_idx = ious.argmax(dim=1)  # (N,)
        det_is_high_conf = (det_scores >= self.high_conf_thresh) & ~is_new_det  # (N,)
        det_max_iou = ious.max(dim=1)[0]  # (N,)
        det_is_high_iou = det_max_iou >= self.high_iou_thresh  # (N,)
        det_is_high_conf_and_iou = det_is_high_conf & det_is_high_iou  # (N,)
        high_conf_and_iou_mask = det_is_high_conf_and_iou  # Keep as tensor

        for det_idx in range(det_masks.size(0)):
            # Find which tracks match this detection using tensor boolean indexing
            matched_trk_mask = ious[det_idx] >= iou_threshold  # (M,)
            det_to_matched_trk_obj_ids[det_idx] = trk_obj_ids_tensor[matched_trk_mask].tolist()

            if high_conf_and_iou_mask[det_idx].item():
                trk_idx = det_to_max_iou_trk_idx[det_idx].item()
                trk_obj_id = trk_obj_ids_tensor[trk_idx].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = det_idx

        return (
            new_det_out_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _process_hotstart(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, list[int]],
        new_det_obj_ids: list[int],
        empty_trk_obj_ids: list[int],
        unmatched_trk_obj_ids: list[int],
        extra_metadata: dict[str, Any],
        streaming: bool = False,
    ):
        """
        Handle hotstart heuristics to remove unmatched or duplicated objects.

        In streaming mode, hotstart removal logic is disabled since we don't have
        future frames to make informed decisions about object removal.
        """
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = extra_metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = extra_metadata["unmatched_frame_inds"]
        trk_keep_alive = extra_metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = extra_metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = extra_metadata["removed_obj_ids"]
        suppressed_obj_ids = extra_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()  # object IDs to be newly removed on this frame
        hotstart_diff = frame_idx - self.hotstart_delay if not reverse else frame_idx + self.hotstart_delay

        # Step 1: log the frame index where each object ID first appears
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            trk_keep_alive[int(obj_id)] = self.init_trk_keep_alive

        matched_trks = set()
        # We use the det-->tracks list to check for matched objects. Otherwise, we need to compute areas to decide whether they're occluded
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update({int(obj_id) for obj_id in matched_trks_per_det})
        for obj_id in matched_trks:
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the max value of trk_keep_alive
            trk_keep_alive[int(obj_id)] = min(self.max_trk_keep_alive, trk_keep_alive[int(obj_id)] + 1)
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
            # The max keep alive is 2x the min, means the model prefers to keep the prediction rather than suppress it if it was matched long enough.
            trk_keep_alive[int(obj_id)] = max(self.min_trk_keep_alive, trk_keep_alive[int(obj_id)] - 1)
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
                trk_keep_alive[int(obj_id)] = max(self.min_trk_keep_alive, trk_keep_alive[int(obj_id)] - 1)

        # Step 2: removed tracks that has not matched with detections for `hotstart_unmatch_thresh` frames with hotstart period
        # a) add unmatched frame indices for each existing object ID
        # note that `unmatched_trk_obj_ids` contains those frames where the SAM2 output mask
        # doesn't match any FA detection; it excludes those frames where SAM2 gives an empty mask
        # b) remove a masklet if it first appears after `hotstart_diff` and is unmatched for more
        # than `self.hotstart_unmatch_thresh` frames
        # NOTE: In streaming mode, we skip hotstart removal logic since we don't have future frames
        if not streaming:
            for obj_id, frame_indices in unmatched_frame_inds.items():
                if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                    continue  # skip if the object is already removed
                if len(frame_indices) >= self.hotstart_unmatch_thresh:
                    is_within_hotstart = (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                        obj_first_frame_idx[obj_id] < hotstart_diff and reverse
                    )
                    if is_within_hotstart:
                        obj_ids_newly_removed.add(obj_id)
                        logger.info(
                            f"Removing object {obj_id} at frame {frame_idx} "
                            f"since it is unmatched for frames: {frame_indices}"
                        )
                if (
                    trk_keep_alive[obj_id] <= 0  # Object has not been matched for too long
                    and not self.suppress_unmatched_only_within_hotstart
                    and obj_id not in removed_obj_ids
                    and obj_id not in obj_ids_newly_removed
                ):
                    logger.debug(f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched")
                    suppressed_obj_ids.add(obj_id)

        # Step 3: removed tracks that overlaps with another track for `hotstart_dup_thresh` frames
        # a) find overlaps tracks -- we consider overlap if they match to the same detection
        # NOTE: In streaming mode, we still track overlaps for metadata but skip removal logic
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            if len(matched_trk_obj_ids) < 2:
                continue  # only count detections that are matched to multiple (>=2) masklets
            # if there are multiple matched track ids, we need to find the one that appeared first;
            # these later appearing ids may be removed since they may be considered as duplicates
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b) remove a masklet if it first appears after `hotstart_diff` and it overlaps with another
        # masklet (that appears earlier) for more than `self.hotstart_dup_thresh` frames
        # NOTE: In streaming mode, we skip hotstart removal logic since we don't have future frames
        if not streaming:
            for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
                if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                    continue  # skip if the object is already removed
                if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                    obj_first_frame_idx[obj_id] < hotstart_diff and reverse
                ):
                    if len(frame_indices) >= self.hotstart_dup_thresh:
                        obj_ids_newly_removed.add(obj_id)
                        logger.info(
                            f"Removing object {obj_id} at frame {frame_idx} "
                            f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                        )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, extra_metadata

    def run_memory_encoder(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        high_res_masks: torch.Tensor,
        object_score_logits: torch.Tensor,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        cached_features = inference_session.cache.get_vision_features(frame_idx)
        current_vision_feats = cached_features["vision_feats"]
        maskmem_features, maskmem_pos_enc = self.tracker_model._encode_new_memory(
            current_vision_feats=current_vision_feats[-1],
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
        return maskmem_features, maskmem_pos_enc

    def _prepare_recondition_masks(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        det_out: dict[str, Tensor],
        trk_masks: Tensor,
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        tracker_obj_scores_global: Tensor,
    ) -> dict[int, Tensor]:
        """
        Prepare high-resolution masks for reconditioned objects.
        Returns a dict of obj_idx -> high_res_mask for objects that should be reconditioned.

        When recondition_on_trk_masks=True, uses detector as validation signal to strengthen tracker memory.
        When False, uses detector to correct tracker drift by replacing with detection masks.
        """
        reconditioned_masks = {}
        reconditioned_obj_ids = set()

        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            obj_idx = inference_session.obj_id_to_idx(trk_obj_id)
            obj_score = tracker_obj_scores_global[obj_idx]
            if obj_score <= self.high_conf_thresh:
                continue

            if self.recondition_on_trk_masks:
                # Validation mode: detector confirms tracker quality, strengthen memory with tracked mask
                new_mask = trk_masks[obj_idx : obj_idx + 1].unsqueeze(1)
                reconditioned_masks[obj_idx] = new_mask
                reconditioned_obj_ids.add(trk_obj_id)
            else:
                # Correction mode: detector corrects drift, replace tracker mask with detection mask
                new_mask = det_out["mask"][det_idx : det_idx + 1].unsqueeze(1)
                reconditioned_masks[obj_idx] = new_mask >= 0.5
                reconditioned_obj_ids.add(trk_obj_id)

        return reconditioned_masks, reconditioned_obj_ids

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: list[int],
        obj_ids: list[int],
        reverse: bool = False,
    ):
        # Suppress overlapping masks for objects that were most recently occluded
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)  # [N,N]

        # Create masks for upper triangular matrix (i < j) and IoU threshold
        mask_iou_thresh = iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # Suppress most recently occluded
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)  # (last_occ_expanded_i > last_occ_expanded_j)
            & (last_occ_expanded_j > -1)  # j can suppress i only if i was previously occluded
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)  # i can suppress j only if j was previously occluded
        )
        # Apply suppression
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        return to_suppress

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_new: dict[str, Any],
        obj_ids_newly_removed: set[int],
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (Dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (Set[int]): The object IDs that have been removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = inference_session.obj_ids
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000  # This value should be larger than any possible frame index, indicates that the object was removed by hotstart logic
            last_occluded_prev = torch.cat(
                [
                    inference_session.obj_id_to_last_occluded.get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(NEVER_OCCLUDED if obj_id not in obj_ids_newly_removed else ALWAYS_OCCLUDED),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )

            prompt_ids_global = torch.tensor(
                [inference_session.obj_id_to_prompt_id[obj_id] for obj_id in obj_ids_global],
                device=binary_tracker_low_res_masks_global.device,
                dtype=torch.long,
            )
            to_suppress = torch.zeros(
                batch_size,
                device=binary_tracker_low_res_masks_global.device,
                dtype=torch.bool,
            )

            # Only suppress overlaps within the same prompt group.
            unique_prompts = prompt_ids_global.unique(sorted=True)
            for prompt_id in unique_prompts:
                prompt_mask = prompt_ids_global == prompt_id
                prompt_indices = torch.nonzero(prompt_mask, as_tuple=True)[0]
                if prompt_indices.numel() <= 1:
                    continue

                prompt_masks = binary_tracker_low_res_masks_global[prompt_indices]
                prompt_last_occ = last_occluded_prev[prompt_indices]
                prompt_obj_ids = [obj_ids_global[idx] for idx in prompt_indices.tolist()]
                prompt_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                    prompt_masks,
                    prompt_last_occ,
                    prompt_obj_ids,
                    reverse,
                )
                to_suppress[prompt_indices] = prompt_suppress

            # Update metadata with occlusion information
            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # Slice out the last occluded frame for each object
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1] for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # Zero out suppressed masks before memory encoding
            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _suppress_shrinked_masks(self, pred_masks, new_pred_masks, shrink_threshold=0.3):
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks_after

    def _suppress_object_pw_area_shrinkage(
        self,
        pred_masks,
        prompt_ids: list[int] | None = None,
    ):
        """
        This function suppresses masks that shrink in area after applying pixelwise non-overlapping constraints.
        When `prompt_ids` are provided, constraints are enforced independently per prompt group.
        """
        if prompt_ids is None:
            return self._suppress_object_pw_area_shrinkage_impl(pred_masks)

        if len(prompt_ids) != pred_masks.size(0):
            raise ValueError("prompt_ids must have the same length as pred_masks")

        prompt_ids_tensor = torch.tensor(prompt_ids, device=pred_masks.device, dtype=torch.long)
        pred_masks_grouped = pred_masks.clone()
        for prompt_id in prompt_ids_tensor.unique(sorted=True):
            indices = torch.nonzero(prompt_ids_tensor == prompt_id, as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            pred_masks_grouped[indices] = self._suppress_object_pw_area_shrinkage_impl(pred_masks_grouped[indices])
        return pred_masks_grouped

    def _suppress_object_pw_area_shrinkage_impl(self, pred_masks):
        if pred_masks.size(0) <= 1:
            return pred_masks

        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(pred_masks)
        pred_masks = self._suppress_shrinked_masks(pred_masks, pixel_level_non_overlapping_masks)
        return pred_masks

    def _tracker_update_memories(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        low_res_masks: Tensor,
        reconditioned_masks: dict[int, Tensor] | None = None,
    ):
        """
        Run Sam3Tracker memory encoder, enforcing non-overlapping constraints globally.
        Now with batched memory encoding for better performance.

        Args:
            inference_session: The inference session state
            frame_idx: Current frame index
            low_res_masks: Low-resolution tracker masks for all objects
            reconditioned_masks: Optional dict of obj_idx -> high_res_mask for objects that
                                should use detection masks instead of tracker masks
        """
        if len(inference_session.obj_ids) == 0:
            return

        if reconditioned_masks is None:
            reconditioned_masks = {}
        # Interpolate tracker masks to high resolution
        high_res_masks = low_res_masks.unsqueeze(1)

        # Override with detection masks for reconditioned objects
        for obj_idx, recond_mask in reconditioned_masks.items():
            high_res_masks[obj_idx] = recond_mask.float()
            # Mark as conditioning frame for reconditioned objects
            output_dict = inference_session.output_dict_per_obj[obj_idx]
            if frame_idx in output_dict["non_cond_frame_outputs"]:
                current_out = output_dict["non_cond_frame_outputs"].pop(frame_idx)
                output_dict["cond_frame_outputs"][frame_idx] = current_out

        # Apply non-overlapping constraints before memory encoding.
        # Constraints are enforced independently per prompt group.
        # Every object ID has a prompt_id assigned when it's created.
        prompt_ids_for_objects = [
            inference_session.obj_id_to_prompt_id[obj_id] for obj_id in inference_session.obj_ids
        ]
        high_res_masks = self._suppress_object_pw_area_shrinkage(high_res_masks, prompt_ids_for_objects)
        # Use mask areas as a proxy for object scores
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        # Run memory encoder in batch for all objects at once
        num_objects = len(inference_session.obj_ids)
        object_score_logits_batched = object_score_logits.unsqueeze(-1)  # Shape: (num_objects, 1)

        # Encode memories for all objects in one batch call
        maskmem_features_batched, maskmem_pos_enc_batched = self.run_memory_encoder(
            inference_session,
            frame_idx,
            high_res_masks,  # Shape: (num_objects, 1, H, W)
            object_score_logits_batched,  # Shape: (num_objects, 1)
        )

        # Split and store encoded memories per object
        for obj_idx in range(num_objects):
            output_dict = inference_session.output_dict_per_obj[obj_idx]
            # Extract per-object memory from batched result
            maskmem_features = maskmem_features_batched[:, obj_idx : obj_idx + 1]
            maskmem_pos_enc = maskmem_pos_enc_batched[:, obj_idx : obj_idx + 1]

            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                current_out = output_dict[storage_key][frame_idx]
                current_out["maskmem_features"] = maskmem_features
                current_out["maskmem_pos_enc"] = maskmem_pos_enc

    def run_tracker_update_planning_phase(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        reverse: bool,
        det_out: dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        det_idx_to_prompt_id: dict[int, int],
        streaming: bool = False,
    ):
        # initialize new metadata from previous metadata (its values will be updated later)
        tracker_metadata_new = {
            "obj_ids": deepcopy(inference_session.obj_ids),
            "obj_id_to_score": deepcopy(inference_session.obj_id_to_score),
            "obj_id_to_tracker_score_frame_wise": deepcopy(inference_session.obj_id_to_tracker_score_frame_wise),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(inference_session.max_obj_id),
        }

        # Initialize reconditioned_obj_ids early to avoid UnboundLocalError
        reconditioned_obj_ids = set()

        # Step 1: make the update plan and resolve heuristics
        det_mask_preds: Tensor = det_out["mask"]  # low-res mask logits
        det_scores: Tensor = det_out["scores"].float()  # Keep as tensor!
        # det_idx_to_prompt_id maps every detection index to its prompt_id (created by _merge_detections_from_prompts).
        det_prompt_ids = (
            torch.tensor(
                [det_idx_to_prompt_id[idx] for idx in range(det_mask_preds.size(0))],
                device=det_mask_preds.device,
                dtype=torch.long,
            )
            if det_mask_preds.size(0) > 0
            else torch.empty(0, device=det_mask_preds.device, dtype=torch.long)
        )
        # Get prompt IDs for tracked objects.
        trk_prompt_ids = (
            torch.tensor(
                [inference_session.obj_id_to_prompt_id[obj_id] for obj_id in inference_session.obj_ids],
                device=tracker_low_res_masks_global.device
                if tracker_low_res_masks_global.numel() > 0
                else det_mask_preds.device,
                dtype=torch.long,
            )
            if tracker_low_res_masks_global.numel() > 0
            else torch.empty(0, device=det_mask_preds.device, dtype=torch.long)
        )
        # a) match FA and SAM2 masks and find new objects
        (
            new_det_out_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        ) = self._associate_det_trk(
            det_masks=det_mask_preds,
            det_scores=det_scores,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=inference_session.obj_ids,
            det_prompt_ids=det_prompt_ids,
            trk_prompt_ids=trk_prompt_ids,
        )

        # check whether we've hit the maximum number of objects we can track (and if so, drop some detections)
        prev_obj_num = len(inference_session.obj_ids)
        new_det_num = len(new_det_out_inds)
        num_obj_dropped_due_to_limit = 0
        if prev_obj_num + new_det_num > self.max_num_objects:
            logger.warning(f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}")
            new_det_num_to_keep = self.max_num_objects - prev_obj_num
            num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
            # Keep top scoring detections
            new_det_inds_tensor = torch.tensor(new_det_out_inds, dtype=torch.long, device=det_scores.device)
            scores_for_new_dets = det_scores[new_det_inds_tensor]
            _, top_inds = torch.topk(scores_for_new_dets, k=new_det_num_to_keep, largest=True)
            new_det_out_inds = [new_det_out_inds[i] for i in top_inds]
            new_det_num = len(new_det_out_inds)

        # assign object IDs to new detections
        new_det_start_obj_id = inference_session.max_obj_id + 1
        new_det_obj_ids = list(range(new_det_start_obj_id, new_det_start_obj_id + new_det_num))

        # Assign prompt IDs to new objects based on which prompt detected them.
        for obj_id, det_idx in zip(new_det_obj_ids, new_det_out_inds):
            prompt_id = det_idx_to_prompt_id[det_idx]
            inference_session.obj_id_to_prompt_id[obj_id] = prompt_id

        # b) handle hotstart heuristics to remove objects
        extra_metadata_new = deepcopy(
            {
                "obj_first_frame_idx": inference_session.obj_first_frame_idx,
                "unmatched_frame_inds": inference_session.unmatched_frame_inds,
                "trk_keep_alive": inference_session.trk_keep_alive,
                "overlap_pair_to_frame_inds": inference_session.overlap_pair_to_frame_inds,
                "removed_obj_ids": inference_session.removed_obj_ids,
                "suppressed_obj_ids": inference_session.suppressed_obj_ids,
            }
        )

        obj_ids_newly_removed, extra_metadata_new = self._process_hotstart(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            new_det_obj_ids=new_det_obj_ids,
            empty_trk_obj_ids=empty_trk_obj_ids,
            unmatched_trk_obj_ids=unmatched_trk_obj_ids,
            extra_metadata=extra_metadata_new,
            streaming=streaming,
        )
        tracker_metadata_new["extra_metadata"] = extra_metadata_new

        # Step 3 (optional): prepare reconditioned masks based on high-confidence detections
        reconditioned_masks = {}
        reconditioned_obj_ids = set()
        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        if should_recondition_periodic:
            reconditioned_masks, reconditioned_obj_ids = self._prepare_recondition_masks(
                inference_session=inference_session,
                frame_idx=frame_idx,
                det_out=det_out,
                trk_masks=tracker_low_res_masks_global,
                trk_id_to_max_iou_high_conf_det=trk_id_to_max_iou_high_conf_det,
                tracker_obj_scores_global=tracker_obj_scores_global,
            )

        tracker_update_plan = {
            "new_det_out_inds": new_det_out_inds,  # List[int]
            "new_det_obj_ids": new_det_obj_ids,  # List[int]
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # List[int]
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,  # dict
            "reconditioned_obj_ids": reconditioned_obj_ids,  # set
        }

        # Step 4: Run SAM2 memory encoder on the current frame's prediction masks
        # This uses tracker masks for most objects, but detection masks for reconditioned objects
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                # NOTE: tracker_low_res_masks_global is updated in-place then returned
                tracker_low_res_masks_global = self._suppress_overlapping_based_on_recent_occlusion(
                    inference_session=inference_session,
                    frame_idx=frame_idx,
                    tracker_low_res_masks_global=tracker_low_res_masks_global,
                    tracker_metadata_new=tracker_metadata_new,
                    obj_ids_newly_removed=obj_ids_newly_removed,
                    reverse=reverse,
                )

            # Unified memory encoding: uses detection masks for reconditioned objects
            self._tracker_update_memories(
                inference_session=inference_session,
                frame_idx=frame_idx,
                low_res_masks=tracker_low_res_masks_global,
                reconditioned_masks=reconditioned_masks,
            )

        # Step 5: update the SAM2 metadata based on the update plan
        updated_obj_ids = tracker_metadata_new["obj_ids"]
        if len(new_det_obj_ids) > 0:
            updated_obj_ids = updated_obj_ids + new_det_obj_ids
        if len(obj_ids_newly_removed) > 0:
            updated_obj_ids = [obj_id for obj_id in updated_obj_ids if obj_id not in obj_ids_newly_removed]
        tracker_metadata_new["obj_ids"] = updated_obj_ids

        # update object scores and the maximum object ID assigned so far
        if len(new_det_obj_ids) > 0:
            # Index tensor with list of indices and convert to list
            new_det_scores = det_scores[
                torch.tensor(new_det_out_inds, dtype=torch.long, device=det_scores.device)
            ].tolist()
            tracker_metadata_new["obj_id_to_score"].update(zip(new_det_obj_ids, new_det_scores))
            # tracker scores are not available for new objects, use det score instead.
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, new_det_scores)
            )
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                max(new_det_obj_ids),
            )
        # for removed objects, we set their scores to a very low value (-1e4) but still
        # keep them in "obj_id_to_score" (it's easier to handle outputs this way)
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)

        return tracker_update_plan, tracker_metadata_new

    def _tracker_add_new_objects(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        new_obj_ids: list[int],
        new_obj_masks: Tensor,
        reverse: bool = False,
    ):
        """Add a new object to SAM2 inference states."""
        new_obj_masks = new_obj_masks >= 0.5
        for obj_id, mask in zip(new_obj_ids, new_obj_masks):
            obj_idx = inference_session.obj_id_to_idx(obj_id)
            inference_session.add_mask_inputs(obj_idx, frame_idx, mask.unsqueeze(0).unsqueeze(0))

        inference_session.obj_with_new_inputs = list(new_obj_ids)

        self.tracker_model(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            run_mem_encoder=True,
        )

    def run_tracker_update_execution_phase(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        det_out: dict[str, Tensor],
        tracker_update_plan: dict,
        reverse: bool = False,
    ):
        # initialize tracking scores with detection scores
        new_det_out_inds: list[int] = tracker_update_plan["new_det_out_inds"]
        new_det_obj_ids: list[int] = tracker_update_plan["new_det_obj_ids"]
        obj_ids_newly_removed: set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # Step 1: add new objects from FA detection to SAM2 inference states
        if len(new_det_out_inds) > 0:
            new_det_out_inds_t = torch.tensor(new_det_out_inds, dtype=torch.long)
            new_det_masks: Tensor = det_out["mask"][new_det_out_inds_t]
            # initialize SAM2 with new object masks
            self._tracker_add_new_objects(
                inference_session=inference_session,
                frame_idx=frame_idx,
                new_obj_ids=new_det_obj_ids,
                new_obj_masks=new_det_masks,
                reverse=reverse,
            )

        # Step 2: remove from SAM2 inference states those objects removed by heuristics
        for obj_id in obj_ids_newly_removed:
            inference_session.remove_object(obj_id, strict=False)  # implement remove_object in inference_session?

    def build_outputs(
        self,
        inference_session: Sam3VideoInferenceSession,
        det_out: dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_update_plan: dict,
        reconditioned_obj_ids: set | None = None,
    ):
        """
        Build output dictionary with low-resolution masks.
        Interpolation to video resolution is handled by the processor.

        Returns:
            obj_id_to_mask: dict mapping obj_id to low-res mask tensor (1, H_low, W_low)
        """
        new_det_out_inds: list[int] = tracker_update_plan["new_det_out_inds"]
        new_det_obj_ids: list[int] = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> low-res mask tensor

        # Part 1: masks from tracker propagation (existing objects)
        existing_masklet_obj_ids = inference_session.obj_ids
        for obj_id, mask in zip(existing_masklet_obj_ids, tracker_low_res_masks_global):
            obj_id_to_mask[int(obj_id)] = mask.unsqueeze(0)  # (1, H_low, W_low)

        # Part 2: masks from new detections
        if len(new_det_out_inds) > 0:
            new_det_out_inds_t = torch.tensor(new_det_out_inds, dtype=torch.long, device=det_out["mask"].device)
            new_det_low_res_masks = det_out["mask"][new_det_out_inds_t]
            # Apply hole filling to new detection masks
            new_det_low_res_masks = fill_holes_in_mask_scores(
                new_det_low_res_masks.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            ).squeeze(1)

            for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
                obj_id_to_mask[int(obj_id)] = mask.unsqueeze(0)  # (1, H_low, W_low)

        # Part 3: Override masks for reconditioned objects using detection masks
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get("trk_id_to_max_iou_high_conf_det", {})

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)
                if det_idx is not None:
                    det_mask = det_out["mask"][det_idx].unsqueeze(0)  # (1, H_low, W_low)
                    obj_id_to_mask[int(obj_id)] = det_mask

        return obj_id_to_mask

    def _merge_detections_from_prompts(
        self,
        all_detections: dict[int, dict[str, Tensor]],
        inference_session: Sam3VideoInferenceSession,
    ) -> tuple[dict[str, Tensor], dict[int, int]]:
        """
        Merge detections from multiple prompts into a single detection output.
        Assigns unique object IDs and tracks which prompt detected each object.

        Args:
            all_detections: Dictionary mapping prompt_id to detection outputs
            inference_session: Session to track obj_id -> prompt_id mapping

        Returns:
            Tuple of (merged_det_out, det_idx_to_prompt_id) where det_idx_to_prompt_id
            maps detection index in the merged output to the prompt that produced it.
        """
        merged_bboxes, merged_masks, merged_scores = [], [], []
        det_idx_to_prompt_id = {}
        det_idx = 0

        for prompt_id, det_out in all_detections.items():
            num_dets = len(det_out["bbox"])
            if num_dets > 0:
                merged_bboxes.append(det_out["bbox"])
                merged_masks.append(det_out["mask"])
                merged_scores.append(det_out["scores"])
                for i in range(num_dets):
                    det_idx_to_prompt_id[det_idx + i] = prompt_id
                det_idx += num_dets

        if merged_bboxes:
            merged_det_out = {
                "bbox": torch.cat(merged_bboxes),
                "mask": torch.cat(merged_masks),
                "scores": torch.cat(merged_scores),
            }
        else:
            device = inference_session.inference_device
            merged_det_out = {
                "bbox": torch.zeros(0, 4, device=device),
                "mask": torch.zeros(0, self.low_res_mask_size, self.low_res_mask_size, device=device),
                "scores": torch.zeros(0, device=device),
            }

        return merged_det_out, det_idx_to_prompt_id

    def _det_track_one_frame(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int,
        reverse: bool,
        streaming: bool = False,
    ):
        """
        This function handles one-step inference for the DenseTracking model.

        - `inference_session` contains all the information needed for inference, including the input video frames, text prompts, and any other relevant metadata
        - The function processes detection and tracking for a single frame
        - `streaming` indicates whether this is streaming inference mode (frames provided one at a time)
        """

        pixel_values = inference_session.get_frame(frame_idx).unsqueeze(0)
        vision_embeds = self.detector_model.get_vision_features(pixel_values=pixel_values)

        # Step 1: run detection for all prompts (efficiently reusing vision embeddings)
        # Returns dict mapping prompt_id to detection outputs
        all_detections = self.run_detection(
            inference_session=inference_session,
            vision_embeds=vision_embeds,
        )

        # Merge detections from all prompts into single output for tracking
        det_out, det_idx_to_prompt_id = self._merge_detections_from_prompts(all_detections, inference_session)

        # share the vision encoder outputs from the detector to the tracker
        vision_feats, vision_pos_embeds = self.get_vision_features_for_tracker(
            vision_embeds=vision_embeds,
        )
        inference_session.cache.cache_vision_features(
            frame_idx, {"vision_feats": vision_feats, "vision_pos_embeds": vision_pos_embeds}
        )

        # Step 2: propagate SAM2 states to get the SAM2 prediction masks.
        # The returned `tracker_low_res_masks_global` contains the masklet predictions.
        # Note that this step only runs the SAM2 propagation step, but doesn't encode new memory for the predicted masks;
        # we defer memory encoding to `run_tracker_update_execution_phase` after resolving all heuristics.
        tracker_low_res_masks_global, tracker_obj_scores_global = self.run_tracker_propagation(
            inference_session=inference_session, frame_idx=frame_idx, reverse=reverse
        )

        # Step 3: based on detection outputs and the propagated SAM2 prediction masks, we make plans
        # for SAM2 masklet updates (i.e. which objects to add and remove, etc).
        # We also run SAM2 memory encoder in this step to resolve non-overlapping constraints.
        # **This step should involve all the heuristics needed for any updates.**
        # This step also generates the new masklet metadata `tracker_metadata_new` (based on its previous version).
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

        # Step 4: based on `tracker_update_plan`, execute the update w.r.t. the tracker states
        self.run_tracker_update_execution_phase(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_update_plan=tracker_update_plan,
        )

        # Step 5: finally, build the outputs for this frame
        reconditioned_obj_ids = tracker_update_plan["reconditioned_obj_ids"]
        obj_id_to_mask = self.build_outputs(
            inference_session=inference_session,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        # add tracker scores to metadata, it should be fired for frames except the first frame
        if tracker_obj_scores_global.shape[0] > 0:
            # Convert tracker_obj_scores_global to sigmoid scores before updating
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = inference_session.obj_ids
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )

        return (
            obj_id_to_mask,  # a dict: obj_id --> output mask
            obj_id_to_score,  # a dict: obj_id --> output score (prob)
            tracker_metadata_new,
            tracker_obj_scores_global,  # a dict: obj_id --> tracker frame-level scores
        )

    @torch.inference_mode()
    @auto_docstring(custom_intro="Propagate the objects through a streamed video frame.")
    def forward(
        self,
        inference_session: Sam3VideoInferenceSession,
        frame_idx: int | None = None,
        frame: torch.Tensor | None = None,
        reverse: bool = False,
        **kwargs,
    ):
        r"""
        inference_session (`Sam3VideoInferenceSession`):
            The video inference session object.
        frame_idx (`int`, *optional*):
            The index of the frame on which to run inference. No need to provide when inferring
            on a new streamed frame.
        frame (`torch.Tensor`, *optional*):
            The frame to process. Provide when streaming.
        reverse (`bool`, *optional*, defaults to `False`):
            Whether to propagate in reverse.
        """
        if frame is not None:
            frame_idx = inference_session.add_new_frame(frame, frame_idx)

        if frame_idx is None:
            raise ValueError("frame_idx must be provided when frame is not provided for streaming.")

        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            _,
        ) = self._det_track_one_frame(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            streaming=frame is not None,
        )
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        # inference_session.previous_stages_out[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        extra_metadata = tracker_metadata_new["extra_metadata"]
        removed_obj_ids = extra_metadata["removed_obj_ids"]

        # Update inference session state
        inference_session.obj_id_to_score = obj_id_to_score
        inference_session.obj_id_to_tracker_score_frame_wise = tracker_metadata_new[
            "obj_id_to_tracker_score_frame_wise"
        ]
        inference_session.obj_id_to_last_occluded = tracker_metadata_new["obj_id_to_last_occluded"]
        inference_session.max_obj_id = tracker_metadata_new["max_obj_id"]
        inference_session.obj_ids = list(tracker_metadata_new["obj_ids"])

        inference_session.obj_first_frame_idx = extra_metadata["obj_first_frame_idx"]
        inference_session.unmatched_frame_inds = extra_metadata["unmatched_frame_inds"]
        inference_session.trk_keep_alive = extra_metadata["trk_keep_alive"]
        inference_session.overlap_pair_to_frame_inds = extra_metadata["overlap_pair_to_frame_inds"]
        inference_session.removed_obj_ids = removed_obj_ids
        inference_session.suppressed_obj_ids[frame_idx] = extra_metadata["suppressed_obj_ids"][frame_idx]

        return Sam3VideoSegmentationOutput(
            object_ids=list(tracker_metadata_new["obj_ids"]),
            obj_id_to_mask=obj_id_to_mask,
            obj_id_to_score=obj_id_to_score,
            obj_id_to_tracker_score=tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
            removed_obj_ids=removed_obj_ids,
            suppressed_obj_ids=extra_metadata["suppressed_obj_ids"][frame_idx],
            frame_idx=frame_idx,
        )

    def _get_processing_order(
        self,
        inference_session: Sam3VideoInferenceSession,
        start_frame_idx: int,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        num_frames = inference_session.num_frames
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = start_frame_idx - max_frame_num_to_track
            end_frame_idx = max(end_frame_idx, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = start_frame_idx + max_frame_num_to_track
            end_frame_idx = min(end_frame_idx, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    @torch.inference_mode()
    @auto_docstring(
        custom_intro="""
        Propagate the prompts to get grounding results for the entire video. Used when initializing an inference session with a whole video.
        Yields Sam3VideoSegmentationOutput for each frame.
        """
    )
    def propagate_in_video_iterator(
        self,
        inference_session: Sam3VideoInferenceSession,
        start_frame_idx: int = 0,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        show_progress_bar: bool = False,
    ) -> Iterator[Sam3VideoSegmentationOutput]:
        r"""
        inference_session (`Sam3VideoInferenceSession`):
            The video inference session object.
        start_frame_idx (`int`, *optional*, defaults to `0`):
            The starting frame index for propagation.
        max_frame_num_to_track (`int`, *optional*):
            The maximum number of frames to track. If not provided, all frames in the video will be tracked.
        reverse (`bool`, *optional*, defaults to `False`):
            Whether to propagate in reverse.
        show_progress_bar (`bool`, *optional*, defaults to `False`):
            Whether to show a progress bar during propagation.
        """
        processing_order, end_frame_idx = self._get_processing_order(
            inference_session,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )

        hotstart_buffer = []
        for frame_idx in tqdm(processing_order, desc="propagate in video", disable=not show_progress_bar):
            out = self(inference_session=inference_session, frame_idx=frame_idx, reverse=reverse)

            if self.hotstart_delay > 0:
                # accumulate the outputs for the first `hotstart_delay` frames
                hotstart_buffer.append(out)
                # update the object IDs removed by hotstart so that we don't output them
                inference_session.hotstart_removed_obj_ids.update(out.removed_obj_ids)

                if frame_idx == end_frame_idx:
                    # we reached the end of propagation -- yield all frames in the buffer
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    # we have enough frames -- yield and remove the first (oldest) frame from the buffer
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    # not enough frames yet -- skip yielding
                    yield_list = []
            else:
                yield_list = [out]  # output the current frame

            yield from yield_list


@torch.jit.script
def fast_diag_box_iou(boxes1, boxes2):
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    area1 = (box1_xy - box1_XY).prod(-1)
    area2 = (box2_xy - box2_XY).prod(-1)

    lt = torch.max(box1_XY, box2_XY)
    rb = torch.min(box1_xy, box2_xy)

    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape

    # Flatten masks: (N, 1, H*W) and (1, M, H*W)
    pred_flat = pred_masks.view(N, 1, H * W)
    gt_flat = gt_masks.view(1, M, H * W)

    # Compute intersection and union: (N, M)
    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    ious = intersection / union.clamp(min=1)
    return ious  # shape: (N, M)


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Args:
      - pred_probs: (num_det,) float Tensor, containing the score (probability) of each detection
      - pred_masks: (num_det, H_mask, W_mask) float Tensor, containing the binary segmentation mask of each detection
      - prob_threshold: float, score threshold to prefilter detections (NMS is performed on detections above threshold)
      - iou_threshold: float, mask IoU threshold for NMS

    Returns:
     - keep: (num_det,) bool Tensor, indicating whether each detection is kept after score thresholding + NMS
    """
    # prefilter the detections with prob_threshold ("valid" are those above prob_threshold)
    is_valid = pred_probs > prob_threshold  # (num_det,)
    probs = pred_probs[is_valid]  # (num_valid,)
    masks_binary = pred_masks[is_valid] > 0  # (num_valid, H_mask, W_mask)
    if probs.numel() == 0:
        return is_valid  # no valid detection, return empty keep mask

    ious = mask_iou(masks_binary, masks_binary)  # (num_valid, num_valid)

    # Try to use kernels for NMS, fallback to keeping all valid detections if unavailable
    _load_cv_utils_kernel_once()
    if not cv_utils_kernel:
        return is_valid  # Fallback: keep all valid detections without NMS

    try:
        kept_inds = cv_utils_kernel.generic_nms(ious, probs, iou_threshold, use_iou_matrix=True)
    except Exception as e:
        logger.warning_once(f"Failed to run NMS using kernels library: {e}. NMS post-processing will be skipped.")
        return is_valid  # Fallback: keep all valid detections without NMS

    # valid_inds are the indices among `probs` of valid detections before NMS (or -1 for invalid)
    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)  # (num_det,)
    keep = torch.isin(valid_inds, kept_inds)  # (num_det,)
    return keep


def fill_holes_in_mask_scores(mask, max_area, fill_holes=True, remove_sprinkles=True):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    Holes are those small connected components in either background or foreground.

    Note that it relies on the "cc_torch" package to find connected components fast. You can
    install it via the following command (`TORCH_CUDA_ARCH_LIST=8.0` is for A100 GPUs):
    ```
    pip uninstall -y cc_torch; TORCH_CUDA_ARCH_LIST=8.0 9.0 pip install git+https://github.com/ronghanghu/cc_torch
    ```
    Otherwise, it will fallback to a slightly slower triton implementation, or skimage if the tensor is on cpu
    """

    if max_area <= 0:
        return mask  # nothing to fill in this case

    if fill_holes:
        # We remove small connected components in background by changing them to foreground
        # with a small positive mask score (0.1).
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        # We remove small connected components in foreground by changing them to background
        # with a small negative mask score (-0.1). Here we only remove connected components
        # whose areas are under both `max_area` and half of the entire mask's area. This
        # removes sprinkles while avoids filtering out tiny objects that we want to track.
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask


def _get_connected_components_with_padding(mask):
    """Get connected components from masks (possibly padding them to an even size)."""
    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape

    # Try to use kernels for connected components, fallback if unavailable
    _load_cv_utils_kernel_once()
    if not cv_utils_kernel:
        # Fallback: return dummy labels and counts that won't trigger filtering
        labels = torch.zeros_like(mask, dtype=torch.int32)
        counts = torch.full_like(mask, fill_value=mask.shape[2] * mask.shape[3] + 1, dtype=torch.int32)
        return labels, counts

    # make sure both height and width are even (to be compatible with cc_torch)
    pad_h = H % 2
    pad_w = W % 2

    try:
        if pad_h == 0 and pad_w == 0:
            labels, counts = cv_utils_kernel.cc_2d(mask.contiguous(), get_counts=True)
        else:
            # pad the mask to make its height and width even
            # padding format is (padding_left,padding_right,padding_top,padding_bottom)
            mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
            labels, counts = cv_utils_kernel.cc_2d(mask_pad.contiguous(), get_counts=True)
            labels = labels[:, :, :H, :W]
            counts = counts[:, :, :H, :W]
    except Exception as e:
        logger.warning_once(
            f"Failed to compute connected components using kernels library: {e}. "
            "Hole filling and sprinkle removal will be skipped."
        )
        # Fallback: return dummy labels and counts that won't trigger filtering
        labels = torch.zeros_like(mask, dtype=torch.int32)
        counts = torch.full_like(mask, fill_value=H * W + 1, dtype=torch.int32)
        return labels, counts

    return labels, counts


__all__ = [
    "Sam3VideoModel",
    "Sam3VideoPreTrainedModel",
    "Sam3VideoInferenceSession",
    "Sam3VideoSegmentationOutput",
]
