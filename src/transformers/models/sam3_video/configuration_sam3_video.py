# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""SAM3 Video model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3VideoConfig(PreTrainedConfig):
    r"""
    detector_config (`dict` or `Sam3Config`, *optional*):
        Configuration for the Sam3 detector model. If not provided, default Sam3Config will be used.
    tracker_config (`dict` or `Sam2VideoConfig`, *optional*):
        Configuration for the Sam2Video tracker model. If not provided, default Sam2VideoConfig will be used.
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
    >>> from transformers import Sam3VideoConfig, Sam3VideoModel

    >>> # Initializing a SAM3 Video configuration with default detector and tracker
    >>> configuration = Sam3VideoConfig()

    >>> # Changing image size for custom resolution inference (automatically propagates to all nested configs)
    >>> configuration.image_size = 560

    >>> # Initializing a model from the configuration
    >>> model = Sam3VideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> detector_config = configuration.detector_config
    >>> tracker_config = configuration.tracker_config
    ```
    """

    model_type = "sam3_video"
    is_composition = True
    sub_configs = {
        "detector_config": AutoConfig,
        "tracker_config": AutoConfig,
    }

    detector_config: dict | PreTrainedConfig | None = None
    tracker_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    low_res_mask_size: int = 288
    score_threshold_detection: float = 0.5
    det_nms_thresh: float = 0.1
    assoc_iou_thresh: float = 0.1
    trk_assoc_iou_thresh: float = 0.5
    new_det_thresh: float = 0.7
    recondition_on_trk_masks: bool = True
    hotstart_delay: int = 15
    hotstart_unmatch_thresh: int = 8
    hotstart_dup_thresh: int = 8
    suppress_unmatched_only_within_hotstart: bool = True
    init_trk_keep_alive: int = 30
    max_trk_keep_alive: int = 30
    min_trk_keep_alive: int = -1
    suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.7
    decrease_trk_keep_alive_for_empty_masklets: bool = False
    fill_hole_area: int = 16
    max_num_objects: int = 10000
    recondition_every_nth_frame: int = 16
    high_conf_thresh: float = 0.8
    high_iou_thresh: float = 0.8

    def __post_init__(self, **kwargs):
        if self.detector_config is None:
            self.detector_config = CONFIG_MAPPING["sam3"]()
            logger.info("detector_config is None. Initializing the Sam3Config with default values.")
        if isinstance(self.detector_config, dict):
            self.detector_config["model_type"] = self.detector_config.get("model_type", "sam3")
            self.detector_config = CONFIG_MAPPING[self.detector_config["model_type"]](**self.detector_config)

        if self.tracker_config is None:
            self.tracker_config = CONFIG_MAPPING["sam3_tracker_video"]()
            logger.info("tracker_config is None. Initializing the Sam3TrackerVideoConfig with default values.")
        if isinstance(self.tracker_config, dict):
            self.tracker_config["model_type"] = self.tracker_config.get("model_type", "sam3_tracker_video")
            self.tracker_config = CONFIG_MAPPING[self.tracker_config["model_type"]](**self.tracker_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hotstart_delay > 0:
            if self.hotstart_unmatch_thresh > self.hotstart_delay:
                raise ValueError(
                    f"hotstart_unmatch_thresh ({self.hotstart_unmatch_thresh}) must be <= hotstart_delay ({self.hotstart_delay})"
                )
            if self.hotstart_dup_thresh > self.hotstart_delay:
                raise ValueError(
                    f"hotstart_dup_thresh ({self.hotstart_dup_thresh}) must be <= hotstart_delay ({self.hotstart_delay})"
                )

    @property
    def image_size(self):
        """Image size for the video model."""
        return self.detector_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Recursively propagate the image size to detector and tracker configs."""
        self.detector_config.image_size = value
        self.tracker_config.image_size = value


__all__ = ["Sam3VideoConfig"]
