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

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class Sam3VideoConfig(PreTrainedConfig):
    r"""
    Configuration class for [`Sam3VideoModel`]. This combines configurations for the detector (Sam3) and tracker
    (Sam2Video) components, along with detection-tracking fusion hyperparameters.

    Instantiating a configuration defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    This model integrates detection and tracking with various fusion heuristics including NMS, association,
    hotstart, reconditioning, and occlusion handling.

    Args:
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

    def __init__(
        self,
        detector_config=None,
        tracker_config=None,
        initializer_range=0.02,
        low_res_mask_size=288,
        # Detection-tracking fusion hyperparameters
        score_threshold_detection=0.5,
        det_nms_thresh=0.1,
        assoc_iou_thresh=0.1,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.7,
        recondition_on_trk_masks=True,
        # Hotstart parameters
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=True,
        # Keep-alive parameters
        init_trk_keep_alive=30,
        max_trk_keep_alive=30,
        min_trk_keep_alive=-1,
        # Occlusion and overlap handling
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        decrease_trk_keep_alive_for_empty_masklets=False,
        # Mask post-processing
        fill_hole_area=16,
        # Object tracking limits
        max_num_objects=10000,
        # Reconditioning parameters
        recondition_every_nth_frame=16,
        high_conf_thresh=0.8,
        high_iou_thresh=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize detector config (Sam3)
        if detector_config is None:
            detector_config = {}
            logger.info("detector_config is None. Initializing the Sam3Config with default values.")
        if isinstance(detector_config, dict):
            detector_config["model_type"] = detector_config.get("model_type", "sam3")
            self.detector_config = CONFIG_MAPPING[detector_config["model_type"]](**detector_config)
        elif isinstance(detector_config, PreTrainedConfig):
            self.detector_config = detector_config
        else:
            raise ValueError(f"detector_config must be a dict or Sam3Config, got {type(detector_config)}")

        # Initialize tracker config (Sam2Video)
        if tracker_config is None:
            tracker_config = {}
            logger.info("tracker_config is None. Initializing the Sam3TrackerVideoConfig with default values.")
        if isinstance(tracker_config, dict):
            tracker_config["model_type"] = tracker_config.get("model_type", "sam3_tracker_video")
            self.tracker_config = CONFIG_MAPPING[tracker_config["model_type"]](**tracker_config)
        elif isinstance(tracker_config, PreTrainedConfig):
            self.tracker_config = tracker_config
        else:
            raise ValueError(f"tracker_config must be a dict or Sam3TrackerVideoConfig, got {type(tracker_config)}")

        # Model initialization
        self.initializer_range = initializer_range

        self.low_res_mask_size = low_res_mask_size

        # Detection-tracking fusion hyperparameters
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        self.recondition_on_trk_masks = recondition_on_trk_masks

        # Hotstart parameters
        if hotstart_delay > 0:
            if hotstart_unmatch_thresh > hotstart_delay:
                raise ValueError(
                    f"hotstart_unmatch_thresh ({hotstart_unmatch_thresh}) must be <= hotstart_delay ({hotstart_delay})"
                )
            if hotstart_dup_thresh > hotstart_delay:
                raise ValueError(
                    f"hotstart_dup_thresh ({hotstart_dup_thresh}) must be <= hotstart_delay ({hotstart_delay})"
                )
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = suppress_unmatched_only_within_hotstart

        # Keep-alive parameters
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive

        # Occlusion and overlap handling
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.decrease_trk_keep_alive_for_empty_masklets = decrease_trk_keep_alive_for_empty_masklets

        # Mask post-processing
        self.fill_hole_area = fill_hole_area

        # Object tracking limits
        self.max_num_objects = max_num_objects

        # Reconditioning parameters
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.high_conf_thresh = high_conf_thresh
        self.high_iou_thresh = high_iou_thresh

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
