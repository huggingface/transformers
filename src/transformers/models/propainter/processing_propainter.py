# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for ProPainter.
"""

import os
import sys


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from ...feature_extraction_utils import BatchFeature
from ...image_utils import VideoInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from ...utils import logging
from ..auto import AutoImageProcessor


logger = logging.get_logger(__name__)


class ProPainterProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "video_kwargs": {
            "video_painting_mode": "video_inpainting",
            "mask_dilation": 4,
        },
    }


class ProPainterProcessor(ProcessorMixin):
    r"""
    Constructs a ProPainter processor which wraps and abstract a ProPainter video processor into a single processor.

    [`ProPainterProcessor`] offers all the functionalities of [`ProPainterVideoProcessor`]. See the [`~ProPainterVideoProcessor.__call__`],
    for more information.

    Args:
        video_processor ([`ProPainterVideoProcessor`], *optional*):
            The video processor is a required input.
    """

    attributes = ["video_processor"]
    video_processor_class = "ProPainterVideoProcessor"

    def __init__(
        self,
        video_processor=None,
        **kwargs,
    ):
        super().__init__(video_processor)

    def __call__(
        self,
        videos: VideoInput = None,
        masks: VideoInput = None,
        **kwargs: Unpack[ProPainterProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several video(s) and their respective masks for all the frames. To prepare the video(s)
        and masks, this method forwards the `videos`, `masks` and `kwrags` arguments to ProPainterProcessor's 
        [`~ProPainterProcessor.__call__`] if `videos` and `masks` are not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The video or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            masks (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The masks(for all frames of a single video) or batch of masks to be prepared. Each set of masks for a single video
                can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **flow_masks** -- Pixel values of flow masks for all frames of a video input to be fed to a model. Returned when `masks` is not `None`.
            - **masks_dilated** -- Pixel values of dilated masks for all frames of a video input to be fed to a model. Returned when `masks` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            ProPainterProcessorKwargs,
            **kwargs,
        )

        video_inputs = {}

        if videos is not None and masks is not None:
            video_inputs = self.video_processor(videos, masks = masks, **output_kwargs["videos_kwargs"])

        return BatchFeature(data={**video_inputs})

    @property
    # Adapted from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        video_processor_input_names = self.video_processor.model_input_names
        return list(dict.fromkeys(video_processor_input_names))

    # override to save video-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        video_processor_path = os.path.join(save_directory, "video_processor")
        self.video_processor.save_pretrained(video_processor_path)

        video_processor_present = "video_processor" in self.attributes
        if video_processor_present:
            self.attributes.remove("video_processor")

        outputs = super().save_pretrained(save_directory, **kwargs)

        if video_processor_present:
            self.attributes += ["video_processor"]
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]

        try:
            video_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, subfolder="video_processor"
            )
            processor.video_processor = video_processor
        except EnvironmentError:
            # this means users are using prev version of saved processor where we had only one preprocessor_config.json
            # for loading back that should work and load a ProPainterVideoProcessor class
            logger.info(
                "You are loading `ProPainterProcessor` but the indicated `path` doesn't contain a folder called "
                "`video_processor`. It is strongly recommended to load and save the processor again so the video processor is saved "
                "in a separate config."
            )

        return processor
