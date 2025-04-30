# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for TVLT.
"""

from ....processing_utils import ProcessorMixin


class TvltProcessor(ProcessorMixin):
    r"""
    Constructs a TVLT processor which wraps a TVLT image processor and TVLT feature extractor into a single processor.

    [`TvltProcessor`] offers all the functionalities of [`TvltImageProcessor`] and [`TvltFeatureExtractor`]. See the
    docstring of [`~TvltProcessor.__call__`] for more information.

    Args:
        image_processor (`TvltImageProcessor`):
            An instance of [`TvltImageProcessor`]. The image processor is a required input.
        feature_extractor (`TvltFeatureExtractor`):
            An instance of [`TvltFeatureExtractor`]. The feature extractor is a required input.
    """

    attributes = ["image_processor", "feature_extractor"]
    image_processor_class = "TvltImageProcessor"
    feature_extractor_class = "TvltFeatureExtractor"

    def __init__(self, image_processor, feature_extractor):
        super().__init__(image_processor=image_processor, feature_extractor=feature_extractor)

        self.image_processor = image_processor
        self.feature_extractor = feature_extractor

    def __call__(
        self,
        images=None,
        audio=None,
        images_mixed=None,
        sampling_rate=None,
        mask_audio=False,
        mask_pixel=False,
        *args,
        **kwargs,
    ):
        """
        Forwards the `images` argument to TvltImageProcessor's [`~TvltImageProcessor.preprocess`] and the `audio`
        argument to TvltFeatureExtractor's [`~TvltFeatureExtractor.__call__`]. Please refer to the docstring of the
        above two methods for more information.
        """

        if images is None and audio is None:
            raise ValueError("You need to specify either an `images` or `audio` input to process.")

        images_mixed_dict = None
        if images is not None:
            images_dict = self.image_processor(images, mask_pixel=mask_pixel, *args, **kwargs)
        if images_mixed is not None:
            images_mixed_dict = self.image_processor(images_mixed, is_mixed=True, *args, **kwargs)
        if audio is not None:
            audio_dict = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, mask_audio=mask_audio, **kwargs
            )

        output_dict = {}
        if audio is not None:
            output_dict.update(audio_dict)
        if images is not None:
            output_dict.update(images_dict)
        if images_mixed_dict is not None:
            output_dict.update(images_mixed_dict)
        return output_dict

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + feature_extractor_input_names))


__all__ = ["TvltProcessor"]
