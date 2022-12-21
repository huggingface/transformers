# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

from ...processing_utils import ProcessorMixin


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

        self.current_image_processor = image_processor
        self.current_feature_extractor = feature_extractor

    def __call__(
        self, visual_inputs=None, audio_inputs=None, visual_inputs_mixed=None, sampling_rate=None, *args, **kwargs
    ):
        """
        Forwards the `visual_inputs` argument to TvltImageProcessor's [`~TvltImageProcessor.preprocess`]
        and the `audio_inputs` argument to TvltFeatureExtractor's [`~TvltFeatureExtractor.__call__`].
        Please refer to the docstring of the above two methods for more information.
        """

        if visual_inputs is None and audio_inputs is None:
            raise ValueError("You need to specify either an `visual_inputs` or `audio_inputs` input to process.")

        if visual_inputs is not None:
            visual_inputs_dict = self.current_image_processor(visual_inputs, *args, **kwargs)
            visual_inputs_mixed_dict = None
        if visual_inputs_mixed is not None:
            visual_inputs_mixed_dict = self.current_image_processor(
                visual_inputs_mixed, is_mixed=True, *args, **kwargs
            )
        if audio_inputs is not None:
            audio_inputs_dict = self.current_feature_extractor(
                audio_inputs, *args, sampling_rate=sampling_rate, **kwargs
            )

        if audio_inputs is None:
            output_dict = visual_inputs_dict
        elif visual_inputs is None:
            output_dict = audio_inputs_dict
            return output_dict
        else:
            output_dict = dict(visual_inputs_dict.items() + audio_inputs.items())
        if visual_inputs_mixed_dict is not None:
            output_dict.update(visual_inputs_mixed_dict)
        return output_dict

    @property
    def model_input_names(self):
        image_processor_input_names = self.current_image_processor.model_input_names
        feature_extractor_input_names = self.current_feature_extractor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + feature_extractor_input_names))
