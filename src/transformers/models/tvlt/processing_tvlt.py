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

from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...utils import TensorType


class TvltProcessor(ProcessorMixin):
    r"""
    Constructs a TVLT processor which wraps a TVLT image processor and TVLT audio feature extractor into a single processor.

    [`TvltProcessor`] offers all the functionalities of [`TvltImageProcessor`] and [`TvltAudioFeatureExtractor`]. See the
    docstring of [`~TvltProcessor.__call__`] for more information.

    Args:
        image_processor (`TvltImageProcessor`):
            An instance of [`TvltImageProcessor`]. The image processor is a required input.
        feature_extractor (`TvltAudioFeatureExtractor`):
            An instance of [`TvltAudioFeatureExtractor`]. The audio feature extractor is a required input.
    """
    image_processor_class = "TvltImageProcessor"
    audio_feature_extractor_class = "TvltAudioFeatureExtractor"

    def __init__(self, image_processor, audio_feature_extractor):
        super().__init__(image_processor, audio_feature_extractor)
        self.current_image_processor = self.image_processor_class
        self.current_audio_processor = self.audio_feature_extractor

    def __call__(self, *args, **kwargs):
        """
        Forwards the `visual_inputs` argument to TvltImageProcessor's [`~TvltImageProcessor.preprocess`] and the `audio_inputs` argument to TvltAudioFeatureExtractor's [`~TvltAudioFeatureExtractor.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            visual_inputs_dict = self.current_image_processor(*args, **kwargs) 
            audio_inputs_dict = self.current_image_processor(*args, **kwargs) 
            return dict(visual_inputs_dict.items() + audio_inputs_dict.items())

        visual_inputs = kwargs.pop("visual_inputs", None)
        audio_inputs = kwargs.pop("audio_inputs", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if visual_inputs is None and audio_inputs is None:
            raise ValueError("You need to specify either an `visual_inputs` or `audio_inputs` input to process.")

        if visual_inputs is not None:
            visual_inputs_dict = self.current_image_processor(visual_inputs, *args, **kwargs)
        if audio_inputs is not None:
            audio_inputs_dict = self.audio_feature_extractor(audio_inputs, *args, sampling_rate=sampling_rate, **kwargs)

        if audio_values is None:
            return visual_inputs_dict
        elif pixel_values is None:
            return audio_inputs_dict
        else:
            return dict(pixel_inputs.items() + audio_inputs.items())

    @property
    def model_input_names(self):
        image_processor_input_names = self.current_image_processor.model_input_names
        audio_feature_extractor_input_names = self.audio_feature_extractor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + audio_feature_extractor_input_names))
