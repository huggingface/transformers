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


class TVLTProcessor(ProcessorMixin):
    r"""
    Constructs a TVLT processor which wraps a TVLT pixel feature extractor and TVLT audio feature extractor into a single processor.

    [`TVLTProcessor`] offers all the functionalities of [`TVLTPixelFeatureExtractor`] and [`TVLTAudioFeatureExtractor`]. See the
    docstring of [`~TVLTProcessor.__call__`] and [`~TVLTProcessor.decode`] for more information.

    Args:
        feature_extractor (`TVLTPixelFeatureExtractor`):
            An instance of [`TVLTPixelFeatureExtractor`]. The pixel feature extractor is a required input.
        feature_extractor (`TVLTAudioFeatureExtractor`):
            An instance of [`TVLTAudioFeatureExtractor`]. The audio feature extractor is a required input.
    """
    pixel_feature_extractor_class = "TVLTPixelFeatureExtractor"
    audio_feature_extractor_class = "TVLTAudioFeatureExtractor"

    def __init__(self, pixel_feature_extractor, audio_feature_extractor):
        super().__init__(pixel_feature_extractor, audio_feature_extractor)
        self.current_pixel_processor = self.pixel_feature_extractor
        self.current_audio_processor = self.audio_feature_extractor

    def __call__(self, *args, **kwargs):
        """
        Forwards the `pixel_values` argument to TVLTPixelFeatureExtractor's [`~TVLTPixelTokenizer.__call__`] and the `audio_values` argument to TVLTAudioFeatureExtractor's [`~TVLTAudioFeatureExtractor.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        pixel_values = kwargs.pop("pixel_values", None)
        audio_values = kwargs.pop("audio_values", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio_values is None and pixel_values is None:
            raise ValueError("You need to specify either an `audio_values` or `pixel_values` input to process.")

        if pixel_values is not None:
            pixel_inputs = self.pixel_feature_extractor(pixel_inputs, *args, **kwargs)
        if audio_values is not None:
            audio_inputs = self.audio_feature_extractor(audio_values, *args, sampling_rate=sampling_rate, **kwargs)

        if audio_values is None:
            return pixel_inputs
        elif pixel_values is None:
            return audio_inputs
        else:
            return dict(pixel_inputs.items() + audio_inputs.items())

    @property
    def model_input_names(self):
        pixel_feature_extractor_input_names = self.video_feature_extractor.model_input_names
        audio_feature_extractor_input_names = self.audio_feature_extractor.model_input_names
        return list(dict.fromkeys(pixel_feature_extractor_input_names + audio_feature_extractor_input_names))
