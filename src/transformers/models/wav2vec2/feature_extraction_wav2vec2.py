# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
Feature extractor class for Wav2Vec2
"""

from ...feature_extraction_utils import PreTrainedFeatureExtractor


class Wav2Vec2FeatureExtractor(PreTrainedFeatureExtractor):

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_dim=1,
        sampling_rate=16000,
        padding_value=0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs
    ):
        super().__init__(feature_dim=feature_dim, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def __call__(self, raw_speech):
        pass
