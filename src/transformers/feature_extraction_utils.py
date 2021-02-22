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
 Feature extraction classes for python tokenizers.
"""


class BatchFeature(UserDict):
    """"""

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)
        # add similar functionality as BatchEncoding


class PreTrainedFeatureExtractor:
    """
    This is a general feature extraction class for speech recognition
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # IMPORTANT: Feature Extractor are always deterministic -> they are never trained
        # in any way like Tokenizers are -> therefore all configuration params should be
        # stored in a json config
        self.sampling_rate = kwargs.get("sampling_rate", None)
        self.pad_vector = kwargs.get("pad_vector", None)
        self.feature_dim = kwargs.get("feature_dim", None)  # this will be 1 for Wav2Vec2, but 768 for Speech2TextTransformers

    def pad(self, feature: BatchFeature):
        """
        Implement general padding method
        """
        pass

    def from_pretained(self, path):
        """
        General loading method
        """
        pass

    def save_pretrained(self, path):
        """
        General saving method
        """
        pass
