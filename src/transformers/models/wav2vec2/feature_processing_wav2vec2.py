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
Speech processor class for Wav2Vec2
"""

# NOTE inheritance from feature extractor
class Wav2Vec2FeatureExtractor(PreTrainedFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, raw_speech):
        """
        Implement the call method
        """
        pass


# NOTE inheritance from tokenizer
class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, text):
        """
        Implement encoding functionality
        """
        pass

    def _decode(self, text):
        """
        Implement decoding functionality
        """
        pass


class Wav2Vec2SpeechProcessor:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise ValueError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        feature_extractor_path = os.path.join(save_directory, "feature_extractor")
        tokenizer_path = os.path.join(save_directory, "tokenizer")
        self.feature_extractor.save_pretrained(feature_extractor_path)
        self.tokenizer.save_pretrained(tokenizer_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "feature_extractor")
        )
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, "tokenizer"))

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
