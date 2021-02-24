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

# class Wav2Vec2Processor:
#    def __init__(self, feature_extractor, tokenizer):
#        self.feature_extractor = feature_extractor
#        self.tokenizer = tokenizer
#        self.current_processor = self.feature_extractor
#
#    def save_pretrained(self, pretrained_model_name_or_path):
#        self.feature_extractor.save_pretrained(pretrained_model_name_or_path)
#        self.tokenizer.save_pretrained(pretrained_model_name_or_path)
#
#    @classmethod
#    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
# will look for a `feature_extractor_config.json` file
#        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path)
# will look for the tokenizer files
#        tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_model_name_or_path)
#
#        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
#
#    def __call__(self, *args, **kwargs):
#        return self.current_processor(*args, **kwargs)
#
#    def batch_decode(self, *args, **kwargs):
#        return self.tokenizer.batch_decode(*args, **kwargs)
#
#    def decode(self, *args, **kwargs):
#        return self.tokenizer.decode(*args, **kwargs)
#
#    @contextmanager
#    def as_target_tokenizer(self):
#        """
# Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to # sequence-to-sequence
# models that need a slightly different processing for the labels. #
# """
#        self.current_processor = self.tokenizer
#        yield
#        self.current_processor = self.feature_extractor
