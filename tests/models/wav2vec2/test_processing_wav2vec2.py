# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import json
import os
import unittest

from transformers.models.wav2vec2 import Wav2Vec2Processor
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES

from ...test_processing_common import ProcessorTesterMixin
from ..wav2vec2.test_feature_extraction_wav2vec2 import floats_list


class Wav2Vec2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Wav2Vec2Processor
    audio_input_name = "input_values"
    text_input_name = "labels"

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")

        feature_extractor_map = {
            "feature_size": 1,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "return_attention_mask": False,
            "do_normalize": True,
        }
        return feature_extractor_class(**feature_extractor_map)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab = "<pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        add_kwargs_tokens_map = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        return tokenizer_class.from_pretrained(cls.tmpdirname, **add_kwargs_tokens_map)

    # todo: check why this test is failing
    @unittest.skip("Failing for unknown reason")
    def test_overlapping_text_audio_kwargs_handling(self):
        pass

    @unittest.skip("Wav2Vec2BertProcessor changes input_features")
    def test_processor_with_multiple_inputs(self):
        pass

    def test_feature_extractor(self):
        feature_extractor = self.get_component("feature_extractor")
        processor = self.get_processor()
        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_model_input_names(self):
        processor = self.get_processor()

        text = "lower newer"
        audio_inputs = self.prepare_audio_inputs()

        inputs = processor(text=text, audio=audio_inputs, return_attention_mask=True, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))
