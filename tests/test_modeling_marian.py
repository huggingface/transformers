# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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


import os
import tempfile
import unittest
from pathlib import Path

from transformers import is_torch_available
from transformers.marian2hf import main

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import BertConfig, BertModel, MarianModel, MarianSPTokenizer, BartConfig


LOCAL_PATH = "/Users/shleifer/transformers_fork/converted-en-de/"
LOCAL_MARIAN = "/Users/shleifer/transformers_fork/en-de/"



class IntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dest_dir = Path('utest-converted-en-de')
        dest_dir.mkdir(exist_ok=True)
        main(Path(LOCAL_MARIAN), dest_dir)
        cls.tokenizer = MarianSPTokenizer.from_pretrained(dest_dir.name)
        cls.model = MarianModel.from_pretrained(dest_dir.name)
        cls.dest_dir = dest_dir
        cls.eos_token_id = cls.model.config.eos_token_id
        return cls

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls.dest_dir)

    def test_forward(self):
        src, tgt = ["What's for dinner?", "life"], ["Was gibt es zum Abendessen", "Leben"]
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src, tgt_texts=tgt)
        shapes = {k: v.shape for k, v in model_inputs.items()}
        desired_keys = {
            "input_ids",
            "attention_mask",
            #"token_type_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
            #"decoder_token_type_ids",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        outputs = self.model(**model_inputs)

    def test_generate(self):
        """Should produce a good translation."""
        src, tgt = ["What's for dinner?", "life"], ["Was gibt es zum Abendessen", "Leben"]
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src)
        result_ids = self.model.generate(**model_inputs, num_beams=6, decoder_start_token_id=self.eos_token_id, no_repeat_ngram_size=3)
        print(result_ids)
        predicted_de_text = [self.tokenizer.decode(r) for r in result_ids]
        self.assertListEqual(predicted_de_text, tgt)
