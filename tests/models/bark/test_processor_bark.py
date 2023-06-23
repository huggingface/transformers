# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

from transformers import AutoTokenizer, BarkProcessor
from transformers.testing_utils import require_torch


@require_torch
class BarkProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "bert-base-multilingual-cased"
        self.tmpdirname = tempfile.mkdtemp()
        self.repo_id = "ylacombe/bark-small"
        self.subfolder = "speaker_embeddings"
        self.voice_preset = "en_speaker_1"
        self.input_string = "This is a test string"

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor = BarkProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

    def test_save_load_pretrained_additional_features(self):
        processor = BarkProcessor(tokenizer=self.get_tokenizer(), repo_id=self.repo_id, subfolder=self.subfolder)
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")

        processor = BarkProcessor.from_pretrained(
            self.tmpdirname,
            bos_token="(BOS)",
            eos_token="(EOS)",
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

    def test_speaker_embeddings(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer, repo_id=self.tmpdirname)

        seq_len = 35
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8

        voice_preset = {
            "semantic_prompt": np.ones(seq_len),
            "coarse_prompt": np.ones((nb_codebooks_coarse, seq_len)),
            "fine_prompt": np.ones((nb_codebooks_total, seq_len)),
        }

        # test providing already loaded voice_preset
        _, processed_voice_preset = processor(text=self.input_string, voice_preset=voice_preset)
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        tmpfilename = os.path.join(self.tmpdirname, "file.npz")
        np.savez(tmpfilename, **voice_preset)
        _, processed_voice_preset = processor(text=self.input_string, voice_preset="file.npz")
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        # test loading voice preset from npz file

        processor = BarkProcessor(tokenizer=tokenizer, repo_id=self.repo_id, subfolder=self.subfolder)

        # test loading voice preset from the hub
        processor(text=self.input_string, voice_preset=self.voice_preset)

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        encoded_processor, _ = processor(text=self.input_string)

        encoded_tok = tokenizer(
            self.input_string,
            padding="max_length",
            max_length=256,
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key].squeeze().tolist())
