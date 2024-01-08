# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import pytest

from transformers import AutoTokenizer, BarkProcessor
from transformers.testing_utils import require_torch, slow


@require_torch
class BarkProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "suno/bark-small"
        self.tmpdirname = tempfile.mkdtemp()
        self.voice_preset = "en_speaker_1"
        self.voice_presets = ["en_speaker_0", "fr_speaker_0", "en_speaker_1"] * 7
        self.input_string = "This is a test string"
        self.speaker_embeddings_dict_path = "speaker_embeddings_path.json"
        self.speaker_embeddings_directory = "speaker_embeddings"

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

    @slow
    def test_save_load_pretrained_additional_features(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )
        processor.save_pretrained(
            self.tmpdirname,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
            speaker_embeddings_directory=self.speaker_embeddings_directory,
        )

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")

        processor = BarkProcessor.from_pretrained(
            self.tmpdirname,
            self.speaker_embeddings_dict_path,
            bos_token="(BOS)",
            eos_token="(EOS)",
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

    def test_speaker_embeddings(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        seq_len = 35
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8

        voice_presets = {
            "semantic_prompt": np.ones(seq_len),
            "coarse_prompt": np.ones((nb_codebooks_coarse, seq_len)),
            "fine_prompt": np.ones((nb_codebooks_total, seq_len)),
        }

        # test providing already loaded voice_preset
        inputs = processor(text=self.input_string, voice_preset=voice_presets)

        processed_voice_preset = inputs["history_prompt"]
        for key in voice_presets:
            self.assertListEqual(
                voice_presets[key].tolist(), processed_voice_preset.get(key, np.array([]))[0].tolist()
            )

        # test loading voice preset from npz file
        tmpfilename = os.path.join(self.tmpdirname, "file.npz")
        np.savez(tmpfilename, **voice_presets)
        inputs = processor(text=self.input_string, voice_preset=tmpfilename)
        processed_voice_preset = inputs["history_prompt"]

        for key in voice_presets:
            self.assertListEqual(
                voice_presets[key].tolist(), processed_voice_preset.get(key, np.array([]))[0].tolist()
            )

        # test loading voice preset from the hub
        inputs = processor(text=self.input_string, voice_preset=self.voice_preset)

    def test_multiple_speaker_embeddings(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        seq_lens = [35, 45]
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8
        num_text = 21

        voice_presets = [
            {
                "semantic_prompt": np.ones(seq_lens[0]),
                "coarse_prompt": np.ones((nb_codebooks_coarse, seq_lens[0])),
                "fine_prompt": np.ones((nb_codebooks_total, seq_lens[0])),
            }
        ] * 11
        voice_presets += [
            {
                "semantic_prompt": np.ones(seq_lens[1]),
                "coarse_prompt": np.ones((nb_codebooks_coarse, seq_lens[1])),
                "fine_prompt": np.ones((nb_codebooks_total, seq_lens[1])),
            }
        ] * 10

        # test providing already loaded voice_preset
        inputs = processor(text=self.input_string, voice_preset=voice_presets)

        processed_voice_presets = inputs["history_prompt"]
        for i, voice_preset in enumerate(voice_presets):
            for key in voice_preset:
                self.assertListEqual(
                    voice_preset[key].tolist(), processed_voice_presets.get(key, np.array([]))[i].tolist()
                )

        # test loading voice preset from npz file
        tmpfilenames = []
        for i in range(num_text):
            tmpfilename = os.path.join(self.tmpdirname, f"file_{i}.npz")
            np.savez(tmpfilename, **voice_presets[i])
            tmpfilenames.append(tmpfilename)

        inputs = processor(text=self.input_string, voice_preset=tmpfilenames)
        processed_voice_presets = inputs["history_prompt"]

        for i, voice_preset in enumerate(voice_presets):
            for key in voice_preset:
                self.assertListEqual(
                    voice_preset[key].tolist(), processed_voice_presets.get(key, np.array([]))[i].tolist()
                )
        # test loading voice preset from the hub
        inputs = processor(text=self.input_string, voice_preset=self.voice_presets)

        # if the number of voices is different than 1 or batch size, an error should be raised
        with pytest.raises(ValueError) as exception:
            inputs = processor(text=self.input_string, voice_preset=self.voice_presets[:2])
        assert (
            exception.value.args[0] == "The number of voice presets 2 does not match batch size of size 21. "
            "The number of voice presets should either be 1 or 21."
        )

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        encoded_processor = processor(text=self.input_string)

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
