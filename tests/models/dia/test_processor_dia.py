# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from transformers import DacModel, DiaFeatureExtractor, DiaProcessor, DiaTokenizer
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available:
    import torch


# Copied from tests.utils.test_modeling_utils.check_models_equal
def check_models_equal(model1, model2):
    models_are_equal = True
    for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
        if model1_p.data.ne(model2_p.data).sum() > 0:
            models_are_equal = False

    return models_are_equal


@require_torch
class DiaProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "AntonV/Dia-1.6B"
        self.audio_tokenizer_checkpoint = "descript/dac_44khz"
        self.tmpdirname = tempfile.mkdtemp()

        # Audio tokenizer is a bigger model so we will reuse this if possible
        self.processor = DiaProcessor(
            tokenizer=self.get_tokenizer(),
            feature_extractor=self.get_feature_extractor(),
            audio_tokenizer=self.get_audio_tokenizer(),
        )

        # Default audio values based on Dia and Dac
        self.pad_id = 1025
        self.bos_id = 1026
        self.dac_chunk_len = 512
        self.delay_pattern = [0, 8, 9, 10, 11, 12, 13, 14, 15]

    def get_tokenizer(self, **kwargs):
        return DiaTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return DiaFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def get_audio_tokenizer(self, **kwargs):
        return DacModel.from_pretrained(self.audio_tokenizer_checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
        del self.processor

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        audio_tokenizer = self.get_audio_tokenizer()

        processor = DiaProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, audio_tokenizer=audio_tokenizer
        )

        processor.save_pretrained(self.tmpdirname)
        processor = DiaProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, DiaTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, DiaFeatureExtractor)

        self.assertEqual(processor.audio_tokenizer.__class__.__name__, audio_tokenizer.__class__.__name__)
        self.assertEqual(processor.audio_tokenizer.name_or_path, audio_tokenizer.name_or_path)
        self.assertTrue(check_models_equal(processor.audio_tokenizer, audio_tokenizer))
        self.assertIsInstance(processor.audio_tokenizer, DacModel)

    def test_save_load_pretrained_additional_features(self):
        processor = DiaProcessor(
            tokenizer=self.get_tokenizer(),
            feature_extractor=self.get_feature_extractor(),
            audio_tokenizer=self.get_audio_tokenizer(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer()
        feature_extractor_add_kwargs = self.get_feature_extractor()
        audio_tokenizer_add_kwargs = self.get_audio_tokenizer()

        processor = DiaProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, DiaTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, DiaFeatureExtractor)

        self.assertEqual(processor.audio_tokenizer.__class__.__name__, audio_tokenizer_add_kwargs.__class__.__name__)
        self.assertEqual(processor.audio_tokenizer.name_or_path, audio_tokenizer_add_kwargs.name_or_path)
        self.assertTrue(check_models_equal(processor.audio_tokenizer, audio_tokenizer_add_kwargs))
        self.assertIsInstance(processor.audio_tokenizer, DacModel)

    def test_model_input_names(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(
            self.processor.model_input_names,
            list(dict.fromkeys(tokenizer.model_input_names + ["decoder_input_ids", "decoder_attention_mask"])),
            msg="`processor` model input names do not match the expected names.",
        )

    def test_tokenize(self):
        tokenizer = self.get_tokenizer()
        random_text = ["This is a processing test for tokenization", "[S1] Dia template style [S2] Nice"]

        input_tokenizer = tokenizer(random_text, padding=True, return_tensors="pt")
        input_processor = self.processor(random_text)

        for key in input_tokenizer:
            self.assertTrue((input_tokenizer[key] == input_processor[key]).all())

    def test_no_audio(self):
        random_text = ["Dummy Input"] * 2
        input_processor = self.processor(random_text)
        audio_tokens, audio_mask = input_processor["decoder_input_ids"], input_processor["decoder_attention_mask"]

        # full mask with +1 for bos
        self.assertTrue(audio_mask.sum() == (max(self.delay_pattern) + 1) * len(random_text))
        self.assertTrue(
            audio_tokens.shape
            == (
                len(random_text),
                max(self.delay_pattern) + 1,
                len(self.delay_pattern),
            )
        )

        for channel_idx, delay in enumerate(self.delay_pattern):
            expected_sequence = torch.ones(size=(audio_tokens.shape[:-1])) * self.pad_id
            expected_sequence[:, : delay + 1] = self.bos_id
            self.assertTrue((audio_tokens[..., channel_idx] == expected_sequence).all())

    def test_audio(self):
        audio_tokenizer = self.get_audio_tokenizer()
        feature_extractor = self.get_feature_extractor()

        random_text = ["Dummy Input"] * 2
        # Dac only starts accepting audio from a certain length (ensured via >=1024)
        raw_speeches = [np.random.rand(2048).astype(np.float32), np.random.rand(1024).astype(np.float32)]
        input_processor = self.processor(random_text, raw_speeches)
        audio_tokens, audio_mask = input_processor["decoder_input_ids"], input_processor["decoder_attention_mask"]

        sequence_len = audio_mask.shape[1]
        for batch_idx, speech in enumerate(raw_speeches):
            raw_audio = feature_extractor(speech, return_tensors="pt")["input_values"]
            codebooks = audio_tokenizer(raw_audio).audio_codes.transpose(1, 2)

            pad_len = sequence_len - audio_mask.sum(dim=-1)[batch_idx]
            for channel_idx, delay in enumerate(self.delay_pattern):
                # Left padding filled bos, right padding (delay) are pad
                start_idx = pad_len + delay + 1
                end_idx = start_idx + codebooks.shape[1]

                encoded_sequence = audio_tokens[batch_idx, :, channel_idx]
                expected_sequence = torch.ones(size=(sequence_len,)) * self.pad_id
                expected_sequence[:start_idx] = self.bos_id
                expected_sequence[start_idx:end_idx] = codebooks[0, :, channel_idx]

                self.assertTrue((encoded_sequence == expected_sequence).all())

        # Just to make sure the masking correctly only ignores bos tokens
        self.assertTrue((audio_tokens[~audio_mask.bool()] == self.bos_id).all())

    @parameterized.expand([([1, 1],), ([1, 5],), ([2, 4, 6],)])
    def test_decode_audio(self, audio_lens):
        feature_extractor = self.get_feature_extractor()
        audio_tokenizer = self.get_audio_tokenizer()

        random_text = ["Dummy Input"] * len(audio_lens)
        raw_speeches = [np.random.rand(self.dac_chunk_len * l).astype(np.float32) for l in audio_lens]
        # we need eos (given if training) to decode properly, also enforced via custom logits processor
        input_processor = self.processor(random_text, raw_speeches, generation=False)
        audio_tokens = input_processor["decoder_input_ids"]

        decoded_speeches = self.processor.batch_decode(audio_tokens)
        for batch_idx, speech in enumerate(raw_speeches):
            raw_audio = feature_extractor(speech, return_tensors="pt")["input_values"]
            codebooks = audio_tokenizer(raw_audio).audio_codes

            decoded_audio = decoded_speeches[batch_idx]
            expected_audio = audio_tokenizer.decode(audio_codes=codebooks).audio_values

            self.assertTrue((expected_audio == decoded_audio).all())
            self.assertTrue(decoded_speeches[batch_idx].shape[-1] == audio_lens[batch_idx] * self.dac_chunk_len)

    @parameterized.expand([(1, 2, [0, 1, 4]), (2, 4, [1, 3, 2]), (4, 8, [0, 5, 7])])
    def test_delay_in_audio(self, bsz, seq_len, delay_pattern):
        # static functions which are crucial, hence we also test them here
        build_indices_fn = DiaProcessor.build_indices
        delay_fn = DiaProcessor.apply_audio_delay

        bos, pad = -2, -1
        num_channels = len(delay_pattern)

        audio_input = torch.arange(bsz * seq_len * num_channels).view(bsz, seq_len, num_channels)
        # imitate a delay mask with zeroes
        audio_input = torch.cat([audio_input, torch.zeros(size=(bsz, max(delay_pattern), num_channels))], dim=1)

        precomputed_idx = build_indices_fn(
            bsz=bsz,
            seq_len=seq_len + max(delay_pattern),
            num_channels=num_channels,
            delay_pattern=delay_pattern,
            revert=False,
        )
        delayed_audio_out = delay_fn(
            audio=audio_input,
            pad_token_id=pad,
            bos_token_id=bos,
            precomputed_idx=precomputed_idx,
        )

        # every channel idx is shifted by delay_pattern[idx]
        delayed_audio_res = audio_input.clone()
        for idx, delay in enumerate(delay_pattern):
            delayed_audio_res[:, :delay, idx] = bos
            remaining_input = seq_len + max(delay_pattern) - delay
            delayed_audio_res[:, delay:, idx] = audio_input[:, :remaining_input, idx]

        self.assertTrue((delayed_audio_out == delayed_audio_res).all())

        # we should get back to the original audio we had (when removing the delay pad)
        bsz, new_seq_len, num_channels = delayed_audio_out.shape
        precomputed_idx = build_indices_fn(
            bsz=bsz,
            seq_len=new_seq_len,
            num_channels=num_channels,
            delay_pattern=delay_pattern,
            revert=True,
        )
        reverted_audio_out = delay_fn(
            audio=delayed_audio_out,
            pad_token_id=pad,
            bos_token_id=bos,
            precomputed_idx=precomputed_idx,
        )

        reverted_audio_res = audio_input.clone()[:, :seq_len]

        self.assertTrue((reverted_audio_out[:, :seq_len] == reverted_audio_res).all())
