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

# TODO: completely --> tests and the class itself

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


@require_torch
class DiaProcessorTest(unittest.TestCase):
    def setUp(self):
        # TODO: checkpoints + save/load pretrained with audio tokenizer
        self.checkpoint = "AntonV/Dia-1.6B"
        self.audio_tokenizer_checkpoint = "descript/dac_44khz"  # imo should be within the processor as ref
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        return DiaTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_audio_tokenizer(self, **kwargs):
        return DacModel.from_pretrained(self.audio_tokenizer_checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return DiaFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = DiaProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, DiaTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, DiaFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = DiaProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer()
        feature_extractor_add_kwargs = self.get_feature_extractor()

        processor = DiaProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, DiaTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, DiaFeatureExtractor)

    # TODO: not valid anymore since we do more than just feat extract
    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = np.random.rand(1000).astype(np.float32)

        input_feat_extract = feature_extractor(raw_speech, sampling_rate=44100, return_tensors="pt")
        input_processor = processor(raw_speech)

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    # TODO: decode will be the audio tokenizer
    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )

    @require_torch
    @parameterized.expand([(1, 2, [0, 1, 4]), (2, 4, [1, 3, 2]), (4, 8, [0, 5, 7])])
    def test_audio_delay(self, bsz, seq_len, delay_pattern):
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
