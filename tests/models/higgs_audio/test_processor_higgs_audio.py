# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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

from transformers import DacFeatureExtractor, HiggsAudioProcessor, HiggsAudioTokenizer, PreTrainedTokenizerFast
from transformers.models.higgs_audio.processing_higgs_audio import HiggsAudioProcessorKwargs, build_delay_pattern_mask
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
class HiggsAudioProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "szhengac25/higgs-audio-v2-generation-3B-base"
        self.tmpdirname = tempfile.mkdtemp()
        self.chunk_len = 24000

        self.processor = HiggsAudioProcessor.from_pretrained(self.checkpoint, torch_dtype="auto")

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
        del self.processor

    def test_save_load_pretrained_default(self):
        self.processor.save_pretrained(self.tmpdirname)
        processor = HiggsAudioProcessor.from_pretrained(self.tmpdirname, torch_dtype="auto")

        self.assertEqual(processor.tokenizer.get_vocab(), self.processor.tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(
            processor.feature_extractor.to_json_string(), self.processor.feature_extractor.to_json_string()
        )
        self.assertIsInstance(processor.feature_extractor, DacFeatureExtractor)

        self.assertEqual(
            processor.audio_tokenizer.__class__.__name__, self.processor.audio_tokenizer.__class__.__name__
        )
        self.assertEqual(processor.audio_tokenizer.name_or_path, self.processor.audio_tokenizer.name_or_path)
        self.assertTrue(check_models_equal(processor.audio_tokenizer, self.processor.audio_tokenizer))
        self.assertIsInstance(processor.audio_tokenizer, HiggsAudioTokenizer)

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is a test sentence 0."},
                    {"type": "audio"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a test sentence 1."},
                    {"type": "audio"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This is a prompt."},
                ],
            },
        ]
        processor = HiggsAudioProcessor.from_pretrained(self.checkpoint)
        rendered = processor.apply_chat_template(messages, tokenize=False)

        expected_rendered = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "This is a test sentence 0.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "This is a test sentence 1.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "This is a prompt.<|eot_id|>"
        )
        self.assertEqual(rendered, expected_rendered)

        # now let's very that it expands audio tokens correctly
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "This is a test sentence."}, {"type": "audio"}],
            },
        ]

        input_ids = processor.apply_chat_template(messages, tokenize=True)

        expected_ids = torch.tensor(
            [[128000, 128006, 9125, 128007, 271, 2028, 374, 264, 1296, 11914, 13, 128011, 128015, 128012, 128009]]
        )
        torch.testing.assert_close(input_ids, expected_ids)

    @parameterized.expand([([1, 2],), ([2, 4, 6],)])
    def test_decode_audio(self, audio_lens):
        raw_speeches = [np.random.rand(self.chunk_len * l).astype(np.float32) for l in audio_lens]
        decoder_text_ids = torch.randint(0, 1024, (10,))
        prompt_token_length = 5
        kwargs = HiggsAudioProcessorKwargs._defaults

        for batch_idx, speech in enumerate(raw_speeches):
            input_values = self.processor.feature_extractor(
                speech,
            )
            input_values = torch.tensor(
                input_values["input_values"][0], device=self.processor.audio_tokenizer.device
            ).unsqueeze(0)
            audio_ids = self.processor.audio_tokenizer.encode(input_values)
            audio_codes = build_delay_pattern_mask(
                audio_ids.audio_codes,
                bos_token_id=kwargs["audio_kwargs"]["audio_stream_bos_id"],
                pad_token_id=kwargs["audio_kwargs"]["audio_stream_eos_id"],
            )[0].squeeze(0)

            response = self.processor.decode(
                decoder_text_ids=decoder_text_ids,
                decoder_audio_ids=[audio_codes],
                prompt_token_length=prompt_token_length,
            )

            # hop_length samples -> 1 token and Higgs Audio assume the audio codes start with audio_stream_bos and end with audio_stream_eos
            self.assertTrue(len(response.audio) == len(speech) - self.processor.audio_tokenizer.config.hop_length * 2)
