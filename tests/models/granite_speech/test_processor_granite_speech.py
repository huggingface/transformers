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
import pytest
import torch
from parameterized import parameterized

from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torchaudio,
    torch_device,
)
from transformers.utils import is_torchaudio_available


if is_torchaudio_available():
    from transformers import GraniteSpeechFeatureExtractor, GraniteSpeechProcessor


@require_torch
@require_torchaudio
class GraniteSpeechProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.checkpoint = "ibm-granite/granite-speech-3.3-8b"
        processor = GraniteSpeechProcessor.from_pretrained(self.checkpoint)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_audio_processor(self, **kwargs):
        return GraniteSpeechFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        """Ensure we can save / reload a processor correctly."""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        processor.save_pretrained(self.tmpdirname)
        processor = GraniteSpeechProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, GPT2TokenizerFast)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor.to_json_string())
        self.assertIsInstance(processor.audio_processor, GraniteSpeechFeatureExtractor)

    def test_requires_text(self):
        """Ensure we require text"""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None)

    def test_bad_text_fails(self):
        """Ensure we gracefully fail if text is the wrong type."""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()

        processor = GraniteSpeechProcessor(tokenizer=tokenizer, audio_processor=audio_processor)
        with pytest.raises(TypeError):
            processor(text=424, audio=None)

    def test_bad_nested_text_fails(self):
        """Ensure we gracefully fail if text is the wrong nested type."""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=[424], audio=None)

    def test_bad_audio_fails(self):
        """Ensure we gracefully fail if audio is the wrong type."""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None, audio="foo")

    def test_nested_bad_audio_fails(self):
        """Ensure we gracefully fail if audio is the wrong nested type."""
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None, audio=["foo"])

    @parameterized.expand(
        [
            ([1, 269920], [171], torch.rand),
            ([1, 269920], [171], np.random.rand),
        ]
    )
    def test_audio_token_filling_same_len_feature_tensors(self, vec_dims, num_expected_features, random_func):
        """Ensure audio token filling is handled correctly when we have
        one or more audio inputs whose features are all the same length
        stacked into a tensor / numpy array.

        NOTE: Currently we enforce that each sample can only have one audio.
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )
        audio = random_func(*vec_dims) - 0.5

        audio_tokens = processor.audio_token * vec_dims[0]
        inputs = processor(text=f"{audio_tokens} Can you compare this audio?", audio=audio, return_tensors="pt")

        # Check the number of audio tokens
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        # Make sure the number of audio tokens matches the number of features
        num_computed_features = processor.audio_processor._get_num_audio_features(
            [vec_dims[1] for _ in range(vec_dims[0])],
        )
        num_audio_tokens = int(torch.sum(inputs["input_ids"] == audio_token_id))
        assert list(inputs["input_features"].shape) == [vec_dims[0], 844, 160]
        assert sum(num_computed_features) == num_audio_tokens

    def test_audio_token_filling_varying_len_feature_list(self):
        """Ensure audio token filling is handled correctly when we have
        multiple varying len audio sequences passed as a list.
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )
        vec_dims = [[1, 142100], [1, 269920]]
        num_expected_features = [90, 171]
        audio = [torch.rand(dims) - 0.5 for dims in vec_dims]

        inputs = processor(
            text=[
                f"{processor.audio_token} Can you describe this audio?",
                f"{processor.audio_token} How does it compare with this audio?",
            ],
            audio=audio,
            return_tensors="pt",
        )

        # Check the number of audio tokens
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        # Make sure the number of audio tokens matches the number of features
        num_calculated_features = processor.audio_processor._get_num_audio_features(
            [dims[1] for dims in vec_dims],
        )
        num_audio_tokens = int(torch.sum(inputs["input_ids"] == audio_token_id))
        assert num_calculated_features == [90, 171]
        assert sum(num_expected_features) == num_audio_tokens

    @require_torch_accelerator
    def test_device_override(self):
        """Ensure that we regardless of the processing device, the tensors
        produced are on the CPU.
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        vec_dims = [1, 269920]
        wav = torch.rand(vec_dims) - 0.5

        inputs = processor(
            text=f"{processor.audio_token} Can you transcribe this audio?",
            audio=wav,
            return_tensors="pt",
            device=torch_device,
        )

        assert inputs["input_features"].device.type == "cpu"
