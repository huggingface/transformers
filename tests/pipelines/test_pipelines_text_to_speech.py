# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import (
    MODEL_FOR_TEXT_TO_AUDIO_MAPPING,
    TextToAudioPipeline,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    is_torch_available,
    require_torch,
    require_torch_or_tf,
    slow,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch


@is_pipeline_test
@require_torch_or_tf
class TextToAudioPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TEXT_TO_AUDIO_MAPPING

    @require_torch
    def test_small_model_pt(self):
        speech_generator = pipeline(
            task="text-to-audio", model="microsoft/speecht5_tts", framework="pt", vocoder="microsoft/speecht5_hifigan"
        )

        # test if sampling_rate defined

        # Using `do_sample=False` to force deterministic output
        outputs = speech_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            ANY(torch.Tensor),
        )

        outputs = speech_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                ANY(torch.Tensor),
                ANY(torch.Tensor),
            ],
        )

        speaker_embeddings = torch.zeros((1, 512))

        outputs = speech_generator(
            "This is a test", do_sample=True, num_return_sequences=2, speaker_embeddings=speaker_embeddings
        )
        self.assertEqual(
            outputs,
            ANY(torch.Tensor),
        )

    @slow
    @require_torch
    def test_large_model_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt")

        # test if sampling_rate defined, test text-to-speech

        # Using `do_sample=False` to force deterministic output
        outputs = speech_generator("This is a test", do_sample=False, semantic_max_new_tokens=100)
        self.assertEqual(
            outputs,
            ANY(torch.Tensor),
        )

        outputs = speech_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                ANY(torch.Tensor),
                ANY(torch.Tensor),
            ],
        )

        outputs = speech_generator(
            "This is a test",
            do_sample=True,
            semantic_num_return_sequences=2,
            speaker_embeddings="en_speaker_1",
            semantic_max_new_tokens=100,
        )
        self.assertEqual(
            outputs,
            ANY(torch.Tensor),
        )

    def get_test_pipeline(self, model, tokenizer, processor):
        speech_generator = TextToAudioPipeline(model=model, tokenizer=tokenizer)
        return speech_generator, ["This is a test", "Another test"]

    def run_pipeline_test(self, speech_generator, _):
        outputs = speech_generator("This is a test")
        self.assertEqual(
            outputs,
            ANY(torch.Tensor),
        )

        outputs = speech_generator(["This is great !", "Something else"], num_return_sequences=2, do_sample=True)
        self.assertEqual(
            outputs,
            [
                ANY(torch.Tensor),
                ANY(torch.Tensor),
            ],
        )
