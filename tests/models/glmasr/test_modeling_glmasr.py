# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch glmasr model."""

import unittest

from transformers import (
    AutoProcessor,
    GlmAsrConfig,
    GlmAsrForConditionalGeneration,
    LlamaConfig,
    is_torch_available,
)
from transformers.models.glmasr.configuration_glmasr import GlmAsrEncoderConfig
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester


if is_torch_available():
    import torch


class GlmAsrModelTester(ALMModelTester):
    config_class = GlmAsrConfig
    conditional_generation_class = GlmAsrForConditionalGeneration
    text_config_class = LlamaConfig
    audio_config_class = GlmAsrEncoderConfig
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("head_dim", 8)
        super().__init__(parent, **kwargs)

    def get_audio_embeds_mask(self, audio_mask):
        # conv1 (s=1) preserves length; conv2 (s=2, k=3, p=1) halves; merge_factor=4 post-projector.
        audio_lengths = audio_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1
        max_len = int(post_lengths.max().item())
        positions = torch.arange(max_len, device=audio_mask.device)[None, :]
        return (positions < post_lengths[:, None]).long()


@require_torch
class GlmAsrForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `GlmAsrForConditionalGeneration`.
    """

    model_tester_class = GlmAsrModelTester
    pipeline_model_mapping = {"audio-text-to-text": GlmAsrForConditionalGeneration} if is_torch_available() else {}

    @unittest.skip(
        reason="This test does not apply to GlmAsr since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass


@require_torch
class GlmAsrForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "zai-org/GLM-ASR-Nano-2512"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single_batch_sub_30(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                    },
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            },
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "Yesterday it was thirty five degrees in Barcelona, but today the temperature will go down to minus twenty degrees."
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_single_batch_over_30(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
                    },
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            },
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_batched(self):
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                        },
                        {"type": "text", "text": "Please transcribe this audio into text"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
                        },
                        {"type": "text", "text": "Please transcribe this audio into text"},
                    ],
                },
            ],
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            [
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
            ],
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "Yesterday it was thirty five degrees in Barcelona, but today the temperature will go down to minus twenty degrees.",
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the",
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
