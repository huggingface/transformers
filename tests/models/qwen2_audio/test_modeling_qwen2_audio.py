# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen2Audio model."""

import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa
import pytest

from transformers import (
    AutoProcessor,
    Qwen2AudioConfig,
    Qwen2AudioForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...audio_tester import AudioModelTest, AudioModelTester


if is_torch_available():
    import torch


class Qwen2AudioModelTester(AudioModelTester):
    config_class = Qwen2AudioConfig
    conditional_generation_class = Qwen2AudioForConditionalGeneration

    def get_audio_mask_key(self):
        return "feature_attention_mask"


@require_torch
class Qwen2AudioForConditionalGenerationModelTest(AudioModelTest, unittest.TestCase):
    """
    Model tester for `Qwen2AudioForConditionalGeneration`.
    """

    model_tester_class = Qwen2AudioModelTester
    pipeline_model_mapping = {"any-to-any": Qwen2AudioForConditionalGeneration} if is_torch_available() else {}

    @unittest.skip(reason="Compile not yet supported because in Qwen2Audio models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen2Audio models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class Qwen2AudioForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map=torch_device, dtype=torch.float16
        )

        url = "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": url},
                    {"type": "text", "text": "What's that sound?"},
                ],
            }
        ]

        raw_audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=formatted_prompt, audio=[raw_audio], return_tensors="pt", padding=True).to(
            torch_device
        )

        torch.manual_seed(42)
        output = model.generate(**inputs, max_new_tokens=32)

        # fmt: off
        EXPECTED_INPUT_IDS = torch.tensor(
            [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14755, 220, 16, 25, 220, 151647, *[151646] * 101 , 151648, 198, 3838, 594, 429, 5112, 30, 151645, 198, 151644, 77091, 198]],
            device=torch_device
        )
        # fmt: on
        torch.testing.assert_close(inputs["input_ids"], EXPECTED_INPUT_IDS)

        # fmt: off
        EXPECTED_DECODED_TEXT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|>" + "<|AUDIO|>" * 101 + "<|audio_eos|>\nWhat's that sound?<|im_end|>\n<|im_start|>assistant\nIt is the sound of glass breaking.<|im_end|>"
        # fmt: on
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=False),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map=torch_device, dtype=torch.float16
        )

        conversation1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "What can you hear?"},
                ],
            },
        ]

        conversation2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                    },
                    {"type": "text", "text": "What does the person say?"},
                ],
            },
        ]

        conversations = [conversation1, conversation2]

        text = [
            self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            for conversation in conversations
        ]

        audios = []
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    BytesIO(urlopen(ele["audio_url"]).read()),
                                    sr=self.processor.feature_extractor.sampling_rate,
                                )[0]
                            )

        inputs = self.processor(text=text, audio=audios, return_tensors="pt", padding=True).to(torch_device)

        torch.manual_seed(42)
        output = model.generate(**inputs, max_new_tokens=32)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat's that sound?\nassistant\nIt is the sound of glass shattering.\nuser\nAudio 2: \nWhat can you hear?\nassistant\ncough and throat clearing.",
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat does the person say?\nassistant\nThe original content of this audio is: 'Mister Quiller is the apostle of the middle classes and we are glad to welcome his gospel.'",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_multiurn(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map=torch_device, dtype=torch.float16
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "How about this one?"},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                BytesIO(urlopen(ele["audio_url"]).read()),
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )

        inputs = self.processor(text=formatted_prompt, audio=audios, return_tensors="pt", padding=True).to(
            torch_device
        )

        torch.manual_seed(42)
        output = model.generate(**inputs, max_new_tokens=32, top_k=1)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat's that sound?\nassistant\nIt is the sound of glass shattering.\nuser\nAudio 2: \nHow about this one?\nassistant\nThroat clearing."
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
