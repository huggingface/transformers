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

from transformers import (
    AutoProcessor,
    Qwen2AudioConfig,
    Qwen2AudioEncoderConfig,
    Qwen2AudioForConditionalGeneration,
    Qwen2Config,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester


if is_torch_available():
    import torch


class Qwen2AudioModelTester(ALMModelTester):
    config_class = Qwen2AudioConfig
    conditional_generation_class = Qwen2AudioForConditionalGeneration
    text_config_class = Qwen2Config
    audio_config_class = Qwen2AudioEncoderConfig
    audio_mask_key = "feature_attention_mask"

    def __init__(self, parent, **kwargs):
        # feat_seq_length=60 → after conv2 s=2: 30 → after avg_pool s=2: 15 audio embed tokens.
        kwargs.setdefault("feat_seq_length", 60)
        # Encoder asserts input_features.shape[-1] == max_source_positions * conv1.stride * conv2.stride == 2 * max_source_positions.
        kwargs.setdefault("max_source_positions", kwargs["feat_seq_length"] // 2)
        super().__init__(parent, **kwargs)

    def create_audio_mask(self):
        # Deterministic full-length mask: the base default randomizes via Python's `random`, which isn't
        # re-seeded per test call and desynchronizes the two `prepare_config_and_inputs_for_common`
        # invocations inside generation-comparison tests (e.g. test_greedy_generate_dict_outputs).
        return torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.bool).to(torch_device)

    def get_audio_embeds_mask(self, audio_mask):
        # Mirrors Qwen2AudioEncoder._get_feat_extract_output_lengths: conv2 (k=3,s=2,p=1) then avg_pool (k=2,s=2).
        input_lengths = audio_mask.sum(-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        max_len = int(output_lengths.max().item())
        positions = torch.arange(max_len, device=audio_mask.device)[None, :]
        return (positions < output_lengths[:, None]).long()


@require_torch
class Qwen2AudioForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `Qwen2AudioForConditionalGeneration`.
    """

    model_tester_class = Qwen2AudioModelTester
    pipeline_model_mapping = {"any-to-any": Qwen2AudioForConditionalGeneration} if is_torch_available() else {}

    @unittest.skip(reason="inputs_embeds is the audio-fused path; can't match raw token-only embeddings.")
    def test_inputs_embeds_matches_input_ids(self):
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
