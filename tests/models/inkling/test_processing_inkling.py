# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from huggingface_hub import download_bucket_files
from parameterized import parameterized
from safetensors.torch import load_file

from transformers import AutoProcessor, InklingProcessor, is_torch_available
from transformers.testing_utils import get_tests_dir, require_librosa, require_vision, slow
from transformers.utils import is_vision_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    pass

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class InklingProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InklingProcessor
    audio_input_name = "audio_input_ids"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")
        gemma4_feature_extractor_kwargs = {}
        return feature_extractor_class(**gemma4_feature_extractor_kwargs)

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        gemma4_image_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
        }
        return image_processor_class(**gemma4_image_processor_kwargs)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        extra_special_tokens = {
            "image_token": "<|image|>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "audio_token": "<audio_soft_token>",
            "boa_token": "<start_of_audio>",
            "eoa_token": "<end_of_audio>",
        }
        tokenizer = tokenizer_class.from_pretrained(
            SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<|image|>' }}\n       {%- elif item['type'] == 'video' -%}\n{{ '<video_soft_token>' }}\n      {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n",            "image_seq_length": 3,
        }  # fmt: skip

    # Override as Inkling needs images to be an explicitly nested batch
    def prepare_image_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_image_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=5,
            )

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]

        # Overwritten because Gemma3 needs nested image inputs
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append([np.random.randint(255, size=(h, w, 3), dtype=np.uint8)])

        text = [f"This is an image {getattr(self, 'image_token', '')}"] * len(image_inputs)
        inputs = processor(
            text=text, images=image_inputs, padding=True, return_mm_token_type_ids=True, return_tensors="pt"
        )

        if "mm_token_type_ids" not in inputs:
            self.skipTest("Processor doesn't support `mm_token_type_ids`")

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    def test_get_num_audio_tokens(self):
        """Tests the audio path of the helper used internally in vLLM."""

        processor = self.get_processor()
        if not hasattr(processor, "_compute_audio_num_tokens") or processor.audio_token is None:
            self.skipTest("Processor doesn't support audio token counting")

        # The golden counts are keyed on raw sample counts and assume 16 kHz framing
        # (frame_length=320, hop_length=160 = round(16000 * {20, 10} ms)). Those framing
        # params are derived from the feature extractor's sampling_rate and, because of
        # integer rounding, are not rate-invariant -- so pin a 16 kHz feature extractor
        # here instead of depending on (and asserting) the class default.
        processor.feature_extractor = type(processor.feature_extractor)(sampling_rate=16000)

        # {num_samples (at 16 kHz): expected_audio_tokens}. Some samples diverge from the naive
        # ceil(duration_ms / 40ms) shortcut for each length -- it disagrees with the real
        # arithmetic for most entries except for the 3s/40s ones.
        expected_num_tokens = {
            38560: 60,  # 2.41s
            48000: 75,  # 3.00s
            48800: 76,  # 3.05s
            99360: 155,  # 6.21s
            640000: 750,  # 40s
        }

        audio_lengths = list(expected_num_tokens)
        num_from_helper = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)["num_audio_tokens"]
        self.assertListEqual(num_from_helper, list(expected_num_tokens.values()))

    @unittest.skip("This test seems to be loading a different video, check for all models and fix")
    def test_apply_chat_template_video_frame_sampling(self):
        pass

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("Inkling audio quantization requires PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    @unittest.skip("Inkling packs image patches across the batch instead of keeping one tensor per image")
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        pass

    @unittest.skip("Inkling quantizes input features into discrete audio input IDs")
    def test_feature_extractor_defaults(self):
        pass

    @unittest.skip("The test fixture passes image_seq_length, which is not an InklingProcessor attribute")
    def test_processor_to_json_string(self):
        pass


@slow
class InklingProcessingIntegrationTest(unittest.TestCase):
    """
    Check against sglang reference..

    reproducers (one per modality, regenerate from sglang and upload the golden to
    ``hf://buckets/hf-internal-testing/tml-integration-tests/<case>/expected_processing.safetensors``):
        ~/tml/reproducers/reproducer_processing_{text,image,audio,image_audio,multi_image,multi_audio}.py
    gist: https://gist.github.com/eustlb/cb2a5df1676911fa0eb07d0a76a38ae7
    """

    # sglang sentinels
    IMAGE_SENTINEL = -101
    AUDIO_SENTINEL = -102

    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    IMAGE_URL_2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
    AUDIO_URL = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_medium.wav"
    )
    AUDIO_URL_2 = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_short.wav"
    )

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "hf-internal-testing/tiny-inkling"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)
        cls.bucket = "hf-internal-testing/tml-integration-tests"

    def _load_expected(self, case: str) -> dict:
        remote = f"{case}/expected_processing.safetensors"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "expected_processing.safetensors")
            download_bucket_files(self.bucket, files=[(remote, local)])
            return load_file(local)

    def _remap_sentinels(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        input_ids = input_ids.clone()
        input_ids[input_ids == self.IMAGE_SENTINEL] = self.processor.image_token_id
        input_ids[input_ids == self.AUDIO_SENTINEL] = self.processor.audio_token_id
        return input_ids

    def _expected_dmel_from_inputs(self, inputs) -> "torch.Tensor":
        # Trim each padded audio's dmel by its mask and concatenate in order
        audio_input_ids = inputs["audio_input_ids"]
        mask = inputs.get("audio_input_ids_mask")
        per_audio = [
            audio_input_ids[i][mask[i].bool()] if mask is not None else audio_input_ids[i]
            for i in range(audio_input_ids.shape[0])
        ]
        return torch.cat(per_audio, dim=0)

    def _assert_matches_sglang(self, case: str, messages: list, has_audio: bool = False):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        expected = self._load_expected(case)

        input_ids = inputs["input_ids"][0]
        expected_input_ids = self._remap_sentinels(expected["input_ids"].to(torch.int64))
        torch.testing.assert_close(input_ids, expected_input_ids, rtol=0, atol=0)

        if has_audio:
            dmel = self._expected_dmel_from_inputs(inputs)
            torch.testing.assert_close(dmel, expected["audio_dmel"].to(torch.int32), rtol=0, atol=0)

    def test_apply_chat_template_text(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
        self._assert_matches_sglang("text", messages)

    def test_apply_chat_template_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {"type": "image", "url": self.IMAGE_URL},
                ],
            }
        ]
        self._assert_matches_sglang("image", messages)

    def test_apply_chat_template_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in this clip?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        self._assert_matches_sglang("audio", messages, has_audio=True)

    def test_apply_chat_template_image_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image and tell me what is said in the clip."},
                    {"type": "image", "url": self.IMAGE_URL},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        self._assert_matches_sglang("image_audio", messages, has_audio=True)

    def test_apply_chat_template_multi_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images."},
                    {"type": "image", "url": self.IMAGE_URL},
                    {"type": "image", "url": self.IMAGE_URL_2},
                ],
            }
        ]
        self._assert_matches_sglang("multi_image", messages)

    def test_apply_chat_template_multi_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in these two clips?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                    {"type": "audio", "url": self.AUDIO_URL_2},
                ],
            }
        ]
        self._assert_matches_sglang("multi_audio", messages, has_audio=True)

    def test_apply_chat_template_audio_without_attention_mask(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in this clip?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        common = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        with_mask = self.processor.apply_chat_template(messages, **common)
        # TODO: @eustlb, return_attention_mask is not best API and should be changed
        # with audio processors (#44394)
        without_mask = self.processor.apply_chat_template(
            messages, audio_kwargs={"return_attention_mask": False}, **common
        )

        self.assertIsNotNone(with_mask.get("audio_input_ids_mask"))
        self.assertIsNone(without_mask.get("audio_input_ids_mask"))

        audio_id = self.processor.audio_token_id
        num_frames = with_mask["audio_input_ids"].shape[-2]
        n_placeholders_with = int((with_mask["input_ids"] == audio_id).sum())
        n_placeholders_without = int((without_mask["input_ids"] == audio_id).sum())

        # One audio soft token per frame, mask on or off
        self.assertEqual(n_placeholders_with, num_frames)
        self.assertEqual(n_placeholders_without, num_frames)
