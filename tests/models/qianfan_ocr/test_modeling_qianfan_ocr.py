# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch QianfanOCR model."""

import unittest

import pytest

from transformers import (
    AutoProcessor,
    QianfanOCRConfig,
    QianfanOCRVisionConfig,
    is_torch_available,
)
from transformers.models.qwen3 import Qwen3Config
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import QianfanOCRForConditionalGeneration, QianfanOCRModel


class QianfanOCRVisionText2TextModelTester(VLMModelTester):
    base_model_class = QianfanOCRModel
    config_class = QianfanOCRConfig
    text_config_class = Qwen3Config
    vision_config_class = QianfanOCRVisionConfig
    conditional_generation_class = QianfanOCRForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 1)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("hidden_size", 128)
        kwargs.setdefault("intermediate_size", 256)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("vision_hidden_size", 32)
        kwargs.setdefault("vision_intermediate_size", 128)
        kwargs.setdefault("vision_num_hidden_layers", 2)
        kwargs.setdefault("vision_num_attention_heads", 4)
        kwargs.setdefault("vision_hidden_act", "quick_gelu")
        kwargs.setdefault("drop_path_rate", 0.0)
        kwargs.setdefault("use_absolute_position_embeddings", True)
        kwargs.setdefault("image_seq_length", 16)
        kwargs.setdefault("bos_token_id", 3)
        kwargs.setdefault("eos_token_id", 4)
        kwargs.setdefault("pad_token_id", 5)
        kwargs.setdefault("vocab_size", 99)
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("rope_theta", 10000)
        super().__init__(parent, **kwargs)

        # image_seq_length overrides the VLMModelTester default num_image_tokens-based seq_length
        self.seq_length = 7 + self.image_seq_length

    def get_vision_config(self):
        return self.vision_config_class(
            hidden_size=self.vision_hidden_size,
            intermediate_size=self.vision_intermediate_size,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
            hidden_act=self.vision_hidden_act,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            use_absolute_position_embeddings=self.use_absolute_position_embeddings,
            drop_path_rate=self.drop_path_rate,
        )

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            image_seq_length=self.image_seq_length,
            vision_feature_layer=self.vision_feature_layer,
            pad_token_id=self.pad_token_id,
        )

    def create_pixel_values(self):
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        return input_ids


@require_torch
class QianfanOCRModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = QianfanOCRVisionText2TextModelTester
    test_torch_exportable = False

    def test_reverse_loading_mapping(self):
        # Conversion happens only for the `ConditionalGeneration` model, not the base model
        original_classes = self.all_model_classes
        self.all_model_classes = (QianfanOCRForConditionalGeneration,) if is_torch_available() else ()
        try:
            super().test_reverse_loading_mapping()
        finally:
            self.all_model_classes = original_classes

    @unittest.skip(
        reason="Not high prio, fails with `torch._dynamo.exc.InternalTorchDynamoError: ValueRangeError: Invalid ranges [0:-0.500000000000000]`"
    )
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("DataParallel is a deprecated legacy API and not officially supported")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@slow
@require_torch_accelerator
class QianfanOCRIntegrationTest(unittest.TestCase):
    """Original integration test values come from a 4090 (SM 89) and have been adjusted for our CI A10 (SM 86)"""

    def setUp(self):
        # model weights in baidu/Qianfan-OCR will be updated after this PR get released in transformers,
        # use bairongz/QianfanOCR for testing and will update back to baidu/Qianfan-OCR after weight update
        self.model_checkpoint = "bairongz/QianfanOCR"
        self.image_url = url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg")
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_integration_forward(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_url},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, torch.bfloat16)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        self.assertEqual(outputs.logits.dtype, torch.bfloat16)

        actual_logits = outputs.logits[0, -1, :5].cpu().to(torch.float32)
        # fmt: off
        expected_logits = Expectations(
            {
                ("cuda", (8, 6)): torch.tensor([10.1250, 15.8125, 13.0625, 12.3125,  9.4375]),
                ("cuda", (8, 9)): torch.tensor([10.0625, 15.6875, 13.0000, 12.1875,  9.3750]),
                ("xpu", None): torch.tensor([10.1875, 15.8750, 13.1875, 12.3750,  9.6250]),
            }
        )  # fmt: skip
        self.assertTrue(
            torch.allclose(actual_logits, expected_logits.get_expectation(), atol=1e-3, rtol=1e-2),
            f"Actual logits: {actual_logits}\nExpected logits: {expected_logits.get_expectation()}",
        )

    def test_model_integration_generate(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_url},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        decoded = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # fmt: off
        expected_outputs = Expectations(
            {
                ("cuda", (8, 6)): "The image features two striped cats lying down and sleeping on a pink couch. They",
                ("cuda", (8, 9)): "The image features two striped cats lying down on a pink couch, seemingly asleep.",
                ("xpu", None): "The image features two striped cats lying down on a couch, both appearing to be",
            }
        )  # fmt: skip
        self.assertEqual(decoded, expected_outputs.get_expectation())

    def test_model_integration_generate_text_only(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        messages = [{"role": "user", "content": [{"type": "text", "text": "What is 1 + 1?"}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        decoded = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # fmt: off
        expected_outputs = Expectations(
            {
                ("cuda", None): "1 + 1 equals 2.",
                ("xpu", None): "1 + 1 equals 2.",
            }
        )  # fmt: skip
        self.assertEqual(decoded, expected_outputs.get_expectation())

    def test_model_integration_batched_generate(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.tokenizer.padding_side = "left"

        messages1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_url},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        messages2 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_url},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            [messages1, messages2],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        self.assertEqual(output.shape[0], 2)

        decoded_0 = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        decoded_1 = processor.decode(output[1, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # fmt: off
        expected_outputs_0 = Expectations(
            {
                ("cuda", None): "In the tranquil setting of this image, two tabby cats are the stars of",
                ("xpu", None): "In the tranquil setting of this image, two tabby cats are the stars of",
            }
        )  # fmt: skip
        expected_outputs_1 = Expectations(
            {
                ("cuda", (8, 6)): "The image features two striped cats lying down and sleeping on a pink couch. The",
                ("cuda", (8, 9)): "The image features two striped cats lying down on a pink couch, seemingly asleep.",
                ("xpu", None): "The image features two striped cats lying down on a couch, both appearing to be",
            }
        )  # fmt: skip
        self.assertEqual(decoded_0, expected_outputs_0.get_expectation())
        self.assertEqual(decoded_1, expected_outputs_1.get_expectation())
