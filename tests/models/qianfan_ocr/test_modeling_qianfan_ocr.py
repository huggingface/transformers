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
from pathlib import Path

import pytest

from transformers import (
    AutoProcessor,
    QianfanOCRConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from transformers import QianfanOCRForConditionalGeneration, QianfanOCRModel

if is_vision_available():
    from PIL import Image

class QianfanOCRVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        image_seq_length=64,
        vision_feature_layer=-1,
        ignore_index=-100,
        image_token_id=1,
        num_channels=3,
        image_size=64,
        is_training=True,
        text_config={
            "model_type": "qwen3",
            "vocab_size": 99,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "tie_word_embeddings": False,
            "bos_token_id": 3,
            "eos_token_id": 4,
            "pad_token_id": 5,
        },
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 64,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_act": "quick_gelu",
            "use_absolute_position_embeddings": True,
            "drop_path_rate": 0.0,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_id = image_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.vision_feature_layer = vision_feature_layer
        self.is_training = is_training
        self.image_seq_length = image_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length + image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return QianfanOCRConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            image_seq_length=self.image_seq_length,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = QianfanOCRForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_model_fp16_autocast_forward(self, config, input_ids, pixel_values, attention_mask):
        config.dtype = torch.float16
        model = QianfanOCRForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type=torch_device, dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class QianfanOCRModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (QianfanOCRForConditionalGeneration, QianfanOCRModel) if is_torch_available() else ()
    all_generative_model_classes = (QianfanOCRForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": QianfanOCRForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = QianfanOCRVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=QianfanOCRConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Compile not yet supported in multimodal models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile fullgraph not yet supported in multimodal models")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    # @unittest.skip(reason="DataParallel not supported for multimodal models with dynamic image tokens")
    # def test_multi_gpu_data_parallel_forward(self):
    #     pass

    def test_model_fp16_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_fp16_forward(
            config,
            inputs_dict["input_ids"],
            inputs_dict["pixel_values"],
            inputs_dict["attention_mask"],
        )

    def test_model_fp16_autocast_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_fp16_autocast_forward(
            config,
            inputs_dict["input_ids"],
            inputs_dict["pixel_values"],
            inputs_dict["attention_mask"],
        )


@slow
@require_torch_accelerator
class QianfanOCRIntegrationTest(unittest.TestCase):
    def setUp(self):
        # model weights in baidu/Qianfan-OCR will be updated after this PR get released in transformers,
        # use bairongz/QianfanOCR for testing and will update back to baidu/Qianfan-OCR after weight update
        self.model_checkpoint = "/mnt/cfs_bj_mt/workspace/zhuangbairong/qfocr-trs-qa/Qianfan-OCR-hf"
        self.image_path = (
            Path(__file__).parents[2] / "fixtures" / "tests_samples" / "COCO" / "000000039769.png"
        )
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_integration_forward(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        image = Image.open(self.image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
                ("cuda", 8): torch.tensor([8.4375, 14.125, 10.8125, 11.5, 7.5625]),
            }
        )  # fmt: skip
        self.assertTrue(
            torch.allclose(actual_logits, expected_logits.get_expectation(), atol=0.1),
            f"Actual logits: {actual_logits}\nExpected logits: {expected_logits.get_expectation()}",
        )

    def test_model_integration_generate(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        image = Image.open(self.image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # fmt: off
        expected_outputs = Expectations(
            {
                ("cuda", 8): "The image features two cats lying on a couch, both appearing to be asleep. They are positioned next",
            }
        )  # fmt: skip
        self.assertEqual(decoded, expected_outputs.get_expectation())

    def test_model_integration_generate_text_only(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        messages = [{"role": "user", "content": "What is 1 + 1?"}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        decoded = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # fmt: off
        expected_outputs = Expectations(
            {
                ("cuda", 8): "1 + 1 equals 2.",
            }
        )  # fmt: skip
        self.assertEqual(decoded, expected_outputs.get_expectation())

    def test_model_integration_batched_generate(self):
        model = QianfanOCRForConditionalGeneration.from_pretrained(
            self.model_checkpoint, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.tokenizer.padding_side = "left"
        image = Image.open(self.image_path).convert("RGB")

        messages1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        messages2 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        texts = [
            processor.apply_chat_template(messages1, add_generation_prompt=True),
            processor.apply_chat_template(messages2, add_generation_prompt=True),
        ]
        inputs = processor(text=texts, images=[image, image], padding=True, return_tensors="pt").to(
            torch_device, torch.bfloat16
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        self.assertEqual(output.shape[0], 2)

        decoded_0 = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        decoded_1 = processor.decode(output[1, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # fmt: off
        expected_outputs_0 = Expectations(
            {
                ("cuda", 8): "In the tranquil setting of this image, two brown and black striped cats are the stars of the scene",
            }
        )  # fmt: skip
        expected_outputs_1 = Expectations(
            {
                ("cuda", 8): "In the image, two striped cats are comfortably sprawled out on a pink couch, each with a",
            }
        )  # fmt: skip
        self.assertEqual(decoded_0, expected_outputs_0.get_expectation())
        self.assertEqual(decoded_1, expected_outputs_1.get_expectation())
