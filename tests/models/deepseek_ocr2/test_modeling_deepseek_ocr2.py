# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DeepseekOcr2 model."""

import unittest

from transformers import (
    AutoProcessor,
    DeepseekOcr2Config,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        DeepseekOcr2ForConditionalGeneration,
        DeepseekOcr2Model,
    )

if is_vision_available():
    from transformers.image_utils import load_image


class DeepseekOcr2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        image_size=16,
        image_token_index=1,
        is_training=True,
        sam_config={
            "hidden_size": 32,
            "output_channels": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 2,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "window_size": 4,
            "global_attn_indexes": [1],
            "downsample_channels": [32, 64],
        },
        encoder_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
        },
        text_config={
            "model_type": "deepseek_ocr2_text",
            "vocab_size": 99,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "tie_word_embeddings": False,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 4,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "moe_intermediate_size": 64,
            "num_experts_per_tok": 2,
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.image_token_index = image_token_index
        self.is_training = is_training
        self.sam_config = sam_config
        self.encoder_config = encoder_config
        self.text_config = text_config

        # VisionModel always selects query_768 (144 tokens) for small images + 1 separator
        self.num_image_tokens = 145
        self.seq_length = seq_length + self.num_image_tokens

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

        self.pad_token_id = text_config["pad_token_id"]

    def get_config(self):
        vision_cfg = {"encoder_config": self.encoder_config, "sam_config": self.sam_config}
        return DeepseekOcr2Config(
            vision_config=vision_cfg,
            text_config=self.text_config,
            image_token_id=self.image_token_index,
            projector_input_dim=self.encoder_config["hidden_size"],  # 64
            projector_n_embed=self.text_config["hidden_size"],  # 128
            projector_type="linear",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        # Avoid collision with image_token_index and place image tokens at the start
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class DeepseekOcr2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DeepseekOcr2Model,
            DeepseekOcr2ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-text-to-text": DeepseekOcr2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_all_params_have_gradient = False
    _is_composite = True

    def setUp(self):
        self.model_tester = DeepseekOcr2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekOcr2Config, has_text_modality=False)

    @unittest.skip("SDPA/FlexAttn not yet supported")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()


@require_torch
class DeepseekOcr2IntegrationTest(unittest.TestCase):
    model_id = "thisisiron/DeepSeek-OCR-2-hf"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_free_ocr(self):
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device, attn_implementation="eager"
        )
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )

        inputs = self.processor(images=image, text="<image>\nFree OCR.", return_tensors="pt").to(
            model.device, dtype=torch.bfloat16
        )
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM\n\nName/"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_grounding_markdown(self):
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device, attn_implementation="eager"
        )
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )

        inputs = self.processor(
            images=image, text="<image>\n<|grounding|>Convert the document to markdown.", return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        expected_output = "<|ref|>title<|/ref|><|det|>[[330, 198, 558, 230]]<|/det|>\n# R"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_batched(self):
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device, attn_implementation="eager"
        )
        image1 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )
        image2 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        )

        inputs = self.processor(
            images=[image1, image2],
            text=["<image>\nFree OCR.", "<image>\nFree OCR."],
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded_output = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = [
            "R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM\n\nName/",
            "# Reducing the number of images\n\nIt is also believed that the performance of a website is a critical",
        ]
        self.assertEqual(decoded_output, expected_output)
