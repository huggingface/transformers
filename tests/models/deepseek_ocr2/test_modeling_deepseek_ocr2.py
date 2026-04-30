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
from transformers.testing_utils import Expectations, cleanup, require_torch, slow, torch_device

from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        DeepseekOcr2ForConditionalGeneration,
        DeepseekOcr2Model,
    )
    from transformers.models.deepseek_ocr2.configuration_deepseek_ocr2 import (
        DeepseekOcr2TextConfig,
        DeepseekOcr2VisionConfig,
    )

if is_vision_available():
    from transformers.image_utils import load_image


class DeepseekOcr2VisionText2TextModelTester(VLMModelTester):
    base_model_class = DeepseekOcr2Model
    config_class = DeepseekOcr2Config
    conditional_generation_class = DeepseekOcr2ForConditionalGeneration
    text_config_class = DeepseekOcr2TextConfig
    vision_config_class = DeepseekOcr2VisionConfig

    def __init__(self, parent, **kwargs):
        # VisionModel always selects query_768_resolution (144 tokens) for small images + 1 separator
        kwargs.setdefault("num_image_tokens", 145)
        kwargs.setdefault("image_token_id", 1)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("hidden_size", 128)
        kwargs.setdefault("intermediate_size", 256)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("bos_token_id", 2)
        kwargs.setdefault("eos_token_id", 3)
        kwargs.setdefault("pad_token_id", 4)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("moe_intermediate_size", 64)
        kwargs.setdefault("num_experts_per_tok", 2)
        super().__init__(parent, **kwargs)

        self.sam_config = {
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
        }
        self.encoder_config = {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rms_norm_eps": 1.0,
        }

    def get_vision_config(self):
        return DeepseekOcr2VisionConfig(
            sam_config=self.sam_config,
            encoder_config=self.encoder_config,
        )

    def get_config(self):
        return self.config_class(
            vision_config=self.get_vision_config(),
            text_config=self.get_text_config(),
            image_token_id=self.image_token_id,
        )


@require_torch
class DeepseekOcr2ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekOcr2VisionText2TextModelTester
    test_all_params_have_gradient = False
    test_torch_exportable = False

    @unittest.skip(
        reason="DeepseekOcr2VisionModel builds a hybrid bidirectional+causal mask internally, so SDPA is always called with a non-null `attn_mask`."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="DeepseekOcr2VisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="DeepseekOcr2VisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="DeepseekOcr2VisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_disk_offload_safetensors(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = super()._image_features_prepare_config_and_inputs()
        # test_get_image_features_output expects vision_config.hidden_size, but ours is in encoder_config.
        config.vision_config.hidden_size = config.vision_config.encoder_config.hidden_size
        return config, inputs_dict


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
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        inputs = self.processor(images=image, text="<image>\nFree OCR.", return_tensors="pt").to(
            model.device, dtype=torch.bfloat16
        )
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM\n\nName/",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    def test_small_model_integration_test_grounding_markdown(self):
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        inputs = self.processor(
            images=image,
            text="<image>\n<|grounding|>Convert the document to markdown.",
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "<|ref|>title<|/ref|><|det|>[[330, 198, 559, 230]]<|/det|>\n# R",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    def test_small_model_integration_test_batched(self):
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        image1 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        image2 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
            )
        )
        inputs = self.processor(
            images=[image1, image2],
            text=["<image>\nFree OCR.", "<image>\nFree OCR."],
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)
        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): [
                    "R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM\n\nName/",
                    "# Reducing the number of images\n\nIt is also believed that the performance of a website is a critical",
                ],
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
