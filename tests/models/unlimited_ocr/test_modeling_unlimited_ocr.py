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
"""Testing suite for the PyTorch UnlimitedOcr model."""

import unittest

from transformers import (
    AutoProcessor,
    UnlimitedOcrConfig,
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

from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        UnlimitedOcrForConditionalGeneration,
        UnlimitedOcrModel,
    )
    from transformers.models.unlimited_ocr.configuration_unlimited_ocr import (
        UnlimitedOcrTextConfig,
        UnlimitedOcrVisionConfig,
    )

if is_vision_available():
    from transformers.image_utils import load_image


class UnlimitedOcrVisionText2TextModelTester(VLMModelTester):
    base_model_class = UnlimitedOcrModel
    config_class = UnlimitedOcrConfig
    conditional_generation_class = UnlimitedOcrForConditionalGeneration
    text_config_class = UnlimitedOcrTextConfig
    vision_config_class = UnlimitedOcrVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("num_image_tokens", 7)
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
            "mlp_dim": 128,
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
        return UnlimitedOcrVisionConfig(
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
class UnlimitedOcrModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = UnlimitedOcrVisionText2TextModelTester
    test_all_params_have_gradient = False
    test_torch_exportable = False

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        # `is_moe_model` (used by generation tests to pick a looser tolerance for MoE routing noise) checks
        # `config._experts_implementation`, which is only populated when a model is instantiated. Some tests
        # (e.g. `test_generate_with_static_cache`) check it on the bare config, so mark the model's default
        # experts implementation here to get the MoE tolerance. The ring-buffer vs. static sliding cache
        # differ by ~1e-3 once amplified through the sparse expert routing, which is well within that bound.
        config._experts_implementation = "grouped_mm"
        return config, inputs_dict

    @unittest.skip(
        reason="UnlimitedOcrVisionModel builds a hybrid bidirectional+causal mask internally, so SDPA is always called with a non-null `attn_mask`."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="UnlimitedOcrVisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="UnlimitedOcrVisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="UnlimitedOcrVisionModel uses `self.query_*.weight` directly, causing device mismatch when offloading."
    )
    def test_disk_offload_safetensors(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = super()._image_features_prepare_config_and_inputs()
        # `get_image_features` returns the concatenation of the SAM feature map and the CLIP encoder output as
        # `last_hidden_state`, so its hidden size is the sum of the two. `vision_config` has no `hidden_size` of
        # its own, so set it here for `test_get_image_features_output`.
        config.vision_config.hidden_size = (
            config.vision_config.sam_config.downsample_channels[-1] + config.vision_config.encoder_config.hidden_size
        )
        return config, inputs_dict


@require_torch
class UnlimitedOcrIntegrationTest(unittest.TestCase):
    model_id = "guarin/Unlimited-OCR"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(self.model_id, device_map=torch_device).eval()
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        inputs = self.processor(images=image, text="<image>\ndocument parsing.", return_tensors="pt").to(model.device)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "image [383, 87, 497, 171]\ntitle [333",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing_grounding(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(self.model_id, device_map=torch_device).eval()
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        inputs = self.processor(
            images=image,
            text="<image>\ndocument parsing.",
            return_tensors="pt",
        ).to(model.device)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "<|det|>image [383, 87, 497, 171]<|/det|>\n<|det|>title [333",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing_batched(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(self.model_id, device_map=torch_device).eval()
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
            text=["<image>\ndocument parsing.", "<image>\ndocument parsing."],
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)

        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): [
                    "image [383, 87, 497, 171]\ntitle [333",
                    "header [53, 23, 365, 41]Advanced Template and Styl",
                ],
            }
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_multi_page_document_parsing(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(self.model_id, device_map=torch_device).eval()
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
            text="<image><image>\nMulti page parsing.",
            crop_to_patches=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "<PAGE>image [382, 87, 489, 174]\n",
            }
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
