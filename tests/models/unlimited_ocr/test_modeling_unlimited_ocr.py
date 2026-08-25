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
"""Testing suite for the PyTorch UnlimitedOCR model."""

import copy
import unittest

from transformers import (
    LlamaTokenizerFast,
    UnlimitedOCRConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        UnlimitedOCRForConditionalGeneration,
        UnlimitedOCRImageProcessor,
        UnlimitedOCRModel,
        UnlimitedOCRProcessor,
    )
    from transformers.models.unlimited_ocr.configuration_unlimited_ocr import (
        UnlimitedOCRTextConfig,
        UnlimitedOCRVisionConfig,
    )

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import load_image


class UnlimitedOCRVisionText2TextModelTester(VLMModelTester):
    base_model_class = UnlimitedOCRModel
    config_class = UnlimitedOCRConfig
    conditional_generation_class = UnlimitedOCRForConditionalGeneration
    text_config_class = UnlimitedOCRTextConfig
    vision_config_class = UnlimitedOCRVisionConfig

    def __init__(self, parent, **kwargs):
        # Global view: queries_per_side = image_size / patch_size / 4, tokens =
        # queries_per_side * (queries_per_side + 1) + 1 view_separator (no local crops).
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
        kwargs.setdefault("first_k_dense_replace", 1)
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
            "image_size": 16,
            "patch_size": 2,
            "hidden_act": "quick_gelu",
            "num_channels": 3,
        }
        self.projector_input_dim = self.encoder_config["hidden_size"] + self.sam_config["downsample_channels"][-1]

    def get_vision_config(self):
        return UnlimitedOCRVisionConfig(
            sam_config=self.sam_config,
            encoder_config=self.encoder_config,
        )

    def get_config(self):
        return self.config_class(
            vision_config=self.get_vision_config(),
            text_config=self.get_text_config(),
            image_token_id=self.image_token_id,
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        return {
            "num_local_patches": torch.tensor([[1, 1]] * self.batch_size, dtype=torch.long, device=torch_device),
        }


@require_torch
class UnlimitedOCRModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = UnlimitedOCRVisionText2TextModelTester
    test_all_params_have_gradient = False

    @unittest.skip(
        reason="UnlimitedOCR CLIP tower always uses SDPA with a non-null attention path independent of LM dispatch."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="CLIP tower forces SDPA inside UnlimitedOCRClipVisionAttention; incompatible with output_attentions."
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        reason="CLIP tower forces SDPA inside UnlimitedOCRClipVisionAttention independent of config setter."
    )
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(reason="Text tower sets _supports_sdpa=False for bit-identical logits vs hub.")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(
        reason="image_newline / view_separator are top-level Parameters not covered by default device_map keys."
    )
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Vision path does not expose recordable hidden_states through get_image_features.")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip(reason="Vision path does not expose recordable attentions through get_image_features.")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip(
        reason="CLIP tower forces SDPA inside UnlimitedOCRClipVisionAttention independent of config setter."
    )
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Text tower sets _supports_sdpa=False for bit-identical logits vs hub.")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip(reason="image_newline / view_separator need dedicated _init_weights coverage beyond common buffers.")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="CLIP/SAM position buffers are not yet covered by meta-device reinit checks.")
    def test_init_weights_can_init_buffers(self):
        pass

    @unittest.skip(reason="Conversion WeightRenaming for sam_encoder.blocks→layers is not reverse-matchable.")
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip(reason="Inherited text TP plan lists q_norm/k_norm unused by UnlimitedOCR MHA.")
    def test_tp_plan_matches_params(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = super()._image_features_prepare_config_and_inputs()
        # Fused SAM+CLIP features are encoder_hidden + sam_downsample, not a nested `hidden_size`.
        config.vision_config.hidden_size = (
            config.vision_config.encoder_config.hidden_size
            + config.vision_config.sam_config.downsample_channels[-1]
        )
        return config, inputs_dict

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)

            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            curr_input_dict["num_local_patches"] = curr_input_dict["num_local_patches"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            curr_input_dict = {key: val[:1] for key, val in curr_input_dict.items()}
            for key in ("input_ids", "attention_mask"):
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)

            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]], dim=0
            )
            curr_input_dict["num_local_patches"] = torch.cat(
                [curr_input_dict["num_local_patches"], curr_input_dict["num_local_patches"]], dim=0
            )
            _ = model(**curr_input_dict)


@require_torch
class UnlimitedOCRIntegrationTest(unittest.TestCase):
    # Hub checkpoint (remote-code layout); in-tree classes + conversion_mapping load it.
    model_id = "baidu/Unlimited-OCR"
    prompt = "<image>document parsing."

    def setUp(self):
        tokenizer = LlamaTokenizerFast.from_pretrained(self.model_id)
        self.processor = UnlimitedOCRProcessor(
            image_processor=UnlimitedOCRImageProcessor(),
            tokenizer=tokenizer,
            patch_size=16,
            downsample_ratio=4,
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_model(self):
        return UnlimitedOCRForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        ).eval()

    def _to_model(self, batch, model):
        inputs = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
        if "pixel_values_local" in inputs:
            inputs["pixel_values_local"] = inputs["pixel_values_local"].to(dtype=model.dtype)
        return inputs

    @slow
    def test_integration_generate_no_tiles(self):
        # Same path as scripts/unlimited_ocr/demo.py `baidu_no_tiles` (image under tile_size → no local crops).
        model = self._load_model()
        image = load_image(
            url_to_local_path("https://huggingface.co/baidu/Unlimited-OCR/resolve/main/assets/baidu.png")
        )
        inputs = self._to_model(
            self.processor(images=image, text=self.prompt, return_tensors="pt"),
            model,
        )
        self.assertNotIn("pixel_values_local", inputs)

        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        generated_ids = output[0, inputs["input_ids"].shape[1] :]

        EXPECTED_OUTPUT_TOKEN_IDS = [
            10212,
            764,
            18,
            14,
            223,
            18,
            14,
            223,
            8834,
            14,
            223,
            8834,
            63,
            128819,
            36,
            2238,
            2937,
            223,
            39063,
            1,
        ]  # fmt: skip
        self.assertEqual(generated_ids.tolist(), EXPECTED_OUTPUT_TOKEN_IDS)

    @slow
    def test_integration_generate_tiled(self):
        # Same path as scripts/unlimited_ocr/demo.py `synthetic_tiled` (local crops + num_local_patches).
        model = self._load_model()
        image = Image.new("RGB", (800, 500), color=(40, 80, 120))
        inputs = self._to_model(
            self.processor(images=image, text=self.prompt, return_tensors="pt"),
            model,
        )
        self.assertIn("pixel_values_local", inputs)
        self.assertEqual(inputs["num_local_patches"].tolist(), [[3, 5]])

        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=8)
        generated_ids = output[0, inputs["input_ids"].shape[1] :]

        EXPECTED_OUTPUT_TOKEN_IDS = [128818, 10253, 764, 18, 14, 223, 18, 14]  # fmt: skip
        self.assertEqual(generated_ids.tolist(), EXPECTED_OUTPUT_TOKEN_IDS)
