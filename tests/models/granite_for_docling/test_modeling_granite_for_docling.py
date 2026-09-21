# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteForDocling model."""

import unittest

from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    GraniteForDoclingConfig,
    GraniteForDoclingForConditionalGeneration,
    GraniteForDoclingModel,
    GraniteForDoclingTextConfig,
    GraniteForDoclingVisionConfig,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class GraniteForDoclingModelTester(VLMModelTester):
    base_model_class = GraniteForDoclingModel
    config_class = GraniteForDoclingConfig
    conditional_generation_class = GraniteForDoclingForConditionalGeneration
    text_config_class = GraniteForDoclingTextConfig
    vision_config_class = GraniteForDoclingVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("shared_intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        # The checkpoint defaults scale attention scores by 1/64; keep the usual 1/sqrt(head_dim) for the tiny model
        kwargs.setdefault("attention_multiplier", (kwargs["hidden_size"] // kwargs["num_attention_heads"]) ** -0.5)
        # A 16x16 tile with 4x4 patches gives 16 patch tokens per tile; pixel shuffling by 2 leaves 4 image tokens
        # on the coarse path and 16 on the fine path.
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("scale_factor", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("num_tiles", 1)
        kwargs.setdefault("deepstack_visual_indexes", [1])
        kwargs.setdefault("deepstack_attn_layers", [0])
        kwargs.setdefault("num_mtp_layers", 2)
        kwargs.setdefault("density_router_hidden_size", 8)
        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        return floats_tensor(
            [self.batch_size, self.num_tiles, self.num_channels, self.image_size, self.image_size], scale=1.0
        )


@require_torch
class GraniteForDoclingModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = GraniteForDoclingModelTester
    # The fine connector path only receives gradients when tiles are routed to it
    test_all_params_have_gradient = False

    def test_fine_route_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingModel(config).to(torch_device).eval()
        pixel_values = inputs_dict["pixel_values"]
        batch_size = pixel_values.shape[0]
        hidden_size = config.text_config.hidden_size
        coarse_tokens = self.model_tester.num_image_tokens
        fine_tokens = coarse_tokens * 4

        with torch.no_grad():
            coarse = model.get_image_features(pixel_values)
            fine = model.get_image_features(pixel_values, tile_fine_mask=torch.ones(batch_size, 1, dtype=torch.bool))
            mixed_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
            mixed_mask[0] = True
            mixed = model.get_image_features(pixel_values, tile_fine_mask=mixed_mask)

        self.assertEqual(coarse.pooler_output.shape, (batch_size, coarse_tokens, hidden_size))
        self.assertEqual(fine.pooler_output.shape, (batch_size, fine_tokens, hidden_size))
        # Mixed routes are flattened in tile order: the first tile is fine, the others coarse
        self.assertEqual(mixed.pooler_output.shape, (fine_tokens + (batch_size - 1) * coarse_tokens, hidden_size))
        torch.testing.assert_close(mixed.pooler_output[:fine_tokens], fine.pooler_output[0])
        torch.testing.assert_close(mixed.pooler_output[fine_tokens:], coarse.pooler_output[1:].flatten(0, 1))
        for coarse_features, fine_features, mixed_features in zip(
            coarse.deepstack_features, fine.deepstack_features, mixed.deepstack_features
        ):
            self.assertEqual(coarse_features.shape, (batch_size, coarse_tokens, hidden_size))
            self.assertEqual(fine_features.shape, (batch_size, fine_tokens, hidden_size))
            torch.testing.assert_close(mixed_features[:fine_tokens], fine_features[0])
            torch.testing.assert_close(mixed_features[fine_tokens:], coarse_features[1:].flatten(0, 1))

    def test_mtp_and_router_losses(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device)
        model.train()
        batch_size = inputs_dict["input_ids"].shape[0]
        router_labels = torch.zeros(batch_size, dtype=torch.bool, device=torch_device)
        router_labels[0] = True

        outputs = model(**inputs_dict, labels=inputs_dict["input_ids"], router_labels=router_labels)
        self.assertEqual(outputs.router_logits.shape, (batch_size,))
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(("mtp.", "model.density_router.")):
                self.assertIsNotNone(parameter.grad, f"{name} has no gradient")

        # Without the optional heads the same inputs still run and the loss is the language modeling loss only
        config.num_mtp_layers = 0
        config.density_router_hidden_size = None
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        self.assertIsNone(model.mtp)
        self.assertIsNone(model.model.density_router)
        outputs = model(**inputs_dict, labels=inputs_dict["input_ids"])
        self.assertIsNone(outputs.router_logits)
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_predict_fine_route(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            fine_route = model.predict_fine_route(inputs_dict["pixel_values"])
        self.assertEqual(fine_route.shape, (inputs_dict["pixel_values"].shape[0],))
        self.assertEqual(fine_route.dtype, torch.bool)

    def test_fine_route_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        batch_size = input_ids.shape[0]
        # The tester places `num_image_tokens` image tokens at the start of the sequence; the fine route needs 4x
        extra_image_tokens = torch.full(
            (batch_size, 3 * self.model_tester.num_image_tokens), config.image_token_id, device=input_ids.device
        )
        fine_input_ids = torch.cat([extra_image_tokens, input_ids], dim=1)
        tile_fine_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=fine_input_ids, pixel_values=inputs_dict["pixel_values"], tile_fine_mask=tile_fine_mask
            )
        self.assertEqual(outputs.logits.shape, (batch_size, fine_input_ids.shape[1], config.text_config.vocab_size))
        # The number of image tokens must match the fine features
        with self.assertRaises(ValueError):
            model(input_ids=input_ids, pixel_values=inputs_dict["pixel_values"], tile_fine_mask=tile_fine_mask)

    def test_deepstack_taps_vision_layer_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingModel(config).to(torch_device).eval()
        pixel_values = inputs_dict["pixel_values"]

        # `deepstack_visual_indexes` names vision layers: the tap is the output of that layer
        layer_outputs = {}
        hooks = [
            layer.register_forward_hook(
                lambda module, args, output, index=index: layer_outputs.__setitem__(
                    index, output[0] if isinstance(output, tuple) else output
                )
            )
            for index, layer in enumerate(model.vision_model.encoder.layers)
        ]
        with torch.no_grad():
            features = model.get_image_features(pixel_values)
        for hook in hooks:
            hook.remove()

        self.assertEqual(len(features.deepstack_features), len(config.deepstack_visual_indexes))
        for slot, vision_layer_idx in enumerate(config.deepstack_visual_indexes):
            with torch.no_grad():
                expected = model.connector.deepstack(slot, layer_outputs[vision_layer_idx])
            torch.testing.assert_close(features.deepstack_features[slot], expected)

    def test_coarse_only_checkpoint(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.density_router_hidden_size = None
        config.use_fine_route = False
        model = GraniteForDoclingModel(config).to(torch_device).eval()
        self.assertIsNone(model.connector.proj_fine)
        self.assertIsNone(model.connector.deepstack_mergers_fine)
        self.assertFalse([name for name in model.state_dict() if "_fine" in name])

        pixel_values = inputs_dict["pixel_values"]
        with torch.no_grad():
            coarse = model.get_image_features(pixel_values)
        self.assertEqual(coarse.pooler_output.shape[1], self.model_tester.num_image_tokens)
        with self.assertRaises(ValueError):
            model.get_image_features(
                pixel_values, tile_fine_mask=torch.ones(pixel_values.shape[0], 1, dtype=torch.bool)
            )
        # The density router selects the fine path, so it cannot be built without it
        with self.assertRaises(ValueError):
            GraniteForDoclingConfig(density_router_hidden_size=8, use_fine_route=False)


@require_torch
class GraniteForDoclingIntegrationTest(unittest.TestCase):
    model_id = "docling-project/granite-for-docling-500m"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Page 1 of Auer et al., "Docling Technical Report", arXiv:2408.09869, hosted next to the checkpoint
        self.image = load_image(hf_hub_download(self.model_id, "docling_technical_report_p1.png"))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<doclang>"}]}]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_integration(self):
        model = GraniteForDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt").to(model.device)
        self.assertEqual(inputs["pixel_values"].shape[1], EXPECTED_NUM_TILES)
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), EXPECTED_NUM_TILES * 64)

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_COARSE_OUTPUT.get_expectation())

    @slow
    def test_model_integration_fine_route(self):
        model = GraniteForDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt", fine_route=True).to(
            model.device
        )
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), EXPECTED_NUM_TILES * 256)
        self.assertTrue(inputs["tile_fine_mask"].all())

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_FINE_OUTPUT.get_expectation())

    @slow
    def test_model_integration_batched(self):
        model = GraniteForDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(
            text=[self.prompt, self.prompt], images=[self.image, self.image], return_tensors="pt", padding=True
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_texts = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        self.assertEqual(generated_texts, EXPECTED_BATCHED_OUTPUT.get_expectation())


EXPECTED_NUM_TILES = 13
# skip_special_tokens=True drops dedicated DocLang tag tokens, so these are the visible prefixes.
EXPECTED_COARSE_OUTPUT = Expectations({("cuda", None): "furniture\"/>arXiv:2408.09869v5  [cs.CL]  9 Dec 2024logo\"/>Docling Technical ReportVersion 1.0"})  # fmt: skip
EXPECTED_FINE_OUTPUT = Expectations({("cuda", None): "furniture\"/>arXiv:2408.09869v5 [cs.CL] 9 Dec 2024logo\"/>Docling Technical ReportVersion 1.0"})  # fmt: skip
EXPECTED_BATCHED_OUTPUT = Expectations({("cuda", None): ["furniture\"/>arXiv:2408.09869v5  [cs.CL]  9 Dec ", "furniture\"/>arXiv:2408.09869v5  [cs.CL]  9 Dec "]})  # fmt: skip
