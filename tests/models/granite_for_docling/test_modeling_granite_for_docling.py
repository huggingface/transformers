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

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

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

from ...test_modeling_common import floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


def _concat_features(features):
    if isinstance(features, torch.Tensor):
        return features
    return torch.cat(list(features), dim=0)


class GraniteForDoclingModelTester(VLMModelTester):
    base_model_class = GraniteForDoclingModel
    config_class = GraniteForDoclingConfig
    conditional_generation_class = GraniteForDoclingForConditionalGeneration
    text_config_class = GraniteForDoclingTextConfig
    vision_config_class = GraniteForDoclingVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("out_hidden_size", kwargs["hidden_size"])
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
        kwargs.setdefault("density_router_hidden_size", 8)
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [batch_size, self.num_tiles, self.num_channels, self.image_size, self.image_size], scale=1.0
        )


@require_torch
class GraniteForDoclingModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = GraniteForDoclingModelTester
    # The fine connector path only receives gradients when tiles are routed to it
    test_all_params_have_gradient = False
    # Image tokens of all tiles come flattened in tile order, not one row per image
    skip_test_image_features_output_shape = True

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

        coarse_tokens_all = _concat_features(coarse.pooler_output)
        fine_tokens_all = _concat_features(fine.pooler_output)
        mixed_tokens_all = _concat_features(mixed.pooler_output)
        self.assertEqual(coarse_tokens_all.shape, (batch_size * coarse_tokens, hidden_size))
        self.assertEqual(fine_tokens_all.shape, (batch_size * fine_tokens, hidden_size))
        # Mixed routes: the first tile is fine, the others coarse
        self.assertEqual(mixed_tokens_all.shape, (fine_tokens + (batch_size - 1) * coarse_tokens, hidden_size))
        torch.testing.assert_close(mixed_tokens_all[:fine_tokens], fine_tokens_all[:fine_tokens])
        torch.testing.assert_close(mixed_tokens_all[fine_tokens:], coarse_tokens_all[coarse_tokens:])
        for coarse_features, fine_features, mixed_features in zip(
            coarse.deepstack_features, fine.deepstack_features, mixed.deepstack_features
        ):
            coarse_features = _concat_features(coarse_features)
            fine_features = _concat_features(fine_features)
            mixed_features = _concat_features(mixed_features)
            self.assertEqual(coarse_features.shape, (batch_size * coarse_tokens, hidden_size))
            self.assertEqual(fine_features.shape, (batch_size * fine_tokens, hidden_size))
            torch.testing.assert_close(mixed_features[:fine_tokens], fine_features[:fine_tokens])
            torch.testing.assert_close(mixed_features[fine_tokens:], coarse_features[coarse_tokens:])

    def test_router_loss(self):
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
            if name.startswith("model.density_router."):
                self.assertIsNotNone(parameter.grad, f"{name} has no gradient")

        # Without the router the same inputs still run and the loss is the language modeling loss only
        config.density_router_hidden_size = None
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        self.assertIsNone(model.model.density_router)
        outputs = model(**inputs_dict, labels=inputs_dict["input_ids"])
        self.assertIsNone(outputs.router_logits)
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_mtp_heads_of_a_checkpoint_are_ignored(self):
        # Serving engines use the multi-token prediction heads; transformers loads the checkpoint without them
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        state_dict = model.state_dict()
        state_dict["mtp.blocks.0.proj.weight"] = torch.zeros(2, 2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, state_dict=state_dict)
            reloaded, info = GraniteForDoclingForConditionalGeneration.from_pretrained(
                tmp_dir, output_loading_info=True
            )
        self.assertFalse(info["unexpected_keys"])
        self.assertFalse(hasattr(reloaded, "mtp"))

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

        self.assertEqual(len(features.deepstack_features), len(config.vision_config.deepstack_visual_indexes))
        for slot, vision_layer_idx in enumerate(config.vision_config.deepstack_visual_indexes):
            with torch.no_grad():
                expected = model.vision_model.connector._deepstack(slot, layer_outputs[vision_layer_idx], fine=False)
            torch.testing.assert_close(_concat_features(features.deepstack_features[slot]), expected.flatten(0, 1))

    def test_coarse_only_checkpoint(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.density_router_hidden_size = None
        config.vision_config.use_fine_route = False
        model = GraniteForDoclingModel(config).to(torch_device).eval()
        self.assertNotIn("fine", model.vision_model.connector.routes)
        self.assertIn("coarse", model.vision_model.connector.routes)
        self.assertFalse([name for name in model.state_dict() if "_fine" in name])

        pixel_values = inputs_dict["pixel_values"]
        with torch.no_grad():
            coarse = model.get_image_features(pixel_values)
        self.assertEqual(
            _concat_features(coarse.pooler_output).shape,
            (pixel_values.shape[0] * self.model_tester.num_image_tokens, config.text_config.hidden_size),
        )
        with self.assertRaises(ValueError):
            model.get_image_features(
                pixel_values, tile_fine_mask=torch.ones(pixel_values.shape[0], 1, dtype=torch.bool)
            )
        # The density router selects the fine path, so it cannot be built without it
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "use_fine_route=True"):
            GraniteForDoclingConfig(
                density_router_hidden_size=8,
                vision_config={"use_fine_route": False},
            )

    def test_beam_search_expands_per_sample_image_features(self):
        # Generation repeats each sample's image features once per beam, which needs one feature tensor per sample.
        # The two samples have different tile counts, so the second one carries an all-zero padding tile.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteForDoclingForConditionalGeneration(config).to(torch_device).eval()
        tile_tokens = self.model_tester.num_image_tokens
        seq_length = 2 * tile_tokens + 5
        text_ids = ids_tensor([2, seq_length], config.text_config.vocab_size - 4) + 4
        input_ids = text_ids.clone()
        input_ids[0, : 2 * tile_tokens] = config.image_token_id
        input_ids[1, :tile_tokens] = config.image_token_id
        tester = self.model_tester
        pixel_values = floats_tensor([2, 2, tester.num_channels, tester.image_size, tester.image_size], scale=1.0)
        pixel_values[1, 1] = 0.0
        generate_kwargs = {"max_new_tokens": 4, "min_new_tokens": 4, "pad_token_id": config.text_config.pad_token_id}

        with torch.no_grad():
            beams = model.generate(
                input_ids=input_ids, pixel_values=pixel_values, num_beams=2, num_return_sequences=2, **generate_kwargs
            )
            sampled = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                do_sample=True,
                num_return_sequences=2,
                **generate_kwargs,
            )
            first_alone = model.generate(
                input_ids=input_ids[:1],
                pixel_values=pixel_values[:1],
                num_beams=2,
                num_return_sequences=2,
                **generate_kwargs,
            )
            second_alone = model.generate(
                input_ids=input_ids[1:],
                pixel_values=pixel_values[1:, :1],
                num_beams=2,
                num_return_sequences=2,
                **generate_kwargs,
            )

        self.assertEqual(beams.shape, (4, seq_length + 4))
        self.assertEqual(sampled.shape, (4, seq_length + 4))
        # Beams 0-1 belong to the first sample and 2-3 to the second, as when each sample runs alone
        torch.testing.assert_close(beams[:2], first_alone)
        torch.testing.assert_close(beams[2:], second_alone)


@require_torch
class GraniteForDoclingIntegrationTest(unittest.TestCase):
    model_id = "docling-project/granite-for-docling-500m"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Page 1 of Auer et al., "Docling Technical Report", arXiv:2408.09869, hosted next to the checkpoint
        self.image = load_image(
            url_to_local_path(
                "https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png"
            )
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<doclang>"}]}]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_integration(self):
        expected_num_tiles = 13
        # skip_special_tokens=True drops dedicated DocLang tag tokens, so this is the visible prefix.
        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): 'furniture"/>arXiv:2408.09869v5  [cs.CL]  9 Dec 2024logo"/>Docling Technical ReportVersion 1.0'
            }
        )
        model = GraniteForDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt").to(model.device)
        self.assertEqual(inputs["pixel_values"].shape[1], expected_num_tiles)
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), expected_num_tiles * 64)

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertEqual(generated_text, expected_output.get_expectation())

    @slow
    def test_model_integration_fine_route(self):
        expected_num_tiles = 13
        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): 'furniture"/>arXiv:2408.09869v5 [cs.CL] 9 Dec 2024logo"/>Docling Technical ReportVersion 1.0'
            }
        )
        model = GraniteForDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt", fine_route=True).to(
            model.device
        )
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), expected_num_tiles * 256)
        self.assertTrue(inputs["tile_fine_mask"].all())

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertEqual(generated_text, expected_output.get_expectation())

    @slow
    def test_model_integration_batched(self):
        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): [
                    'furniture"/>arXiv:2408.09869v5  [cs.CL]  9 Dec ',
                    'furniture"/>arXiv:2408.09869v5  [cs.CL]  9 Dec ',
                ]
            }
        )
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
        self.assertEqual(generated_texts, expected_output.get_expectation())
