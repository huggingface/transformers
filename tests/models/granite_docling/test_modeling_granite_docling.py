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
"""Testing suite for the PyTorch GraniteDocling model."""

import copy
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import (
    AutoProcessor,
    GraniteDoclingConfig,
    GraniteDoclingForConditionalGeneration,
    GraniteDoclingModel,
    GraniteDoclingTextConfig,
    GraniteDoclingVisionConfig,
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
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


def _concat_features(features):
    if isinstance(features, torch.Tensor):
        return features
    return torch.cat(list(features), dim=0)


class GraniteDoclingModelTester(VLMModelTester):
    base_model_class = GraniteDoclingModel
    config_class = GraniteDoclingConfig
    conditional_generation_class = GraniteDoclingForConditionalGeneration
    text_config_class = GraniteDoclingTextConfig
    vision_config_class = GraniteDoclingVisionConfig

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
        kwargs.setdefault("seq_length", 7 + 4 * kwargs["num_image_tokens"])
        kwargs.setdefault("deepstack_visual_indexes", [1])
        kwargs.setdefault("density_router_hidden_size", 8)
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [batch_size, self.num_tiles, self.num_channels, self.image_size, self.image_size], scale=1.0
        )

    def place_image_tokens(self, input_ids, config):
        image_token_id = getattr(config, "image_token_id", self.image_token_id)
        input_ids = input_ids.clone()
        input_ids[input_ids == image_token_id] = self.bos_token_id
        input_ids[:, : self.num_image_tokens] = image_token_id
        # The first sample takes the fine path, so both connector routes receive a gradient.
        input_ids[0, : self.num_image_tokens * 4] = image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size=None):
        batch = pixel_values.shape[0]
        tile_fine_mask = torch.zeros(batch, dtype=torch.bool)
        tile_fine_mask[0] = True
        router_labels = torch.zeros(batch, dtype=torch.float32)
        router_labels[0] = 1.0
        return {"tile_fine_mask": tile_fine_mask, "router_labels": router_labels}


@require_torch
class GraniteDoclingModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = GraniteDoclingModelTester
    # Image tokens of all tiles come flattened in tile order, not one row per image
    skip_test_image_features_output_shape = True

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        # Like `labels`, the router labels are a training target, not a generation input
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        inputs_dict.pop("router_labels", None)
        return config, inputs_dict

    def test_fine_route_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingModel(config).to(torch_device).eval()
        pixel_values = inputs_dict["pixel_values"]
        batch_size = pixel_values.shape[0]
        hidden_size = config.text_config.hidden_size
        coarse_tokens = self.model_tester.num_image_tokens
        fine_tokens = coarse_tokens * 4

        with torch.no_grad():
            coarse = model.get_image_features(pixel_values)
            fine = model.get_image_features(pixel_values, tile_fine_mask=torch.ones(batch_size, dtype=torch.bool))
            mixed_mask = torch.zeros(batch_size, dtype=torch.bool)
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
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device)
        model.train()
        batch_size = inputs_dict["input_ids"].shape[0]
        router_labels = torch.zeros(batch_size, dtype=torch.bool, device=torch_device)
        router_labels[0] = True

        inputs = {key: value for key, value in inputs_dict.items() if key != "router_labels"}
        outputs = model(**inputs, labels=inputs["input_ids"], router_labels=router_labels)
        self.assertEqual(outputs.router_logits.shape, (batch_size,))
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("model.density_router."):
                self.assertIsNotNone(parameter.grad, f"{name} has no gradient")

        # Without the router the same inputs still run and the loss is the language modeling loss only
        config.density_router_hidden_size = None
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device).eval()
        self.assertIsNone(model.model.density_router)
        outputs = model(**inputs_dict, labels=inputs_dict["input_ids"])
        self.assertIsNone(outputs.router_logits)
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_mismatching_num_image_tokens(self):
        # The shared test concatenates the batch, which repeats the first sample's fine route.
        # The new pixel batch only marks sample 0, so mark that repeated sample as fine too.
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)

            input_ids = curr_input_dict["input_ids"]
            batch_size = input_ids.shape[0]
            _ = model(**curr_input_dict)

            smaller_input_ids, smaller_input_dict = self.model_tester._prepare_modality_inputs(
                input_ids, config, batch_size=batch_size - 1
            )
            smaller_input_dict.update({k: v for k, v in curr_input_dict.items() if k not in smaller_input_dict})
            if "mm_token_type_ids" in curr_input_dict:
                smaller_input_dict["mm_token_type_ids"] = curr_input_dict["mm_token_type_ids"]
            elif "token_type_ids" in curr_input_dict:
                smaller_input_dict["token_type_ids"] = curr_input_dict["token_type_ids"]
            with self.assertRaises(ValueError):
                _ = model(**smaller_input_dict)

            single_input_ids, single_input_dict = self.model_tester._prepare_modality_inputs(
                input_ids, config, batch_size=1
            )
            batch_tensors_to_double = ["input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"]
            for key in batch_tensors_to_double:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            double_input_ids, double_input_dict = self.model_tester._prepare_modality_inputs(
                input_ids, config, batch_size=batch_size * 2
            )
            double_input_dict.update({k: v for k, v in curr_input_dict.items() if k not in double_input_dict})
            if "mm_token_type_ids" in curr_input_dict:
                double_input_dict["mm_token_type_ids"] = curr_input_dict["mm_token_type_ids"]
            elif "token_type_ids" in curr_input_dict:
                double_input_dict["token_type_ids"] = curr_input_dict["token_type_ids"]
            fine = torch.zeros(batch_size * 2, dtype=torch.bool)
            fine[0] = True
            fine[batch_size] = True
            double_input_dict["tile_fine_mask"] = fine
            _ = model(**double_input_dict)

    def test_mtp_heads_of_a_checkpoint_are_ignored(self):
        # Serving engines use the multi-token prediction heads; transformers loads the checkpoint without them
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device).eval()
        state_dict = model.state_dict()
        state_dict["mtp.blocks.0.proj.weight"] = torch.zeros(2, 2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, state_dict=state_dict)
            reloaded, info = GraniteDoclingForConditionalGeneration.from_pretrained(tmp_dir, output_loading_info=True)
        self.assertFalse(info["unexpected_keys"])
        self.assertFalse(hasattr(reloaded, "mtp"))

    def test_predict_fine_route(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            fine_route = model.model.predict_fine_route(inputs_dict["pixel_values"])
        self.assertEqual(fine_route.shape, (inputs_dict["pixel_values"].shape[0],))
        self.assertEqual(fine_route.dtype, torch.bool)

    def test_fine_route_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device).eval()
        batch_size = inputs_dict["pixel_values"].shape[0]
        fine_tokens = 4 * self.model_tester.num_image_tokens
        fine_input_ids = torch.full((batch_size, fine_tokens + 5), 4, dtype=torch.long, device=torch_device)
        fine_input_ids[:, :fine_tokens] = config.image_token_id
        tile_fine_mask = torch.ones(batch_size, dtype=torch.bool, device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=fine_input_ids, pixel_values=inputs_dict["pixel_values"], tile_fine_mask=tile_fine_mask
            )
        self.assertEqual(outputs.logits.shape, (batch_size, fine_input_ids.shape[1], config.text_config.vocab_size))
        coarse_input_ids = fine_input_ids[:, : self.model_tester.num_image_tokens + 5]
        with self.assertRaises(ValueError):
            model(input_ids=coarse_input_ids, pixel_values=inputs_dict["pixel_values"], tile_fine_mask=tile_fine_mask)

    def test_deepstack_taps_vision_layer_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingModel(config).to(torch_device).eval()
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
            connector = model.vision_model.connector
            with torch.no_grad():
                expected = connector.coarse.deepstack_mergers[slot](
                    connector.pixel_shuffle(
                        layer_outputs[vision_layer_idx].contiguous(), connector.coarse.scale_factor
                    )
                )
            torch.testing.assert_close(_concat_features(features.deepstack_features[slot]), expected.flatten(0, 1))

    def test_coarse_only_checkpoint(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.density_router_hidden_size = None
        config.vision_config.use_fine_route = False
        model = GraniteDoclingModel(config).to(torch_device).eval()
        self.assertIsNone(model.vision_model.connector.fine)
        self.assertIsNotNone(model.vision_model.connector.coarse)
        self.assertFalse([name for name in model.state_dict() if "_fine" in name])

        pixel_values = inputs_dict["pixel_values"]
        with torch.no_grad():
            coarse = model.get_image_features(pixel_values)
        self.assertEqual(
            _concat_features(coarse.pooler_output).shape,
            (pixel_values.shape[0] * self.model_tester.num_image_tokens, config.text_config.hidden_size),
        )
        with self.assertRaises(ValueError):
            model.get_image_features(pixel_values, tile_fine_mask=torch.ones(pixel_values.shape[0], dtype=torch.bool))
        # The density router selects the fine path, so it cannot be built without it
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "use_fine_route=True"):
            GraniteDoclingConfig(
                density_router_hidden_size=8,
                vision_config={"use_fine_route": False},
            )

    def test_beam_search_expands_per_sample_image_features(self):
        # Generation repeats each sample's image features once per beam, which needs one feature tensor per sample.
        # The two samples have different tile counts, so the second one carries an all-zero padding tile.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteDoclingForConditionalGeneration(config).to(torch_device).eval()
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
class GraniteDoclingIntegrationTest(unittest.TestCase):
    # Local copy of the checkpoint until the renamed checkpoint is on the Hub (docling-project/granite-for-docling-500m)
    model_id = "/proj/docling-vision/models/GraniteDocling500m-2026-10-19"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Page 1 of Auer et al., "Docling Technical Report", arXiv:2408.09869, stored next to the checkpoint
        self.image = load_image(f"{self.model_id}/docling_technical_report_p1.png")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<doclang>"}]}]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_integration(self):
        expected_num_tiles = 13
        # Decode keeps the DocLang tags. skip_special_tokens=False is what a parser consumes.
        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): '<doclang><page_header><layer value="furniture"/><location value="15"/><location value="105"/><location value="30"/><location value="359"/>arXiv:2408.09869v5  [cs.CL]  9 Dec 2024</page_header><picture><label value="logo"/><location value="220"/><location value="50"/><location value="290"/><location value="105"/></picture><heading level="2"><location value="178"/><location value="145"/><location value="334"/><location value="156"/>Docling Technical Report</heading><text><location value="237"/><location value="181"/><location value="275"/><location value="187"/>Version 1.0</text>'
            }
        )
        model = GraniteDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt").to(model.device)
        self.assertEqual(inputs["pixel_values"].shape[1], expected_num_tiles)
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), expected_num_tiles * 64)

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        self.assertEqual(generated_text, expected_output.get_expectation())

    @slow
    def test_model_integration_fine_route(self):
        expected_num_tiles = 13
        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): '<doclang><page_header><layer value="furniture"/><location value="15"/><location value="105"/><location value="30"/><location value="359"/>arXiv:2408.09869v5 [cs.CL] 9 Dec 2024</page_header><picture><label value="logo"/><location value="220"/><location value="50"/><location value="290"/><location value="105"/></picture><heading level="2"><location value="178"/><location value="145"/><location value="334"/><location value="156"/>Docling Technical Report</heading><text><location value="237"/><location value="181"/><location value="275"/><location value="187"/>Version 1.0</text><text><location value="95"/>'
            }
        )
        model = GraniteDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt", fine_route=True).to(
            model.device
        )
        self.assertEqual((inputs["input_ids"] == self.processor.image_token_id).sum().item(), expected_num_tiles * 256)
        self.assertTrue(inputs["tile_fine_mask"].all())

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        self.assertEqual(generated_text, expected_output.get_expectation())

    def _batch_coarse_and_fine(self, coarse, fine):
        """Batch one coarse page and one fine page. Prompts are left-padded, tiles are padded at the end."""
        pad_id = self.processor.tokenizer.pad_token_id
        max_len = max(coarse["input_ids"].shape[1], fine["input_ids"].shape[1])
        max_tiles = max(coarse["pixel_values"].shape[1], fine["pixel_values"].shape[1])

        def left_pad(values, value):
            extra = max_len - values.shape[1]
            return torch.cat([values.new_full((values.shape[0], extra), value), values], dim=1)

        def pad_tiles(values):
            extra = max_tiles - values.shape[1]
            return torch.cat([values, values.new_zeros(values.shape[0], extra, *values.shape[2:])], dim=1)

        return {
            "input_ids": torch.cat([left_pad(coarse["input_ids"], pad_id), left_pad(fine["input_ids"], pad_id)]),
            "attention_mask": torch.cat([left_pad(coarse["attention_mask"], 0), left_pad(fine["attention_mask"], 0)]),
            "pixel_values": torch.cat([pad_tiles(coarse["pixel_values"]), pad_tiles(fine["pixel_values"])]),
            "pixel_attention_mask": torch.cat(
                [pad_tiles(coarse["pixel_attention_mask"]), pad_tiles(fine["pixel_attention_mask"])]
            ),
            "tile_fine_mask": torch.tensor([False, True]),
        }

    @slow
    def test_model_integration_batched(self):
        expected_output = Expectations(
            {
                ("cuda", None): [
                    '<doclang><page_header><layer value="furniture"/><location value="15"/><location value="105"/><location value="30"/><location value="359"/>arXiv:2408.09869v5  [cs.CL]  9 Dec ',
                    '<doclang><picture><label value="logo"/><location value="221"/><location value="151"/><location value="290"/><location value="317"/></picture><heading level="2"><location value="177"/><location value="432"/><location value="334"/><location value="467"/>Docling Technical Report</heading><page_footer><layer value="furniture"/><location value="13"/><location value="417"/><location value="29"/><location value="496"/>2024</page_footer></doclang>',
                ]
            }
        )
        model = GraniteDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        # The top third of the report page is a wide strip, so it takes a different tile grid from the full page.
        # The full page stays coarse and the strip takes the fine route.
        strip = self.image.crop((0, 0, self.image.width, self.image.height // 3))
        coarse = self.processor(text=self.prompt, images=self.image, return_tensors="pt")
        fine = self.processor(text=self.prompt, images=strip, return_tensors="pt", fine_route=True)
        self.assertNotEqual(coarse["pixel_values"].shape[1], fine["pixel_values"].shape[1])
        self.assertNotIn("tile_fine_mask", coarse)
        inputs = {name: value.to(model.device) for name, value in self._batch_coarse_and_fine(coarse, fine).items()}

        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_texts = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        self.assertEqual(generated_texts, expected_output.get_expectation())

        # Each page of the batch decodes as it does alone
        for single, generated_text in zip((coarse, fine), generated_texts):
            single = single.to(model.device)
            alone = model.generate(**single, max_new_tokens=32, do_sample=False)
            alone_text = self.processor.decode(alone[0, single["input_ids"].shape[1] :], skip_special_tokens=False)
            self.assertEqual(alone_text, generated_text)

    @slow
    def test_predict_fine_route_integration(self):
        model = GraniteDoclingForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16).to(
            torch_device
        )
        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt").to(model.device)
        route = model.model.predict_fine_route(inputs["pixel_values"])
        self.assertEqual(tuple(route.shape), (1,))
        self.assertEqual(route.dtype, torch.bool)
