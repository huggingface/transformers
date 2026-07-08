# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Focused tests for the native Cosmos3 Edge reasoner implementation.

Cosmos3 Edge's language checkpoint is structurally different from a conventional
Qwen-style decoder: each of the 28 physical Diffusers blocks represents two
Nemotron-H residual modules (attention followed by MLP).  These tests keep the
native topology and the composite-checkpoint contract explicit.
"""

import os
import tempfile
import unittest

from transformers import (
    Cosmos3EdgeConfig,
    Cosmos3EdgeForConditionalGeneration,
    Cosmos3EdgeModel,
    Cosmos3EdgeProjectorConfig,
    Cosmos3EdgeTextConfig,
    Cosmos3EdgeTextModel,
    Cosmos3EdgeVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch
    from safetensors.torch import save_file as safe_save_file


def _tiny_config():
    """Build a CPU-friendly config with the same module topology as the checkpoint."""

    return Cosmos3EdgeConfig(
        text_config={
            "vocab_size": 97,
            "hidden_size": 32,
            "intermediate_size": 64,
            "layers_block_type": ["full_attention", "mlp"] * 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "layer_norm_epsilon": 1e-5,
            "rope_theta": 100_000_000,
            # The real checkpoint uses [24, 20, 20] for head_dim=128. Keep
            # the same 2 * sum(section) == head_dim relation in this tiny config.
            "rope_parameters": {"mrope_section": [2, 1, 1]},
            "use_cache": True,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "patch_size": 2,
            "num_patches": 16,
        },
        projector_config={
            "input_hidden_size": 32,
            "spatial_merge_size": 2,
            "merger_intermediate_size": 64,
            "out_hidden_size": 32,
            "use_postshuffle_norm": False,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        pad_token_id=0,
    )


class Cosmos3EdgeConfigTest(unittest.TestCase):
    def test_auto_config_resolves_the_native_model_type(self):
        from transformers import AutoConfig

        self.assertIsInstance(AutoConfig.for_model("cosmos3_edge"), Cosmos3EdgeConfig)

    def test_checkpoint_topology_is_explicit_dense_attention_then_mlp(self):
        config = Cosmos3EdgeTextConfig()

        self.assertEqual(config.layers_block_type, ["full_attention", "mlp"] * 28)
        self.assertEqual(config.num_hidden_layers, 56)
        self.assertNotIn("linear_attention", config.layers_block_type)
        self.assertNotIn("moe", config.layers_block_type)

    def test_default_component_widths_match_the_reasoner_checkpoint(self):
        config = Cosmos3EdgeConfig()

        self.assertEqual(config.text_config.hidden_size, 2048)
        self.assertEqual(config.text_config.intermediate_size, 9216)
        self.assertEqual(config.vision_config.hidden_size, 1152)
        self.assertEqual(config.vision_config.num_hidden_layers, 27)
        self.assertEqual(config.projector_config.input_hidden_size, 1152)
        self.assertEqual(config.projector_config.merger_intermediate_size, 11520)
        self.assertEqual(config.projector_config.out_hidden_size, 2048)
        self.assertEqual(config.projector_config.spatial_merge_size, 2)

    def test_legacy_checkpoint_pattern_becomes_native_layer_types(self):
        config = Cosmos3EdgeTextConfig(hybrid_override_pattern="*-" * 28)

        self.assertEqual(config.layers_block_type, ["full_attention", "mlp"] * 28)
        self.assertEqual(config.num_hidden_layers, 56)

    def test_composite_config_round_trips_native_component_configs(self):
        config = _tiny_config()

        self.assertEqual(config.model_type, "cosmos3_edge")
        self.assertIsInstance(config.text_config, Cosmos3EdgeTextConfig)
        self.assertIsInstance(config.vision_config, Cosmos3EdgeVisionConfig)
        self.assertIsInstance(config.projector_config, Cosmos3EdgeProjectorConfig)
        self.assertEqual(config.text_config.layers_block_type, ["full_attention", "mlp"] * 2)
        self.assertEqual(config.text_config.rope_theta, 100_000_000)
        self.assertEqual(config.text_config.rope_parameters["mrope_section"], [2, 1, 1])

        reloaded = Cosmos3EdgeConfig.from_dict(config.to_dict())
        self.assertEqual(reloaded.to_dict(), config.to_dict())

    def test_legacy_config_fields_are_normalized_or_ignored(self):
        text_config = Cosmos3EdgeTextConfig(hidden_dropout=0.2, mrope_section=[2, 1, 1])
        vision_config = Cosmos3EdgeVisionConfig()

        self.assertEqual(text_config.rope_parameters["mrope_section"], [2, 1, 1])
        self.assertNotIn("hidden_dropout", text_config.to_dict())
        self.assertNotIn("mrope_section", text_config.to_dict())
        self.assertNotIn("vision_use_head", vision_config.to_dict())

    def test_projector_accepts_the_checkpoint_legacy_field_name(self):
        config = Cosmos3EdgeProjectorConfig(merger_intermedia=73)

        self.assertEqual(config.merger_intermediate_size, 73)


@require_torch
class Cosmos3EdgeModelTest(unittest.TestCase):
    all_model_classes = (
        (
            Cosmos3EdgeTextModel,
            Cosmos3EdgeModel,
            Cosmos3EdgeForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    def test_auto_model_resolves_the_native_conditional_generation_model(self):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_config(_tiny_config())

        self.assertIsInstance(model, Cosmos3EdgeForConditionalGeneration)

    def test_auto_model_resolves_the_native_text_model(self):
        from transformers import AutoModel

        model = AutoModel.from_config(_tiny_config().text_config)

        self.assertIsInstance(model, Cosmos3EdgeTextModel)

    def test_text_model_forward(self):
        config = _tiny_config().text_config
        model = Cosmos3EdgeTextModel(config).to(torch_device).eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=prompt, use_cache=True)

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (1, 3, 32))
        self.assertIsNotNone(outputs.past_key_values)

    def test_native_module_tree_and_checkpoint_key_layout(self):
        model = Cosmos3EdgeForConditionalGeneration(_tiny_config())
        state_dict = model.state_dict()

        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(hasattr(model.model, "visual"))
        self.assertTrue(hasattr(model.model, "projector"))
        self.assertTrue(hasattr(model.model, "language_model"))
        self.assertTrue(hasattr(model, "lm_head"))

        self.assertEqual(len(model.model.language_model.layers), 4)
        self.assertTrue(hasattr(model.model.language_model.layers[0].mixer, "q_proj"))
        self.assertTrue(hasattr(model.model.language_model.layers[1].mixer, "up_proj"))

        expected_keys = {
            "model.language_model.embeddings.weight",
            "model.language_model.layers.0.norm.weight",
            "model.language_model.layers.0.mixer.q_proj.weight",
            "model.language_model.layers.0.mixer.k_proj.weight",
            "model.language_model.layers.0.mixer.v_proj.weight",
            "model.language_model.layers.0.mixer.o_proj.weight",
            "model.language_model.layers.1.norm.weight",
            "model.language_model.layers.1.mixer.up_proj.weight",
            "model.language_model.layers.1.mixer.down_proj.weight",
            "model.language_model.norm_f.weight",
            "model.projector.norm.weight",
            "model.projector.linear_fc1.weight",
            "model.projector.linear_fc2.weight",
            "lm_head.weight",
        }
        self.assertTrue(expected_keys.issubset(state_dict), expected_keys - set(state_dict))

        # The projector sees four spatially grouped visual patches at once.
        self.assertEqual(tuple(state_dict["model.projector.linear_fc1.weight"].shape), (64, 128))
        self.assertEqual(tuple(state_dict["model.projector.linear_fc2.weight"].shape), (32, 64))

    def test_text_forward_cache_and_generate(self):
        model = Cosmos3EdgeForConditionalGeneration(_tiny_config()).to(torch_device).eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=prompt, use_cache=True)
            next_outputs = model(
                input_ids=torch.tensor([[4]], dtype=torch.long, device=torch_device),
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
            generated = model.generate(prompt, do_sample=False, max_new_tokens=1)

        self.assertEqual(tuple(outputs.logits.shape), (1, 3, 97))
        self.assertIsNotNone(outputs.past_key_values)
        self.assertEqual(tuple(next_outputs.logits.shape), (1, 1, 97))
        self.assertEqual(tuple(generated.shape), (1, 4))

    def test_packed_image_forward_replaces_one_projected_placeholder(self):
        model = Cosmos3EdgeForConditionalGeneration(_tiny_config()).to(torch_device).eval()
        # A 2 x 2 raw patch grid becomes exactly one projected language token.
        input_ids = torch.tensor([[5, 3, 6]], dtype=torch.long, device=torch_device)
        pixel_values = torch.zeros((4, 12), dtype=torch.float32, device=torch_device)
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )

        self.assertEqual(tuple(outputs.logits.shape), (1, 3, 97))

    def test_packed_video_forward_uses_one_vision_span_per_frame(self):
        model = Cosmos3EdgeForConditionalGeneration(_tiny_config()).to(torch_device).eval()
        # Two timestamped 2 x 2 frames each become one projected video token.
        input_ids = torch.tensor([[7, 5, 4, 6, 8, 5, 4, 6]], dtype=torch.long, device=torch_device)
        pixel_values_videos = torch.zeros((8, 12), dtype=torch.float32, device=torch_device)
        video_grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                use_cache=False,
            )

        self.assertEqual(tuple(outputs.logits.shape), (1, 8, 97))

    def test_patch_merging_preserves_checkpoint_2x2_row_major_order(self):
        model = Cosmos3EdgeModel(_tiny_config()).to(torch_device)
        image_embeds = torch.arange(8, dtype=torch.float32, device=torch_device).reshape(8, 1)
        image_grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long, device=torch_device)

        merged_embeds, merged_grid_thw = model._patch_merge(image_embeds, image_grid_thw)

        # For a 2 x 4 grid, the first merged token must contain the values at
        # (0, 0), (0, 1), (1, 0), and (1, 1), in that order. This is the
        # checkpoint's `h1 w1 c` flattening convention.
        expected_embeds = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7]], dtype=torch.float32, device=torch_device)
        torch.testing.assert_close(merged_embeds, expected_embeds)
        self.assertEqual(merged_grid_thw.tolist(), [[1, 1, 2]])

    def test_mrope_positions_use_merged_image_grid(self):
        model = Cosmos3EdgeModel(_tiny_config()).to(torch_device)
        input_ids = torch.tensor([[5, 3, 3, 3, 3, 6]], dtype=torch.long, device=torch_device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=torch_device)

        position_ids, rope_deltas = model.get_rope_index(input_ids=input_ids, image_grid_thw=image_grid_thw)

        expected_position_ids = torch.tensor(
            [
                [[0, 1, 1, 1, 1, 3]],
                [[0, 1, 1, 2, 2, 3]],
                [[0, 1, 2, 1, 2, 3]],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        torch.testing.assert_close(position_ids, expected_position_ids)
        self.assertEqual(rope_deltas.tolist(), [[-2]])

    def test_composite_diffusers_checkpoint_key_mapping(self):
        """The composite repo stores 28 physical blocks, while Edge exposes 56 modules."""

        from transformers.conversion_mapping import get_model_conversion_mapping

        model = Cosmos3EdgeForConditionalGeneration(_tiny_config())
        conversions = get_model_conversion_mapping(model, add_legacy=False)

        def rename(source_key):
            for conversion in conversions:
                target_key, matched_pattern = conversion.rename_source_key(source_key)
                if matched_pattern is not None:
                    return target_key
            self.fail(f"No native conversion mapping found for checkpoint key: {source_key}")

        expected = {
            "embed_tokens.weight": "model.language_model.embeddings.weight",
            "norm.weight": "model.language_model.norm_f.weight",
            "layers.0.input_layernorm.weight": "model.language_model.layers.0.norm.weight",
            "layers.0.self_attn.to_q.weight": "model.language_model.layers.0.mixer.q_proj.weight",
            "layers.0.self_attn.to_k.weight": "model.language_model.layers.0.mixer.k_proj.weight",
            "layers.0.self_attn.to_v.weight": "model.language_model.layers.0.mixer.v_proj.weight",
            "layers.0.self_attn.to_out.weight": "model.language_model.layers.0.mixer.o_proj.weight",
            "layers.0.post_attention_layernorm.weight": "model.language_model.layers.1.norm.weight",
            "layers.0.mlp.up_proj.weight": "model.language_model.layers.1.mixer.up_proj.weight",
            "layers.0.mlp.down_proj.weight": "model.language_model.layers.1.mixer.down_proj.weight",
            "layers.27.input_layernorm.weight": "model.language_model.layers.54.norm.weight",
            "layers.27.mlp.down_proj.weight": "model.language_model.layers.55.mixer.down_proj.weight",
        }
        for source_key, target_key in expected.items():
            with self.subTest(source_key=source_key):
                self.assertEqual(rename(source_key), target_key)

    def test_composite_diffusers_checkpoint_loads_with_native_mapping(self):
        """Exercise the loader with the mixed native/composite key layout used on the Hub."""
        torch.manual_seed(0)
        reference_model = Cosmos3EdgeForConditionalGeneration(_tiny_config()).eval()
        composite_state_dict = {}

        for key, value in reference_model.state_dict().items():
            if key == "model.language_model.embeddings.weight":
                composite_state_dict["embed_tokens.weight"] = value
            elif key == "model.language_model.norm_f.weight":
                composite_state_dict["norm.weight"] = value
            elif key.startswith("model.language_model.layers."):
                key_parts = key.split(".")
                layer_idx = int(key_parts[3])
                if key_parts[4] == "norm":
                    layer_name, suffix = "norm", ".".join(key_parts[5:])
                else:
                    layer_name, suffix = key_parts[5], ".".join(key_parts[6:])

                physical_layer_idx = layer_idx // 2
                if layer_idx % 2 == 0:
                    source_layer_names = {
                        "norm": "input_layernorm",
                        "q_proj": "self_attn.to_q",
                        "k_proj": "self_attn.to_k",
                        "v_proj": "self_attn.to_v",
                        "o_proj": "self_attn.to_out",
                    }
                else:
                    source_layer_names = {
                        "norm": "post_attention_layernorm",
                        "up_proj": "mlp.up_proj",
                        "down_proj": "mlp.down_proj",
                    }
                composite_state_dict[f"layers.{physical_layer_idx}.{source_layer_names[layer_name]}.{suffix}"] = value
            else:
                # The vision encoder, projector, and LM head already match the native module tree.
                composite_state_dict[key] = value

        with tempfile.TemporaryDirectory() as temporary_directory:
            reference_model.config.save_pretrained(temporary_directory)
            safe_save_file(
                composite_state_dict,
                os.path.join(temporary_directory, "model.safetensors"),
                metadata={"format": "pt"},
            )
            loaded_model = Cosmos3EdgeForConditionalGeneration.from_pretrained(temporary_directory).eval()

        for key, value in reference_model.state_dict().items():
            torch.testing.assert_close(value, loaded_model.state_dict()[key])
