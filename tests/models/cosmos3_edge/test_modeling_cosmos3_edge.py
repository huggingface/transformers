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
"""Focused tests for the native Cosmos3 Edge reasoner implementation."""

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
    Cosmos3EdgeVisionModel,
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
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "hidden_act": "relu2",
            "rms_norm_eps": 1e-5,
            # The real checkpoint uses [24, 20, 20] for head_dim=128. Keep
            # the same 2 * sum(section) == head_dim relation in this tiny config.
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 100_000_000,
                "mrope_section": [2, 1, 1],
            },
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

    def test_checkpoint_topology_uses_28_dense_decoder_blocks(self):
        config = Cosmos3EdgeTextConfig()

        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.hidden_act, "relu2")
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(
            config.rope_parameters,
            {"rope_type": "default", "rope_theta": 100_000_000.0, "mrope_section": [24, 20, 20]},
        )
        config_dict = config.to_dict()
        for deprecated_field in (
            "hybrid_override_pattern",
            "layers_block_type",
            "layer_norm_epsilon",
            "mlp_hidden_act",
            "num_logits_to_keep",
            "rope_theta",
        ):
            self.assertNotIn(deprecated_field, config_dict)

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

    def test_composite_config_round_trips_native_component_configs(self):
        config = _tiny_config()

        self.assertEqual(config.model_type, "cosmos3_edge")
        self.assertIsInstance(config.text_config, Cosmos3EdgeTextConfig)
        self.assertIsInstance(config.vision_config, Cosmos3EdgeVisionConfig)
        self.assertIsInstance(config.projector_config, Cosmos3EdgeProjectorConfig)
        self.assertEqual(config.text_config.num_hidden_layers, 2)
        self.assertEqual(config.text_config.rope_parameters["rope_theta"], 100_000_000)
        self.assertEqual(config.text_config.rope_parameters["mrope_section"], [2, 1, 1])

        reloaded = Cosmos3EdgeConfig.from_dict(config.to_dict())
        self.assertEqual(reloaded.to_dict(), config.to_dict())

    def test_projector_uses_the_canonical_intermediate_size_field(self):
        config = Cosmos3EdgeProjectorConfig(merger_intermediate_size=73)

        self.assertEqual(config.merger_intermediate_size, 73)


@require_torch
class Cosmos3EdgeModelTest(unittest.TestCase):
    all_model_classes = (
        (
            Cosmos3EdgeTextModel,
            Cosmos3EdgeVisionModel,
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

    def test_auto_model_resolves_the_native_vision_model(self):
        from transformers import AutoModel

        model = AutoModel.from_config(_tiny_config().vision_config)

        self.assertIsInstance(model, Cosmos3EdgeVisionModel)

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

        self.assertEqual(len(model.model.language_model.layers), 2)
        self.assertTrue(hasattr(model.model.language_model.layers[0].self_attn, "q_proj"))
        self.assertTrue(hasattr(model.model.language_model.layers[0].mlp, "fc1"))

        expected_keys = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.0.mlp.fc1.weight",
            "model.language_model.layers.0.mlp.fc2.weight",
            "model.language_model.norm.weight",
            "model.projector.norm.weight",
            "model.projector.linear_fc1.weight",
            "model.projector.linear_fc2.weight",
            "lm_head.weight",
        }
        self.assertTrue(expected_keys.issubset(state_dict), expected_keys - set(state_dict))
        self.assertNotIn("model.language_model.layers.0.mlp.fc1.bias", state_dict)
        self.assertNotIn("model.language_model.layers.0.mlp.fc2.bias", state_dict)

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
                mm_token_type_ids=torch.tensor([[0, 1, 0]], dtype=torch.long, device=torch_device),
                use_cache=False,
            )
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=torch.tensor([[0, 1, 0]], dtype=torch.long, device=torch_device),
                do_sample=False,
                max_new_tokens=1,
            )

        self.assertEqual(tuple(outputs.logits.shape), (1, 3, 97))
        self.assertIsNotNone(outputs.rope_deltas)
        self.assertEqual(tuple(generated.shape), (1, 4))

    def test_multimodal_forward_requires_token_type_ids(self):
        model = Cosmos3EdgeForConditionalGeneration(_tiny_config()).to(torch_device).eval()
        input_ids = torch.tensor([[5, 3, 6]], dtype=torch.long, device=torch_device)
        pixel_values = torch.zeros((4, 12), dtype=torch.float32, device=torch_device)
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long, device=torch_device)

        with self.assertRaisesRegex(ValueError, "mm_token_type_ids"):
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )

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
                mm_token_type_ids=torch.tensor([[0, 0, 2, 0, 0, 0, 2, 0]], dtype=torch.long, device=torch_device),
                use_cache=False,
            )
            generated = model.generate(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=torch.tensor([[0, 0, 2, 0, 0, 0, 2, 0]], dtype=torch.long, device=torch_device),
                do_sample=False,
                num_beams=2,
                max_new_tokens=1,
            )

        self.assertEqual(tuple(outputs.logits.shape), (1, 8, 97))
        self.assertEqual(tuple(generated.shape), (1, 9))

    def test_mrope_positions_use_merged_image_grid(self):
        model = Cosmos3EdgeModel(_tiny_config()).to(torch_device)
        input_ids = torch.tensor([[5, 3, 3, 3, 3, 6]], dtype=torch.long, device=torch_device)
        mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.long, device=torch_device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=torch_device)

        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
        )

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
        """The composite checkpoint and native model share one 28-block text topology."""

        from transformers.conversion_mapping import get_model_conversion_mapping

        model = Cosmos3EdgeForConditionalGeneration(_tiny_config())
        conversions = get_model_conversion_mapping(model, add_legacy=False)

        def rename(source_key):
            target_key = source_key
            matched = False
            for conversion in conversions:
                target_key, matched_pattern = conversion.rename_source_key(target_key)
                if matched_pattern is not None:
                    matched = True
            if not matched:
                self.fail(f"No native conversion mapping found for checkpoint key: {source_key}")
            return target_key

        expected = {
            "embed_tokens.weight": "model.language_model.embed_tokens.weight",
            "norm.weight": "model.language_model.norm.weight",
            "layers.0.input_layernorm.weight": "model.language_model.layers.0.input_layernorm.weight",
            "layers.0.self_attn.to_q.weight": "model.language_model.layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.to_k.weight": "model.language_model.layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.to_v.weight": "model.language_model.layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.to_out.weight": "model.language_model.layers.0.self_attn.o_proj.weight",
            "layers.0.post_attention_layernorm.weight": "model.language_model.layers.0.post_attention_layernorm.weight",
            "layers.0.mlp.up_proj.weight": "model.language_model.layers.0.mlp.fc1.weight",
            "layers.0.mlp.down_proj.weight": "model.language_model.layers.0.mlp.fc2.weight",
            "layers.27.input_layernorm.weight": "model.language_model.layers.27.input_layernorm.weight",
            "layers.27.mlp.down_proj.weight": "model.language_model.layers.27.mlp.fc2.weight",
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
            if key == "model.language_model.embed_tokens.weight":
                composite_state_dict["embed_tokens.weight"] = value
            elif key == "model.language_model.norm.weight":
                composite_state_dict["norm.weight"] = value
            elif key.startswith("model.language_model.layers."):
                key_parts = key.split(".")
                layer_idx = int(key_parts[3])
                layer_name, suffix = ".".join(key_parts[4:-1]), key_parts[-1]
                source_layer_names = {
                    "input_layernorm": "input_layernorm",
                    "self_attn.q_proj": "self_attn.to_q",
                    "self_attn.k_proj": "self_attn.to_k",
                    "self_attn.v_proj": "self_attn.to_v",
                    "self_attn.o_proj": "self_attn.to_out",
                    "post_attention_layernorm": "post_attention_layernorm",
                    "mlp.fc1": "mlp.up_proj",
                    "mlp.fc2": "mlp.down_proj",
                }
                composite_state_dict[f"layers.{layer_idx}.{source_layer_names[layer_name]}.{suffix}"] = value
            else:
                # The vision encoder, projector, and LM head already match the native module tree.
                composite_state_dict[key] = value

        # The unified checkpoint contains this generator-only branch. The reasoner loader deliberately excludes it
        # after the text-key conversion has added the `model.language_model` prefix.
        composite_state_dict["layers.0.mlp_moe_gen.up_proj.weight"] = torch.zeros(1)
        composite_state_dict["layers.0.self_attn.add_q_proj.weight"] = torch.zeros(1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            reference_model.config.save_pretrained(temporary_directory)
            safe_save_file(
                composite_state_dict,
                os.path.join(temporary_directory, "model.safetensors"),
                metadata={"format": "pt"},
            )
            loaded_model, loading_info = Cosmos3EdgeForConditionalGeneration.from_pretrained(
                temporary_directory, output_loading_info=True
            )
            loaded_model = loaded_model.eval()

        for key, value in reference_model.state_dict().items():
            torch.testing.assert_close(value, loaded_model.state_dict()[key])
        self.assertEqual(loading_info["unexpected_keys"], set())
