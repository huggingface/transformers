# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HY4-Flash model."""

import copy
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import (
    HYV4FlashConfig,
    HYV4FlashForConditionalGeneration,
    HYV4FlashModel,
    HYV4FlashTextConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class HYV4FlashVisionText2TextModelTester(VLMModelTester):
    config_class = HYV4FlashConfig
    base_model_class = HYV4FlashModel
    conditional_generation_class = HYV4FlashForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "vocab_size": 256,
        # The fewest layers that still carry one of each: two indexed (the second holding the Engram
        # memory) and one sliding.
        "num_hidden_layers": 3,
        "hidden_size": 48,
        "intermediate_size": 96,
        "moe_intermediate_size": 32,
        "routed_expert_hidden_size": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hc_mult": 2,
        # One head shape for the sliding layers and another for the indexed ones, as in the checkpoint.
        "swa_num_attention_heads": 8,
        "swa_qk_nope_head_dim": 8,
        "swa_qk_rope_head_dim": 8,
        "swa_v_head_dim": 8,
        "sliding_window": 4,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "index_topk": 16,
        "index_kpool": 4,
        "engram_layer_ids": [2],
        "engram_max_ngram_size": 3,
        "engram_n_head_per_ngram": 2,
        "engram_n_embed_per_ngram": 16,
        "engram_layer_head_vocab_sizes": {"2": [[101, 103], [107, 109]]},
        "engram_pad_id": 0,
        # Shorter than the vocabulary, as in the released checkpoint, so random ids also reach past the map
        "engram_lookup_table_size": 250,
        "rope_parameters": {
            "indexed_attention": {"rope_type": "default", "rope_theta": 1e7, "mrope_section": [2, 1, 1]},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4, "mrope_section": [2, 1, 1]},
        },
        "video_token_id": 4,
        "image_size": 16,
        "spatial_merge_size": 2,
        "num_image_tokens": 6,
        "vision_hidden_size": 32,
        "vision_intermediate_size": 64,
        "vision_num_attention_heads": 2,
        "vision_num_key_value_heads": None,
        "vision_num_hidden_layers": 3,
        "vision_hidden_act": "gelu",
        "vision_max_image_size": 64,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values.setdefault("vision_out_hidden_size", values["hidden_size"])
        values["grid_hw"] = values["image_size"] // values["patch_size"]
        values["num_image_patches"] = values["grid_hw"] ** 2
        return values

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id}

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size * self.num_image_patches, self.num_channels * self.patch_size**2]).to(
            torch_device
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_hw, self.grid_hw]] * batch_size, device=torch_device),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }

    def prepare_multimodal_inputs(self, image_grid=None, video_grid=None):
        """Text, then optionally one image and one video, each followed by a text token."""
        input_ids, token_types = [10, 11, 12], [0, 0, 0]
        inputs = {}
        for grid, token_id, modality, pixel_key, grid_key in (
            (image_grid, self.image_token_id, 1, "pixel_values", "image_grid_thw"),
            (video_grid, self.video_token_id, 2, "pixel_values_videos", "video_grid_thw"),
        ):
            if grid is None:
                continue
            frames, height, width = grid
            # One newline token per merged row
            num_tokens = frames * (height // self.spatial_merge_size) * (width // self.spatial_merge_size + 1)
            input_ids.extend([token_id] * num_tokens + [13])
            token_types.extend([modality] * num_tokens + [0])
            inputs[pixel_key] = floats_tensor([frames * height * width, self.num_channels * self.patch_size**2]).to(
                torch_device
            )
            inputs[grid_key] = torch.tensor([grid], device=torch_device)
        inputs["input_ids"] = torch.tensor([input_ids], device=torch_device)
        inputs["mm_token_type_ids"] = torch.tensor([token_types], device=torch_device)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs


@require_torch
class HYV4FlashModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HYV4FlashVisionText2TextModelTester
    # Each token is routed to a subset of the experts, so not every expert receives a gradient.
    test_all_params_have_gradient = False
    # Packed vision features have a leading singleton dimension and one newline token per merged row
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    def _layer_head_counts(self, config):
        """Heads per layer: resolved from `per_layer_config`, where sliding layers carry a wider count."""
        text_config = config.get_text_config()
        return [
            text_config.per_layer_config[layer_idx].num_attention_heads
            for layer_idx in range(text_config.num_hidden_layers)
        ]

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_cache
    ):
        """Override for the two head counts, and for the sliding layers only keeping their window"""
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), output_length - prompt_length)
        text_config = config.get_text_config()
        use_cache = decoder_cache is not None
        head_counts = self._layer_head_counts(config)

        for generated_length, iter_attentions in enumerate(attentions):
            model_input_length = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            query_length = prompt_length + generated_length
            self.assertEqual(len(iter_attentions), len(head_counts))
            for attention, num_heads, layer_type in zip(iter_attentions, head_counts, text_config.layer_types):
                self.assertEqual(attention.shape[:3], (batch_size, num_heads, model_input_length))
                # How many keys a layer sees depends on its cache: a sliding layer is bounded by its
                # window, and a static cache is preallocated beyond the tokens generated so far.
                if layer_type == "sliding_attention" and use_cache and generated_length > 0:
                    self.assertLessEqual(attention.shape[3], text_config.sliding_window)
                else:
                    self.assertGreaterEqual(attention.shape[3], query_length)

    def _get_attention_shape(self, batch_size, seq_length, config):
        """MLA caches the compressed latent and the shared rotary key, not one key/value per head"""
        text_config = config.get_text_config()
        key_shape = (batch_size, 1, seq_length, text_config.kv_lora_rank)
        value_shape = (batch_size, 1, seq_length, text_config.qk_rope_head_dim)
        return key_shape, value_shape

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        """Override because the sliding layers keep at most their window, not the whole sequence"""
        text_config = config.get_text_config()
        for layer, layer_type in zip(cache.layers, text_config.layer_types):
            cached_length = layer.keys.shape[2]
            if layer_type == "sliding_attention":
                self.assertLessEqual(cached_length, text_config.sliding_window)
            else:
                self.assertEqual(cached_length, seq_length)
            key_shape, value_shape = self._get_attention_shape(batch_size, cached_length, text_config)
            self._check_attention_shapes(layer, cached_length, key_shape, value_shape)

    @unittest.skip(
        "The compressed indexer pools fixed-size groups of token slots, so left padding moves the group "
        "boundaries and changes which tokens a query recalls"
    )
    def test_left_padding_compatibility(self):
        pass

    def test_attention_outputs(self):
        """Override to account for the two head counts of the sliding and indexed layers"""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        head_counts = self._layer_head_counts(config)
        seq_len = inputs["input_ids"].shape[1]

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                attentions = model(**inputs, output_attentions=True).attentions

            self.assertEqual(len(attentions), len(head_counts))
            for attention, num_heads in zip(attentions, head_counts):
                self.assertEqual(attention.shape[-3:], (num_heads, seq_len, seq_len))

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """Override to account for the difference in MHC and the final state shapes"""
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            model_input_length = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            # We have raw MHC shapes until the final one which is collapsed
            mhc_shape = (batch_size, model_input_length, config.hc_mult, config.hidden_size)
            final_shape = (batch_size, model_input_length, config.hidden_size)
            expected_shapes = [mhc_shape] * (len(iter_hidden_states) - 1) + [final_shape]
            self.assertListEqual([state.shape for state in iter_hidden_states], expected_shapes)

    def test_hidden_states_output(self):
        """Override to account for the difference in MHC and the final state shapes"""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        text_config = config.text_config
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states

            self.assertEqual(len(hidden_states), text_config.num_hidden_layers + 1)
            batch_size, seq_len = inputs["input_ids"].shape
            for state in hidden_states[:-1]:
                self.assertEqual(state.shape, (batch_size, seq_len, text_config.hc_mult, text_config.hidden_size))
            self.assertEqual(hidden_states[-1].shape, (batch_size, seq_len, text_config.hidden_size))

    @unittest.skip("The text attention (sinks + indexer) is eager-only, so the model cannot switch to SDPA")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("Packed vision attention returns no attention matrix")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("Packed vision attention returns no attention matrix")
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip("The Engram memories read the token ids, which `inputs_embeds` alone cannot carry")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        # Source prefixes describe the complete checkpoint, including the model wrapper
        return super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model)

    @parameterized.expand([((1, 4, 6), None), (None, (2, 4, 4)), ((1, 4, 6), (2, 4, 4))])
    def test_image_video_and_mixed_forward(self, image_grid, video_grid):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs(image_grid, video_grid)
        model = HYV4FlashForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (*inputs["input_ids"].shape, config.text_config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_checkpoint_names_reach_every_renamed_module(self):
        """Walk the released checkpoint's naming through the conversion mapping.

        A save/load round trip cannot catch a rename that never fires: it is reversed symmetrically,
        so the key comes back unchanged either way. These are the names the released checkpoint
        actually uses, one per renaming rule.
        """
        from transformers.core_model_loading import revert_weight_conversion

        # The released architecture, so the prefixes match the ones the checkpoint carries.
        model = HYV4FlashForConditionalGeneration(self.model_tester.get_config())
        checkpoint_names = revert_weight_conversion(model, model.state_dict())

        for checkpoint_name, module_name in {
            "model.hc_head.hc_head_fn": "model.language_model.hc_head.hc_fn",
            "model.hc_head.hc_head_base": "model.language_model.hc_head.hc_base",
            "model.hc_head.hc_head_scale": "model.language_model.hc_head.hc_scale",
            "model.layers.0.hc_attn_layer.hc_pre.hc_fn": "model.language_model.layers.0.attn_hc.fn",
            "model.layers.0.hc_mlp_layer.hc_pre.hc_scale": "model.language_model.layers.0.ffn_hc.scale",
            "model.layers.0.mlp.post_fc2_norm.weight": "model.language_model.layers.0.post_mlp_layernorm.weight",
            "model.layers.1.mlp.routed_expert_down_proj.weight": (
                "model.language_model.layers.1.mlp.pre_expert_proj.weight"
            ),
            "model.layers.1.mlp.routed_expert_up_proj.weight": (
                "model.language_model.layers.1.mlp.post_expert_proj.weight"
            ),
            "model.layers.0.self_attn.learnable_sink_param": "model.language_model.layers.0.self_attn.sinks",
            "model.layers.0.self_attn.linear_gate.weight": "model.language_model.layers.0.self_attn.gate_proj.weight",
            "vit.perceive.proj.0.weight": "model.vision_tower.patch_merger.proj_conv.weight",
            # Names the mapping must leave alone, because the short HY4 targets reach into them
            "model.layers.1.engram_layer.unified_per_hc_scales.0": (
                "model.language_model.layers.1.engram_layer.unified_per_hc_scales.0"
            ),
            "model.layers.1.engram_layer.lookup_table": "model.language_model.layers.1.engram_layer.lookup_table",
        }.items():
            with self.subTest(checkpoint_name):
                self.assertIn(module_name, model.state_dict())
                self.assertIn(checkpoint_name, checkpoint_names)

    def test_text_renames_survive_the_composite_model_type(self):
        """A released config may label its text tower `hy_v4_flash`; the tower's renames must still apply."""
        from transformers.core_model_loading import revert_weight_conversion

        config_dict = self.model_tester.get_config().to_dict()
        config_dict["text_config"]["model_type"] = "hy_v4_flash"
        model = HYV4FlashForConditionalGeneration(HYV4FlashConfig.from_dict(config_dict))
        checkpoint_names = revert_weight_conversion(model, model.state_dict())
        self.assertIn("model.layers.0.self_attn.learnable_sink_param", checkpoint_names)

    def test_rope_parameters_follow_the_legacy_layer_names(self):
        """`layer_types` is remapped from its legacy spellings, so the rope keys naming the same layers must be too."""
        config_dict = self.model_tester.get_config().text_config.to_dict()
        config_dict["layer_types"] = [
            "deepseek_sparse_attention" if layer == "indexed_attention" else layer
            for layer in config_dict["layer_types"]
        ]
        config_dict["rope_parameters"]["deepseek_sparse_attention"] = config_dict["rope_parameters"].pop(
            "indexed_attention"
        ) | {"rope_theta": 123456.0}

        config = HYV4FlashTextConfig.from_dict(config_dict)
        self.assertNotIn("deepseek_sparse_attention", config.layer_types)
        self.assertEqual(config.rope_parameters["indexed_attention"]["rope_theta"], 123456.0)

    def test_layer_types_alternate_sliding_and_indexed(self):
        """Both attention shapes must be built, each with its own head count and rotary base."""
        config = self.model_tester.get_config().text_config
        self.assertIn("sliding_attention", config.layer_types)
        self.assertIn("indexed_attention", config.layer_types)

        model = HYV4FlashModel(self.model_tester.get_config()).language_model
        for layer_idx, (layer, layer_type) in enumerate(zip(model.layers, config.layer_types)):
            attention = layer.self_attn
            self.assertEqual(attention.attention_type, layer_type)
            self.assertEqual(attention.num_heads, config.per_layer_config[layer_idx].num_attention_heads)
            if layer_type == "sliding_attention":
                self.assertIsNone(attention.compressed_indexer)
            else:
                self.assertIsNotNone(attention.compressed_indexer)

        rotary = model.rotary_emb
        self.assertNotEqual(rotary.indexed_attention_inv_freq.tolist(), rotary.sliding_attention_inv_freq.tolist())

    def test_engram_only_fires_where_tokens_carry_an_ngram(self):
        """Visual and padding positions own no n-gram, so the memory must add nothing there."""
        config = self.model_tester.get_config()
        text_config = config.text_config
        model = HYV4FlashModel(config).to(torch_device).eval()
        engram = next(layer.engram_layer for layer in model.language_model.layers if layer.engram_layer is not None)

        input_ids = torch.tensor([[10, 11, config.image_token_id, text_config.engram_pad_id, 12]], device=torch_device)
        ngram_tokens, engram_mask = model.build_engram_inputs(input_ids, None)
        self.assertEqual(engram_mask.tolist(), [[True, True, False, False, True]])

        hidden_states = torch.randn(
            1, input_ids.shape[1], text_config.hc_mult, text_config.hidden_size, device=torch_device
        )
        with torch.no_grad():
            residual = engram(hidden_states, ngram_tokens, engram_mask)
        self.assertTrue((residual[0, 2] == 0).all())
        self.assertTrue((residual[0, 3] == 0).all())
        self.assertFalse((residual[0, 4] == 0).all())

    def test_engram_hashes_depend_on_the_preceding_tokens(self):
        """The n-gram columns are the token and its predecessors, padded at the start of a sequence."""
        config = self.model_tester.get_config()
        text_config = config.text_config
        model = HYV4FlashModel(config).to(torch_device).eval()

        input_ids = torch.tensor([[10, 11, 12, 13]], device=torch_device)
        ngram_tokens, _ = model.build_engram_inputs(input_ids, None)
        self.assertEqual(ngram_tokens.shape, (1, 4, text_config.engram_max_ngram_size))

        pad = text_config.engram_pad_id
        self.assertEqual(ngram_tokens[0, :, 0].tolist(), input_ids[0].tolist())
        self.assertEqual(ngram_tokens[0, 0, 1:].tolist(), [pad, pad])
        self.assertEqual(ngram_tokens[0, 3, 1].item(), input_ids[0, 2].item())

        # Changing a token rewrites the hashes of the positions whose n-gram covers it, and only those.
        changed = input_ids.clone()
        changed[0, 1] = 14
        changed_ngrams, _ = model.build_engram_inputs(changed, None)
        self.assertEqual(changed_ngrams[0, 0].tolist(), ngram_tokens[0, 0].tolist())
        self.assertNotEqual(changed_ngrams[0, 1].tolist(), ngram_tokens[0, 1].tolist())
        self.assertEqual(changed_ngrams[0, 3, :2].tolist(), ngram_tokens[0, 3, :2].tolist())

    def test_engram_vocab_sizes_follow_the_ngram_layout(self):
        """The table holds one list per n-gram order and one size per head; another shape must not load."""
        model = HYV4FlashModel(self.model_tester.get_config())
        engram = next(layer.engram_layer for layer in model.language_model.layers if layer.engram_layer is not None)
        self.assertEqual(engram.head_vocab_sizes.tolist(), [101, 103, 107, 109])

        text_config = self.model_tester.get_config().text_config.to_dict()
        for sizes in ([101, 103, 107, 109], [[101, 103, 107], [109]], [[101, 103]]):
            text_config["engram_layer_head_vocab_sizes"] = {"2": sizes}
            with self.subTest(sizes=sizes), self.assertRaises(StrictDataclassClassValidationError):
                HYV4FlashTextConfig.from_dict(text_config)

    def test_engram_multipliers_are_drawn_from_the_loaded_map(self):
        """The checkpoint stores the seed, not the multipliers: they are drawn again from the map it loads."""
        model = HYV4FlashForConditionalGeneration(self.model_tester.get_config())
        engram = next(layer.engram_layer for layer in model.model.language_model.layers if layer.engram_layer)
        self.assertNotIn("hash_multipliers", "".join(model.state_dict()))
        # The released checkpoint compresses its vocabulary to 92620 ids; these are its layer-2 multipliers.
        released = [81853575799317, 39690085502883, 24229915018165]
        self.assertEqual(engram.draw_hash_multipliers(92620).tolist(), released)
        engram.lookup_table[0] = 92619

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = HYV4FlashForConditionalGeneration.from_pretrained(tmp_dir)
        engram = next(layer.engram_layer for layer in reloaded.model.language_model.layers if layer.engram_layer)
        self.assertEqual(engram.hash_multipliers.tolist(), released)

    def test_engram_reads_ids_past_the_map_as_padding(self):
        """The embedding pads past the tokenizer's ids; those must hash like padding rather than index out of range."""
        model = HYV4FlashModel(self.model_tester.get_config()).to(torch_device).eval()
        engram = next(layer.engram_layer for layer in model.language_model.layers if layer.engram_layer is not None)
        table_size = engram.lookup_table.shape[0]

        beyond = torch.tensor([[[table_size, table_size + 1, table_size + 2]]], device=torch_device)
        padding = torch.full_like(beyond, engram.pad_id)
        self.assertEqual(engram.hash_ngrams(beyond).tolist(), engram.hash_ngrams(padding).tolist())

    def test_engram_hashes_the_normalized_vocabulary(self):
        """Tokens the frozen map merges must reach the memory as one n-gram."""
        model = HYV4FlashModel(self.model_tester.get_config()).to(torch_device).eval()
        engram = next(layer.engram_layer for layer in model.language_model.layers if layer.engram_layer is not None)
        engram.lookup_table[11] = 10

        ngram_tokens, _ = model.build_engram_inputs(torch.tensor([[5, 10], [5, 11]], device=torch_device), None)
        hashes = engram.hash_ngrams(ngram_tokens)
        self.assertEqual(hashes[0].tolist(), hashes[1].tolist())

    def test_compressed_indexer_leaves_short_contexts_dense(self):
        """Below the recall budget every token is recalled, so the indexer must not change the output."""
        config = self.model_tester.get_config()
        text_config = config.text_config
        budget = text_config.index_topk
        model = HYV4FlashModel(config).to(torch_device).eval()
        layer_idx = text_config.layer_types.index("indexed_attention")
        indexer = model.language_model.layers[layer_idx].self_attn.compressed_indexer

        hidden_states = torch.randn(1, budget, text_config.hidden_size, device=torch_device)
        q_resid = torch.randn(1, budget, text_config.q_lora_rank, device=torch_device)
        position_ids = torch.arange(budget, device=torch_device).view(1, 1, -1).expand(3, 1, -1)
        causal_mask = torch.zeros(1, 1, budget, budget, device=torch_device)
        causal_mask = causal_mask.masked_fill(
            torch.ones(budget, budget, dtype=torch.bool, device=torch_device).triu(1), torch.finfo(torch.float32).min
        )

        with torch.no_grad():
            masked = indexer(hidden_states, q_resid, causal_mask, position_ids, position_ids, None, budget)
        torch.testing.assert_close(masked, causal_mask)

    def test_compressed_indexer_static_cache_matches_dynamic_cache(self):
        """StaticCache must place queries at their seen-token slots, not at the end of the preallocated buffer."""
        config = self.model_tester.get_config()
        # Prompt exceeds `index_topk` (16) so the compressed indexer actively prunes on every step.
        torch.manual_seed(0)
        model = HYV4FlashForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.randint(10, config.text_config.engram_lookup_table_size, (1, 20), device=torch_device)

        with torch.no_grad():
            dynamic = model.generate(
                input_ids,
                max_new_tokens=8,
                min_new_tokens=8,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
            static = model.generate(
                input_ids,
                max_new_tokens=8,
                min_new_tokens=8,
                do_sample=False,
                cache_implementation="static",
                return_dict_in_generate=True,
                output_logits=True,
            )

        self.assertEqual(dynamic.sequences.tolist(), static.sequences.tolist())
        torch.testing.assert_close(torch.stack(dynamic.logits), torch.stack(static.logits), atol=1e-4, rtol=1e-4)
