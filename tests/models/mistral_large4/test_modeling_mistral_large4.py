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
"""Testing suite for the PyTorch Mistral Large 4 model."""

import inspect
import unittest
from contextlib import contextmanager

from parameterized import parameterized

from transformers import (
    MistralLarge4Config,
    MistralLarge4TextConfig,
    PixtralVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MistralLarge4ForConditionalGeneration, MistralLarge4Model
    from transformers.models.mistral_large4.modeling_mistral_large4 import (
        MistralLarge4TextIndexer,
        MistralLarge4TextNoRopeIndexer,
    )


@contextmanager
def indexer_leaky_relu():
    """Swaps `F.relu` for `F.leaky_relu` in the body of the indexer forwards (whatever decorators wrap them).

    With random weights, the indexer's ReLU zeroes many scores and `topk` breaks those exact ties differently depending
    on tensor shapes (cache type, padding). A leaky ReLU keeps the scores distinct, so tests comparing such code paths
    select the same tokens. The real ReLU is covered by the integration tests, on trained weights.
    """
    forwards = [inspect.unwrap(cls.forward) for cls in (MistralLarge4TextIndexer, MistralLarge4TextNoRopeIndexer)]
    codes = [forward.__code__ for forward in forwards]
    for forward, code in zip(forwards, codes):
        forward.__code__ = code.replace(co_names=tuple("leaky_relu" if n == "relu" else n for n in code.co_names))
    try:
        yield
    finally:
        for forward, code in zip(forwards, codes):
            forward.__code__ = code


class MistralLarge4VisionText2TextModelTester(VLMModelTester):
    base_model_class = MistralLarge4Model
    config_class = MistralLarge4Config
    text_config_class = MistralLarge4TextConfig
    vision_config_class = PixtralVisionConfig
    conditional_generation_class = MistralLarge4ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        # Pixtral: 16x16 images, 4x4 patches, merged 2x2 by the projector -> 4 image tokens per image
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("attention_probs_dropout_prob", 0.0)
        # MLA without Q-LoRA
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("qk_rope_head_dim", 8)
        kwargs.setdefault("qk_nope_head_dim", 8)
        kwargs.setdefault("v_head_dim", 16)
        # DSA indexer. `index_topk` is smaller than the test sequences, so the indexer drops tokens
        kwargs.setdefault("index_topk", 4)
        kwargs.setdefault("index_head_dim", 16)
        kwargs.setdefault("index_n_heads", 2)
        # RoPE, NoPE, RoPE, NoPE. Since KV can only be shared by layers with the same PE, layer 3 reuse layer 1's KV,
        # and layer 2 reuses layer 0's KV. We also test index sharing in the last two layers.
        kwargs.setdefault("num_hidden_layers", 4)
        kwargs.setdefault("no_rope_layers", [1, 0, 1, 0])
        kwargs.setdefault("kv_source_layers", [0, 1, 0, 1])
        kwargs.setdefault("indexer_types", ["full", "full", "shared", "shared"])
        # MoE: layer 0 is dense, the others route to 2 of 8 experts plus a shared expert
        kwargs.setdefault("first_k_dense_replace", 1)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("n_group", 1)
        kwargs.setdefault("topk_group", 1)
        super().__init__(parent, **kwargs)

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_sizes": torch.tensor([[self.image_size, self.image_size]] * batch_size, device=torch_device)}


@require_torch
class MistralLarge4ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MistralLarge4VisionText2TextModelTester
    # Pixtral concatenates all images in a single sequence
    skip_test_image_features_output_shape = True
    # The DSA indexer runs under `torch.no_grad`
    test_all_params_have_gradient = False

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # KV-shared layers attend to the cache of their source layer and leave their own cache layer empty, and NoPE
        # layers cache an empty rotary key
        config = config.get_text_config(decoder=True)
        self.assertEqual(config.num_hidden_layers, len(past_key_values))
        k_shape, v_shape = self._get_attention_shape(batch_size, seq_length, config)
        for layer_idx, layer in enumerate(past_key_values.layers):
            if config.kv_source_layers[layer_idx] != layer_idx:
                self.assertIsNone(layer.keys)
            else:
                expected_v_shape = v_shape if config.no_rope_layers[layer_idx] else (*v_shape[:-1], 0)
                self.assertEqual(layer.keys.shape, k_shape)
                self.assertEqual(layer.values.shape, expected_v_shape)

    def test_layers_specificities(self):
        """Checks that no extra projector is created for layers that don't need it, and no RoPE dimension is
        accidentally set for NoPE layers."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = MistralLarge4Model(config)
        for layer_idx, layer in enumerate(model.language_model.layers):
            # Test that KV sharing layers have no kv_a projection
            owns_kv = config.text_config.kv_source_layers[layer_idx] == layer_idx
            self.assertEqual(hasattr(layer.self_attn, "kv_a_proj_with_mqa"), owns_kv)
            # Test that NoPE layers have no rope dimensions
            uses_rope = bool(config.text_config.no_rope_layers[layer_idx])
            expected_rope_dim = config.text_config.qk_rope_head_dim if uses_rope else 0
            self.assertEqual(layer.self_attn.qk_rope_head_dim, expected_rope_dim)

    @parameterized.expand([("dynamic",), ("static",)])
    def test_nope_and_kv_sharing_cache_matches_full_forward(self, cache_implementation):
        """Tests that the KV sharing and NoPE layers behave the same in prefill and decode."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config._attn_implementation = "eager"
        model = MistralLarge4ForConditionalGeneration(config).to(torch_device).eval()
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        prompt_length = inputs["input_ids"].shape[1]

        # The cached decoding steps and the full forward score tokens over keys of different shapes
        with indexer_leaky_relu():
            # A generated image token would have no image features in the full forward below, and a generated pad
            # token (zero embedding) gives a zero indexer query, i.e. scores that all tie
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                cache_implementation=cache_implementation,
                output_logits=True,
                return_dict_in_generate=True,
                suppress_tokens=[config.image_token_id, config.text_config.pad_token_id],
            )
            with torch.no_grad():
                full_logits = model(
                    input_ids=outputs.sequences,
                    pixel_values=inputs["pixel_values"],
                    image_sizes=inputs["image_sizes"],
                ).logits

        # The logits of each decoding step match the ones of a single forward over the whole sequence
        cached_logits = torch.stack(outputs.logits, dim=1)
        torch.testing.assert_close(cached_logits, full_logits[:, prompt_length - 1 : -1], rtol=1e-4, atol=1e-4)

    def test_indexer_limits_attended_tokens(self):
        """Tests that the index actually limits the number of tokens seens during attention"""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.index_topk = 2
        config.text_config._attn_implementation = "eager"
        model = MistralLarge4ForConditionalGeneration(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Each query attends to at most `index_topk` tokens in every layer, shared indexer layers included
        for attentions in outputs.attentions:
            attended = (attentions > 0).sum(dim=-1)
            self.assertLessEqual(attended.max().item(), 2)

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Pixtral backbone already overrides this test, no need to test again.")
    def test_vision_axial_rope(self):
        pass
