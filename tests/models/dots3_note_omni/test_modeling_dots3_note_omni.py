# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the PyTorch Dots3-Note Omni model."""

import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized
from safetensors import safe_open

from transformers import (
    Dots3NoteOmniConfig,
    Dots3NoteOmniForCausalLM,
    Dots3NoteOmniForConditionalGeneration,
    FineGrainedFP8Config,
    is_torch_available,
)
from transformers.cache_utils import (
    DynamicCache,
    DynamicIndexedLayer,
    DynamicSlidingWindowLayer,
    StaticCache,
    StaticIndexedLayer,
    StaticSlidingWindowLayer,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, rename_source_key
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_batched_and_grouped_inference,
)


if is_torch_available():
    import torch

    from transformers.models.dots3_note_omni.modeling_dots3_note_omni import (
        Dots3NoteOmniAudioModel,
        Dots3NoteOmniTextForCausalLM,
        Dots3NoteOmniTextIndexer,
        Dots3NoteOmniTextModel,
        Dots3NoteOmniVisionModel,
        dsa_sparse_attention_forward,
        eager_attention_forward,
        quantize_indexer_fp8,
    )


def get_tiny_config(use_dsa=False):
    vision_config = {
        "embed_dim": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "pyramid_num_routed": [-1, 2],
        "capacity_factor": 2,
        "adapter_in_dim": 32,
        "adapter_out_dim": 32,
        "adapter_merge_size": 2,
    }
    audio_config = {
        "whisper_config": {
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "num_mel_bins": 8,
            "max_source_positions": 32,
            "activation_function": "swiglu",
        },
        "feature_size": 8,
        "n_fft": 16,
        "hop_length": 4,
        "chunk_seconds": 1,
        "downsample_hidden_size": 4,
        "conv_bucket_step": None,
        "conv_bucket_max_elements": None,
        "adapter_input_size": 32,
        "adapter_output_size": 32,
    }
    config = Dots3NoteOmniConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        seq_length=128,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        head_dim=16,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window_size=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_q_lora_rank=16,
        swa_kv_lora_rank=16,
        swa_head_dim=16,
        swa_qk_nope_head_dim=8,
        swa_qk_rope_head_dim=8,
        swa_v_head_dim=8,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        use_dsa=use_dsa,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_experts_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        first_k_dense_replace=1,
        image_token_id=120,
        image_start_token_id=116,
        image_end_token_id=117,
        video_token_id=122,
        audio_start_token_id=118,
        audio_end_token_id=119,
        audio_token_id=121,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vision_config=vision_config,
        audio_config=audio_config,
    )
    config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


class Dots3NoteOmniTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Dots3NoteOmniTextModel
        config_class = Dots3NoteOmniConfig
        causal_lm_class = Dots3NoteOmniTextForCausalLM

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            batch_size=2,
            seq_length=7,
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        # Initial support is inference-focused, so common coverage targets forward, cache, and generation.
        self.is_training = False

    def get_config(self):
        config = get_tiny_config()
        # Generic cross-backend and save/load tests should not depend on arbitrary tie-breaking from
        # a randomly initialized hard top-k indexer. Dedicated DSA tests below use a genuinely sparse top-k.
        config.index_topk = config.max_position_embeddings
        return config


@require_torch
class Dots3NoteOmniTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteOmniTextModelTester
    all_model_classes = (Dots3NoteOmniTextForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-generation": Dots3NoteOmniTextForCausalLM} if is_torch_available() else {}
    _is_stateful = True

    @unittest.skip(reason="Initial Dots3-Note Omni support is inference-only.")
    def test_gradient_checkpointing_enable_disable(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip(reason="Dynamic sparse attention uses a model-specific cache incompatible with assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip(reason="Dynamic sparse attention uses a model-specific cache incompatible with assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="MLA key and value caches intentionally have different head dimensions.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="MLA key and value caches intentionally have different head dimensions.")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="MLA uses a model-specific asymmetric key/value cache format.")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(reason="The model-specific DSA/SWA cache cannot be reconstructed by torch.nn.DataParallel.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # SonicMoE and DeepGEMM are optional integrations. This model's required expert
        # implementations are eager, batched_mm, and grouped_mm.
        with patch("tests.test_modeling_common.is_kernels_available", return_value=False):
            _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

    # DSA selects positions with a hard top-k. Tiny numerical changes caused by padding or
    # sequence packing can change the selected positions, so generic exact-equivalence tests do not apply.
    # Deterministic physical cache/padding coordinate coverage lives in Dots3NoteOmniModelTest below.
    @unittest.skip(reason="DSA hard top-k selection is numerically discontinuous across padding shifts.")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="DSA hard top-k selection is sensitive to sequence packing.")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="DSA hard top-k selection is sensitive to sequence packing.")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason=(
            "The released expert conversion is intentionally many-to-one (gate/up projections are packed); "
            "the explicit checkpoint round-trip test covers the supported conversion path."
        )
    )
    def test_reverse_loading_mapping(self):
        pass


@require_torch
class Dots3NoteOmniModelTest(unittest.TestCase):
    all_model_classes = (
        (
            Dots3NoteOmniAudioModel,
            Dots3NoteOmniForCausalLM,
            Dots3NoteOmniForConditionalGeneration,
            Dots3NoteOmniVisionModel,
        )
        if is_torch_available()
        else ()
    )

    def test_text_forward_and_cache(self):
        config = get_tiny_config()
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9, 2]])

        with torch.no_grad():
            full_logits = model(input_ids, use_cache=False).logits
            prefix = model(input_ids[:, :-1], use_cache=True)
            cached_logits = model(input_ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits

        self.assertEqual(full_logits.shape, (1, 5, config.vocab_size))
        torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], rtol=1e-4, atol=1e-4)

    def test_dsa_text_forward(self):
        config = get_tiny_config(use_dsa=True)
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        with torch.no_grad():
            logits = model(torch.tensor([[1, 5, 6, 7, 2]]), use_cache=False).logits
        self.assertTrue(torch.isfinite(logits).all())

    def test_dsa_sparse_attention_matches_dense_reference(self):
        torch.manual_seed(0)
        batch_size, num_heads, query_length, key_length = 2, 4, 5, 9
        query = torch.randn(batch_size, num_heads, query_length, 8)
        key = torch.randn(batch_size, num_heads, key_length, 8)
        value = torch.randn(batch_size, num_heads, key_length, 6)
        cache_position = torch.arange(4, 9)
        query_positions = cache_position.expand(batch_size, -1)
        topk_indices = torch.stack(
            [query_positions, query_positions - 1, query_positions - 2, query_positions - 3], dim=-1
        ).to(torch.int32)
        padding_mask = torch.ones(batch_size, key_length, dtype=torch.long)
        attention_mask = torch.ones(batch_size, 1, query_length, key_length, dtype=torch.bool)
        attention_mask[:, :, :, 5] = False
        scaling = query.shape[-1] ** -0.5

        actual, _ = dsa_sparse_attention_forward(
            torch.nn.Identity().eval(),
            query,
            key,
            value,
            attention_mask=attention_mask,
            indices=topk_indices,
            cache_position=cache_position,
            padding_mask=padding_mask,
            scaling=scaling,
            query_chunk_size=2,
            head_chunk_size=2,
        )

        scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
        allowed = torch.zeros(batch_size, query_length, key_length, dtype=torch.bool)
        allowed.scatter_(-1, topk_indices.long(), True)
        allowed &= attention_mask[:, 0]
        scores.masked_fill_(~allowed[:, None], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        expected = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_dsa_indexer_left_padding_uses_physical_cache_positions(self):
        config = get_tiny_config(use_dsa=True)
        config.index_topk = 2
        indexer = Dots3NoteOmniTextIndexer(config, layer_idx=0).eval()
        for parameter in indexer.parameters():
            torch.nn.init.zeros_(parameter)

        sequence_length = 5
        hidden_states = torch.zeros(1, sequence_length, config.hidden_size)
        q_lora = torch.zeros(1, sequence_length, config.q_lora_rank)
        cos = torch.ones(1, sequence_length, config.qk_rope_head_dim // 2)
        sin = torch.zeros_like(cos)
        padding_mask = torch.tensor([[0, 0, 0, 1, 1]])

        topk_indices = indexer(
            hidden_states,
            q_lora,
            cos,
            sin,
            padding_mask,
            torch.arange(sequence_length),
        )

        self.assertEqual(set(topk_indices[0, -1].tolist()), {3, 4})

    def test_dsa_indexer_requires_shared_4d_attention_mask(self):
        config = get_tiny_config(use_dsa=True)
        config.index_topk = 1
        indexer = Dots3NoteOmniTextIndexer(config, layer_idx=0).eval()
        for parameter in indexer.parameters():
            torch.nn.init.zeros_(parameter)

        sequence_length = 4
        hidden_states = torch.zeros(1, sequence_length, config.hidden_size)
        q_lora = torch.zeros(1, sequence_length, config.q_lora_rank)
        cos = torch.ones(1, sequence_length, config.qk_rope_head_dim // 2)
        sin = torch.zeros_like(cos)
        allowed = torch.eye(sequence_length, dtype=torch.bool).view(1, 1, sequence_length, sequence_length)

        for attention_mask in (allowed, torch.where(allowed, 0.0, -10_000.0)):
            with self.subTest(dtype=attention_mask.dtype):
                topk_indices = indexer(
                    hidden_states,
                    q_lora,
                    cos,
                    sin,
                    None,
                    torch.arange(sequence_length),
                    attention_mask=attention_mask,
                )
                torch.testing.assert_close(
                    topk_indices.squeeze(0).squeeze(-1),
                    torch.arange(sequence_length, dtype=torch.int32),
                )

        per_head_mask = allowed.expand(-1, config.num_attention_heads, -1, -1).clone()
        per_head_mask[:, 1, :, 0] = False

        with self.assertRaisesRegex(ValueError, "different per-head masks are not supported"):
            indexer(
                hidden_states,
                q_lora,
                cos,
                sin,
                None,
                torch.arange(sequence_length),
                attention_mask=per_head_mask,
            )

    def test_dsa_sparse_attention_left_padding_uses_physical_cache_positions(self):
        query = torch.zeros(1, 1, 1, 2)
        key = torch.zeros(1, 1, 5, 2)
        value = torch.tensor([[[[0.0], [100.0], [0.0], [0.0], [7.0]]]])
        topk_indices = torch.tensor([[[1, 4]]], dtype=torch.int32)
        padding_mask = torch.tensor([[0, 0, 0, 1, 1]])

        actual, _ = dsa_sparse_attention_forward(
            torch.nn.Identity().eval(),
            query,
            key,
            value,
            attention_mask=None,
            indices=topk_indices,
            cache_position=torch.tensor([4]),
            padding_mask=padding_mask,
            scaling=1.0,
        )

        torch.testing.assert_close(actual, torch.tensor([[[[7.0]]]]))

    def test_dsa_sparse_attention_zeroes_finite_fully_masked_rows(self):
        actual, _ = dsa_sparse_attention_forward(
            torch.nn.Identity().eval(),
            query=torch.zeros(1, 1, 1, 2),
            key=torch.zeros(1, 1, 2, 2),
            value=torch.tensor([[[[2.0], [6.0]]]]),
            attention_mask=torch.full((1, 1, 1, 2), -10_000.0),
            indices=torch.tensor([[[0, 1]]], dtype=torch.int32),
            cache_position=torch.tensor([1]),
            scaling=1.0,
        )

        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_dsa_sdpa_zeroes_finite_fully_masked_rows(self):
        config = get_tiny_config(use_dsa=True)
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        config._attn_implementation = "sdpa"
        model = Dots3NoteOmniTextModel(config).eval()
        hidden_states = torch.randn(1, 2, config.hidden_size)
        position_ids = torch.arange(2).unsqueeze(0)
        cos, sin = model.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
            actual, _ = model.layers[0].self_attn(
                hidden_states,
                cos,
                sin,
                position_ids,
                attention_mask=torch.full((1, 1, 2, 2), -10_000.0),
                cache_position=torch.arange(2),
            )

        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_dsa_chunked_prefill_matches_one_shot(self):
        config = get_tiny_config(use_dsa=True)
        # Keep every key in this cache-parity test. Randomly initialized tiny indexers can produce exact
        # score ties, for which `topk` is allowed to pick different tied keys as the cached key width grows.
        # Sparse top-k math and sparse cached generation are covered independently below.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9, 6, 13, 12, 5, 8, 10, 2]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
            past_key_values = None
            chunks = []
            for start in range(0, input_ids.shape[1], 3):
                stop = min(start + 3, input_ids.shape[1])
                outputs = model(
                    input_ids[:, start:stop],
                    attention_mask=attention_mask[:, :stop],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                chunks.append(outputs.logits)

        actual = torch.cat(chunks, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_dsa_chunked_prefill_generate_matches_one_shot(self):
        config = get_tiny_config(use_dsa=True)
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9, 6, 13, 12]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            expected = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=3,
            )
            past_key_values = None
            for start in range(0, input_ids.shape[1] - 1, 2):
                stop = min(start + 2, input_ids.shape[1] - 1)
                outputs = model(
                    input_ids[:, start:stop],
                    attention_mask=attention_mask[:, :stop],
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past_key_values = outputs.past_key_values
            actual = model.generate(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                do_sample=False,
                max_new_tokens=3,
            )

        torch.testing.assert_close(actual, expected)

    def test_dsa_left_padded_batch_cache_matches_unpadded_decode(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True)
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        # Select every key so this test targets physical cache/padding coordinates rather than
        # the numerical discontinuity of hard top-k selection on a randomly initialized indexer.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        model.set_experts_implementation("eager")

        sequences = [
            torch.tensor([[1, 7, 11, 9, 2]]),
            torch.tensor([[1, 8, 6, 13, 12, 5, 10, 2]]),
        ]
        next_tokens = torch.tensor([[14], [15]])
        max_length = max(sequence.shape[1] for sequence in sequences)
        input_ids = torch.cat(
            [torch.nn.functional.pad(sequence, (max_length - sequence.shape[1], 0)) for sequence in sequences]
        )
        attention_mask = input_ids.ne(config.pad_token_id).long()
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        with torch.no_grad():
            prefill = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            batched = model(
                next_tokens,
                attention_mask=torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1),
                position_ids=attention_mask.sum(-1, keepdim=True),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            ).logits[:, -1]

            unpadded = []
            for sequence, next_token in zip(sequences, next_tokens):
                sequence_mask = torch.ones_like(sequence)
                sequence_prefill = model(
                    sequence,
                    attention_mask=sequence_mask,
                    position_ids=torch.arange(sequence.shape[1]).unsqueeze(0),
                    use_cache=True,
                )
                unpadded.append(
                    model(
                        next_token.view(1, 1),
                        attention_mask=torch.ones(1, sequence.shape[1] + 1, dtype=torch.long),
                        position_ids=torch.tensor([[sequence.shape[1]]]),
                        past_key_values=sequence_prefill.past_key_values,
                        use_cache=True,
                    ).logits[:, -1]
                )

        torch.testing.assert_close(batched, torch.cat(unpadded), rtol=1e-4, atol=1e-4)

    def test_dsa_sparse_fallback_uses_attention_interface(self):
        config = get_tiny_config(use_dsa=True)
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        config.index_topk = 2
        model = Dots3NoteOmniTextForCausalLM(config).eval()

        with patch.object(
            ALL_ATTENTION_FUNCTIONS, "get_interface", wraps=ALL_ATTENTION_FUNCTIONS.get_interface
        ) as get_interface:
            with torch.no_grad():
                logits = model(torch.tensor([[1, 7, 11, 9, 6, 2]]), use_cache=False).logits

        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            any(call.args == ("eager", dsa_sparse_attention_forward) for call in get_interface.call_args_list)
        )

    def test_text_attention_backend_kwargs_reach_interface(self):
        config = get_tiny_config(use_dsa=False)
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        marker = object()
        received_markers = []

        def attention_spy(module, query, key, value, attention_mask, backend_marker=None, **kwargs):
            received_markers.append(backend_marker)
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

        with patch.object(ALL_ATTENTION_FUNCTIONS, "get_interface", return_value=attention_spy):
            with torch.no_grad():
                model(torch.tensor([[1, 7, 2]]), use_cache=False, backend_marker=marker)

        self.assertEqual(received_markers, [marker] * config.num_hidden_layers)

    def test_dsa_cache_uses_dynamic_layer_dispatch(self):
        config = get_tiny_config(use_dsa=True)
        model = Dots3NoteOmniTextForCausalLM(config).eval()

        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([[1, 7, 11, 9]]),
                do_sample=False,
                max_new_tokens=2,
                return_dict_in_generate=True,
            )

        cache = outputs.past_key_values
        self.assertIs(type(cache), DynamicCache)
        self.assertIsInstance(cache.layers[0], DynamicIndexedLayer)
        self.assertIsInstance(cache.layers[1], DynamicSlidingWindowLayer)
        self.assertEqual(cache.layers[0].indexer_keys.dtype, torch.float32)

    def test_dsa_static_cache_matches_dynamic_cache(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True)
        model = Dots3NoteOmniTextForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            dynamic = model(input_ids, attention_mask=attention_mask, use_cache=True)
            next_token = dynamic.logits[:, -1:].argmax(dim=-1)
            decode_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)
            dynamic_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=dynamic.past_key_values,
                use_cache=True,
            )

            static_cache = StaticCache(config=config, max_cache_len=16)
            static = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=static_cache,
                cache_position=torch.arange(input_ids.shape[1]),
                use_cache=True,
            )
            static_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=static.past_key_values,
                cache_position=torch.tensor([input_ids.shape[1]]),
                use_cache=True,
            )

        self.assertIsInstance(static_cache.layers[0], StaticIndexedLayer)
        self.assertIsInstance(static_cache.layers[1], StaticSlidingWindowLayer)
        torch.testing.assert_close(static_decode.logits, dynamic_decode.logits, rtol=1e-4, atol=1e-4)

    def test_dsa_fp8_index_key_is_exact_in_bfloat16_cache(self):
        torch.manual_seed(0)
        key = torch.randn(2, 7, 128) * 10
        quantized_key, key_scale = quantize_indexer_fp8(key)
        dequantized_key = quantized_key.float() * key_scale

        torch.testing.assert_close(dequantized_key.bfloat16().float(), dequantized_key, rtol=0, atol=0)

    def test_dsa_precomputed_mask_uses_dispatched_layer_type(self):
        model = Dots3NoteOmniTextForCausalLM(get_tiny_config(use_dsa=True)).eval()
        full_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        sliding_mask = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2)

        actual_full, actual_sliding = model.model._make_masks(
            {
                "deepseek_sparse_attention": full_mask,
                "sliding_attention": sliding_mask,
            },
            inputs_embeds=None,
            past_key_values=None,
            position_ids=None,
        )

        self.assertIs(actual_full, full_mask)
        self.assertIs(actual_sliding, sliding_mask)

    def test_vision_forward(self):
        config = get_tiny_config().vision_config
        model = Dots3NoteOmniVisionModel(config).eval()
        pixel_values = torch.randn(4, 3 * config.temporal_patch_size * config.patch_size**2)
        grid_thw = torch.tensor([[1, 2, 2]])
        with torch.no_grad():
            outputs = model(pixel_values, grid_thw)
        self.assertEqual(outputs.last_hidden_state.shape, (4, config.embed_dim))
        self.assertEqual(outputs.pooler_output.shape, (1, config.hidden_size))

    def test_audio_forward(self):
        config = get_tiny_config().audio_config
        model = Dots3NoteOmniAudioModel(config).eval()
        with torch.no_grad():
            outputs = model(
                input_features=torch.randn(1, config.feature_size, 16),
                chunk_sample_lengths=torch.tensor([64]),
                chunk_token_lengths=torch.tensor([2]),
                audio_chunk_counts=torch.tensor([1]),
            )
        self.assertEqual(outputs.audio_embeds.shape, (2, config.adapter_output_size))
        self.assertEqual(outputs.audio_token_lengths.tolist(), [2])

    def test_audio_flash_attention_3_uses_requested_backend(self):
        config = get_tiny_config().audio_config
        config.attention_backend = "flash_attention_3"
        model = Dots3NoteOmniAudioModel(config).eval()
        requested_backends = []

        def flash_attention_spy(query, key, value, *args, attn_implementation=None, **kwargs):
            requested_backends.append(attn_implementation)
            return query

        with patch(
            "transformers.models.dots3_note_omni.modeling_dots3_note_omni._flash_attention_forward",
            side_effect=flash_attention_spy,
        ):
            with torch.no_grad():
                outputs = model(
                    input_features=torch.randn(1, config.feature_size, 16),
                    chunk_sample_lengths=torch.tensor([64]),
                    chunk_token_lengths=torch.tensor([2]),
                    audio_chunk_counts=torch.tensor([1]),
                )

        self.assertEqual(requested_backends, ["flash_attention_3"] * config.whisper_config["encoder_layers"])
        self.assertTrue(torch.isfinite(outputs.audio_embeds).all())

    def test_omni_image_and_audio_forward(self):
        config = get_tiny_config()
        model = Dots3NoteOmniForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 120, 5, 121, 121, 7, 2]])
        pixel_values = torch.randn(4, 3 * config.vision_config.temporal_patch_size * 2**2)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                input_features=torch.randn(1, config.audio_config.feature_size, 16),
                chunk_sample_lengths=torch.tensor([64]),
                chunk_token_lengths=torch.tensor([2]),
                audio_chunk_counts=torch.tensor([1]),
                audio_token_lengths=torch.tensor([2]),
                use_cache=False,
            )
        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], config.vocab_size))
        self.assertTrue(torch.isfinite(outputs.logits).all())

    def test_omni_video_forward(self):
        config = get_tiny_config()
        model = Dots3NoteOmniForCausalLM(config).eval()
        input_ids = torch.tensor([[1, config.video_token_id, 7, 2]])
        pixel_values_videos = torch.randn(
            4,
            config.vision_config.num_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size**2,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=torch.tensor([[1, 2, 2]]),
                use_cache=False,
            )
        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], config.vocab_size))
        self.assertTrue(torch.isfinite(outputs.logits).all())

    def test_omni_chunked_prefill_processes_cached_media(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True)
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteOmniForCausalLM(config).eval()
        prefix_ids = torch.tensor([[1, 5]])
        patch_width = (
            config.vision_config.num_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size**2
        )
        cases = {
            "image": (
                torch.tensor([[config.image_token_id, 7, 2]]),
                {
                    "pixel_values": torch.randn(4, patch_width),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "video": (
                torch.tensor([[config.video_token_id, 7, 2]]),
                {
                    "pixel_values_videos": torch.randn(4, patch_width),
                    "video_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "audio": (
                torch.tensor([[config.audio_token_id, config.audio_token_id, 7, 2]]),
                {
                    "input_features": torch.randn(1, config.audio_config.feature_size, 16),
                    "chunk_sample_lengths": torch.tensor([64]),
                    "chunk_token_lengths": torch.tensor([2]),
                    "audio_chunk_counts": torch.tensor([1]),
                    "audio_token_lengths": torch.tensor([2]),
                },
            ),
        }

        for modality, (suffix_ids, media_inputs) in cases.items():
            with self.subTest(modality=modality):
                input_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    expected = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **media_inputs,
                    ).logits[:, -suffix_ids.shape[1] :]
                    prefix = model(prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True)
                    actual = model(
                        suffix_ids,
                        attention_mask=attention_mask,
                        past_key_values=prefix.past_key_values,
                        use_cache=True,
                        **media_inputs,
                    ).logits

                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_expert_checkpoint_conversion_roundtrip(self):
        torch.manual_seed(13)
        model = Dots3NoteOmniForCausalLM(get_tiny_config()).eval()
        input_ids = torch.tensor([[1, 5, 6, 2]])
        with torch.no_grad():
            expected = model(input_ids, use_cache=False).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            with safe_open(f"{tmpdirname}/model.safetensors", framework="pt") as checkpoint:
                checkpoint_keys = set(checkpoint.keys())
            self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.up_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.down_proj.weight", checkpoint_keys)
            self.assertNotIn("model.layers.1.mlp.experts.gate_up_proj", checkpoint_keys)

            reloaded = Dots3NoteOmniForCausalLM.from_pretrained(tmpdirname).eval()

        with torch.no_grad():
            actual = reloaded(input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_expert_fp8_scale_conversion(self):
        model = Dots3NoteOmniForCausalLM(get_tiny_config())
        model.config.quantization_config = FineGrainedFP8Config(dequantize=True)
        conversions = get_model_conversion_mapping(model)
        converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]
        prefix = "model.layers.1.mlp.experts.0.gate_proj"

        weight_key, _ = rename_source_key(f"{prefix}.weight", [], converters)
        scale_key, _ = rename_source_key(f"{prefix}.weight_scale_inv", [], converters)
        self.assertEqual(weight_key, "model.layers.1.mlp.experts.gate_up_proj")
        self.assertEqual(scale_key, "model.layers.1.mlp.experts.gate_up_proj_scale_inv")

        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize=True))
        quantizer.pre_quantized = True
        dequant_conversions = quantizer.update_weight_conversions(conversions)
        dequant_converters = [
            conversion for conversion in dequant_conversions if isinstance(conversion, WeightConverter)
        ]
        dequant_weight_key, _ = rename_source_key(f"{prefix}.weight", [], dequant_converters)
        dequant_scale_key, _ = rename_source_key(f"{prefix}.weight_scale_inv", [], dequant_converters)
        self.assertEqual(dequant_weight_key, "model.layers.1.mlp.experts.gate_up_proj")
        self.assertEqual(dequant_scale_key, dequant_weight_key)

    def test_fp8_partial_weight_block_dequantization(self):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize=True, weight_block_size=(128, 128)))
        quantizer.pre_quantized = True

        for rows in (576, 1088):
            with self.subTest(rows=rows):
                weight = torch.ones(rows, 256, dtype=torch.float8_e4m3fn)
                scales = torch.arange(1, ((rows + 127) // 128) * 2 + 1, dtype=torch.float32).reshape(-1, 2)

                actual = Fp8Dequantize(quantizer)._dequantize_one(weight, scales, output_dtype=torch.float32)
                expected = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :256]

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
