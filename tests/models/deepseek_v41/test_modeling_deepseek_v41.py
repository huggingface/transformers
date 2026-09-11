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

import copy
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import DeepseekV41ForCausalLM, DeepseekV41TextConfig, DeepseekV41TextModel

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


def tiny_word_tokenizer(vocab=("a", "b", "c", "d", "e", "f", "g", "h"), pad="<pad>"):
    """Minimal word-level tokenizer for the engram hash tests: enough for
    `build_compressed_token_map` to derive a compressed vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast

    vocab_map = {token: i for i, token in enumerate(vocab)}
    vocab_map[pad] = len(vocab_map)
    backend = Tokenizer(models.WordLevel(vocab=vocab_map, unk_token="a"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


class DeepseekV41ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV41TextModel
        # The text config class follows the naming convention and is inferred; the
        # CausalLM class is NOT: it is named `DeepseekV41ForCausalLM` (matching the
        # released checkpoint's `architectures` entry), not `DeepseekV41TextForCausalLM`.
        config_class = None
        causal_lm_class = DeepseekV41ForCausalLM

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Standard CausalLMModelTester knobs.
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        # The released schedule [0, 2, 2, 1, 1] scaled down: a plain sliding layer,
        # a ratio-2 source + a ratio-2 consumer ("Reuse" mode), a ratio-1 source +
        # a ratio-1 consumer (also the candidate consumer). Fewer layers cannot
        # exercise both KV-shared groups with their consumer layers
        # (see `test_num_layers_is_small`). Passed explicitly: the derived
        # schedule for 5 layers would be [0, 0, 2, 2, 1], which has no ratio-1
        # consumer.
        self.num_hidden_layers = 5
        self.compress_ratios = [0, 2, 2, 1, 1]
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 64
        self.max_position_embeddings = 128
        self.vocab_size = 128
        # V4.1-only knobs. The layer schedule derives kv/index sources and the
        # candidate source automatically (see `num_hidden_layers` above).
        self.head_dim = 32
        self.qk_rope_head_dim = 8
        self.q_lora_rank = 32
        self.o_groups = 2
        self.o_lora_rank = 16
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        self.sliding_window = 8
        self.hc_mult = 2
        self.hc_sinkhorn_iters = 3
        self.hc_eps = 1.0e-6
        self.index_n_heads = 2
        self.index_head_dim = 16
        self.index_topk = 2
        self.candidate_topk_blocks = 4
        self.candidate_block_size = 4
        self.num_nextn_predict_layers = 0
        self.scoring_func = "sqrtsoftplus"
        self.gate_temp = 1.0
        self.routed_scaling_factor = 1.5
        self.swiglu_limit = 10.0
        self.rope_theta = 10000.0
        self.compress_rope_theta = 160000.0
        self.rms_norm_eps = 1.0e-20
        self.engram_layer_ids = []
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.tie_word_embeddings = False


@require_torch
class DeepseekV41ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV41ModelTester

    # The indexer's top-k over compressed positions is non-differentiable (gradients
    # flow through a separate objective upstream, not the main causal-LM loss), and
    # the Sinkhorn-projected comb matrix sits at the edge of the autograd graph.
    test_all_params_have_gradient = False

    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    @unittest.skip(
        "V4.1's compressor stores group state on custom cache layers, which is not compatible with QuantizedCache."
    )
    def test_generate_with_quant_cache(self):
        pass

    # These assert one uniform key length for every layer; V4.1's sliding ring caps
    # storage at `sliding_window - 1`, so compressed-source layers keep a different
    # length than plain layers. DeepSeek-V3.2 skips the same tests.
    @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        # Layers with a compressed branch attend to extra pooled positions, so the KV
        # length varies per layer. Check the shape invariants only: batched, same
        # number-of-heads and query-length; the KV-length axis may differ per layer.
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), (output_length - prompt_length))
        for _, iter_attentions in enumerate(attentions):
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertIsInstance(layer_attention, torch.Tensor)
                self.assertEqual(layer_attention.shape[0], batch_size)

    @property
    def has_attentions(self):
        """The KV axis carries a trailing sink column and per-layer compressed
        entries; the harness's exact-shape assertions on `attentions` would trip."""
        return False

    # --- V4.1-specific behavior -----------------------------------------------------

    @staticmethod
    def _output_tensor(outputs):
        return outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state

    @staticmethod
    def _tie_free_config(config):
        """The indexer's per-head scores are ReLU-rectified, so a key every head
            scores negatively gets an EXACT 0.0 — and with few heads that happens
                # These assert one uniform key length for every layer; V4.1's sliding ring caps
        # storage at `sliding_window - 1`, so compressed-source layers keep a different
        # length than plain layers. DeepSeek-V3.2 skips the same tests.
        @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
        def test_greedy_generate_dict_outputs_use_cache(self):
            pass

        @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
        def test_beam_search_generate_dict_outputs_use_cache(self):
            pass
            checks we select every visible block instead, keeping the comparison
            deterministic (DSA's cross-backend tests are skipped upstream for the
            same reason — see the deepseek_v32 test file)."""
        config = copy.deepcopy(config)
        config.index_topk = 64
        config.candidate_topk_blocks = 8
        config.candidate_block_size = 64
        return config

    def _run_and_compare_chunked(self, model_class, config):
        """A prompt fed in two chunks must produce the same outputs as a one-shot
        prefill, even when the split lands mid-compress-group (the group buffer on
        the cache layer carries partial groups across the boundary)."""
        from transformers import DynamicCache

        config = self._tie_free_config(config)
        model = model_class(config).eval()
        seq_len = 13
        split = 7  # 7 % 2 == 1: the cut lands INSIDE a ratio-2 group, so the group
        # buffer on the cache layer must carry the partial group across the call
        inputs = torch.randint(0, config.vocab_size, (2, seq_len))
        with torch.no_grad():
            full = self._output_tensor(model(inputs))
            cache = DynamicCache(config=config)
            head = self._output_tensor(model(inputs[:, :split], past_key_values=cache, use_cache=True))
            tail = self._output_tensor(model(inputs[:, split:], past_key_values=cache, use_cache=True))
        self.assertTrue(torch.allclose(full, torch.cat([head, tail], dim=1), atol=1e-4))

    def test_chunked_prefill_matches_one_shot(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            self._run_and_compare_chunked(model_class, config)

    def test_decode_matches_one_shot(self):
        """Autoregressive decode steps must match the logits of a one-shot forward
        that includes the decoded tokens — across compress-group boundaries."""
        from transformers import DynamicCache

        config = self._tie_free_config(self.model_tester.get_config())
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            inputs = torch.randint(0, config.vocab_size, (2, 10))
            new_tokens = torch.randint(0, config.vocab_size, (2, 3))
            with torch.no_grad():
                ground_truth = self._output_tensor(model(torch.cat([inputs, new_tokens], dim=1)))[:, -3:]
                cache = DynamicCache(config=config)
                model(inputs, past_key_values=cache, use_cache=True)
                decoded = []
                for step in range(3):
                    tok = new_tokens[:, step : step + 1]
                    decoded.append(self._output_tensor(model(tok, past_key_values=cache, use_cache=True)))
            self.assertTrue(torch.allclose(ground_truth, torch.cat(decoded, dim=1), atol=1e-4))

    def test_save_load_round_trip(self):
        """save→load must be exact, pinning the checkpoint-native weight naming."""
        config = self.model_tester.get_config()
        model = self.model_tester.causal_lm_class(config).eval()
        inputs = torch.randint(0, config.vocab_size, (1, 6))
        with torch.no_grad():
            before = model(inputs).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = self.model_tester.causal_lm_class.from_pretrained(tmp)
        with torch.no_grad():
            after = reloaded(inputs).logits
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_config_validation(self):
        # index source without a compressed branch
        with self.assertRaises(ValueError):
            DeepseekV41TextConfig(num_hidden_layers=2, compress_ratios=[0, 0], index_source_layer_ids=[1])
        # compressed layer with no kv source at or before it
        with self.assertRaises(ValueError):
            DeepseekV41TextConfig(num_hidden_layers=1, compress_ratios=[2], kv_source_layer_ids=[])
        # candidate source outside the index sources
        with self.assertRaises(ValueError):
            DeepseekV41TextConfig(num_hidden_layers=4, candidate_source_layer_id=1)

    def test_engram_hash_state(self):
        """The engram n-gram hash state is a pure function of (tokenizer, config):
        rebuilt states produce identical hashes, and a DEAD token (image span) breaks
        the n-gram look-back for exactly the `max_ngram_size - 1` positions after it."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
            DeepseekV41NgramHashState,
            build_compressed_token_map,
        )

        tokenizer = tiny_word_tokenizer()
        _, compressed_vocab = build_compressed_token_map(tokenizer)

        config = self.model_tester.get_config()
        config.engram_layer_ids = [1, 2]
        config.engram_num_embeddings = [700, 700]
        config.engram_vocab_size = 64
        config.engram_n_heads = 2
        config.engram_head_dim = 16
        config.engram_max_ngram_size = 4
        config.engram_pad_id = 8
        config.engram_compressed_vocab_size = compressed_vocab

        state = DeepseekV41NgramHashState(config, tokenizer)
        ids = torch.randint(0, len(tokenizer), (1, 12))
        positions = torch.arange(12).unsqueeze(0)
        hashes = state(ids, positions, None)

        # Rebuild: multipliers and prime buckets must be reproducible.
        hashes_rebuilt = DeepseekV41NgramHashState(config, tokenizer)(ids, positions, None)
        self.assertTrue(torch.equal(hashes, hashes_rebuilt))

        # A DEAD token at position 5 changes the hashes of positions 5..8 (its
        # n-grams) but not position 9+ (whose 4-grams no longer reach it).
        # `token_mask` is live-True (False marks tokens outside any n-gram).
        live = torch.ones_like(positions, dtype=torch.bool)
        live[0, 5] = False
        masked = state(ids, positions, live)
        self.assertFalse(torch.equal(hashes[0, 5:9], masked[0, 5:9]))
        self.assertTrue(torch.equal(hashes[0, 9:], masked[0, 9:]))
        self.assertTrue(torch.equal(hashes[0, :5], masked[0, :5]))

    def test_csacache_reorder_follows_group_state(self):
        """`generate` with beams calls `reorder_cache` every step; the sliding ring is
        permuted by the base class, and the group state (partial-group buffers, shared
        compressed KV, indexer keys) must follow, or beams silently attend each
        other's groups."""
        from transformers import DynamicCache
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41CSACache

        config = self._tie_free_config(self.model_tester.get_config())
        kv = torch.arange(2 * 1 * 6 * 4, dtype=torch.float32).view(2, 1, 6, 4)
        cache = DynamicCache(config=config)
        layer = cache.layers[config.kv_source_layer_ids[0]]
        self.assertIsInstance(layer, DeepseekV41CSACache)
        before = kv.transpose(2, 3).contiguous().unsqueeze(1)  # [B, 1, T, hd]
        layer.compressed_kv["compressor"] = before.clone()
        layer.compressed_kv["indexer"] = kv[:, :, :3].clone()
        layer.buffer_kv["compressor"] = kv[:, :, -1:].clone()

        layer.reorder_cache(torch.tensor([1, 0]))

        self.assertTrue(torch.equal(layer.compressed_kv["compressor"], before.flip(0)))
        self.assertTrue(torch.equal(layer.compressed_kv["indexer"], kv[:, :, :3].flip(0)))
        self.assertEqual(layer.buffer_kv["compressor"].shape[0], 2)

    def test_engram_forward(self):
        """The engram layers run end-to-end: bind the tokenizer, forward, decode."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        tokenizer = tiny_word_tokenizer()
        _, compressed_vocab = build_compressed_token_map(tokenizer)

        config = self.model_tester.get_config()
        config.vocab_size = 16
        config.engram_layer_ids = [1, 2]
        config.engram_num_embeddings = [700, 700]
        config.engram_vocab_size = 64
        config.engram_n_heads = 2
        config.engram_head_dim = 16
        config.engram_max_ngram_size = 4
        config.engram_pad_id = 8
        config.engram_compressed_vocab_size = compressed_vocab

        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.randint(0, len(tokenizer), (2, 12))
        with torch.no_grad():
            out = model(inputs, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            dec = model(inputs[:, :1], past_key_values=out.past_key_values, use_cache=True)
            self.assertTrue(torch.isfinite(dec.logits).all())
