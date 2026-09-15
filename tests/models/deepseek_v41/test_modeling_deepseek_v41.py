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
import os
import re
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, require_torch_multi_accelerator, slow


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DeepseekV41Config,
        DeepseekV41ForCausalLM,
        DeepseekV41ForConditionalGeneration,
        DeepseekV41Model,
        DeepseekV41TextConfig,
        DeepseekV41TextModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, torch_device


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
        # 32 keeps the indexer head dim divisible by the QAT fp4 block size, so the
        # tests exercise the reference's quantized indexer path.
        self.index_head_dim = 32
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

    @unittest.skip(
        "The top-level renames (`^embed\\.weight$`, `^head\\.weight$`, ...) are anchored at `^` so they cannot "
        "match `layers.N.engram.embed.weight`; the reverse check applies them to `model.`-prefixed "
        "serialized keys, which the anchor rejects by design (same situation as deepseek_v4). The real "
        "round trip is covered by `test_native_checkpoint_names_load` and `test_save_load_round_trip`."
    )
    def test_reverse_loading_mapping(self):
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
        often enough for top-k ties to make the selection order-dependent. This config
        selects every visible block instead, keeping the comparison
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
        # Seeded: an unlucky draw can land a router/indexer top-k tie exactly on the
        # chunk boundary, flipping a pick between the two paths (observed ~1/10 runs).
        torch.manual_seed(0)
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
            torch.manual_seed(0)  # see _run_and_compare_chunked
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

    def test_cached_and_uncached_forward_match(self):
        """KV sources must replace each other's per-forward state, including empty groups."""
        for ratios, seq_len in (([0, 2, 2, 1, 1], 13), ([0, 1, 1, 2, 2], 1)):
            config = self._tie_free_config(self.model_tester.get_config())
            config.compress_ratios = ratios
            for model_class in self.all_model_classes:
                with self.subTest(model_class=model_class.__name__, ratios=ratios):
                    torch.manual_seed(0)
                    model = model_class(config).eval()
                    inputs = torch.randint(0, config.vocab_size, (2, seq_len))
                    with torch.no_grad():
                        cached = self._output_tensor(model(inputs, use_cache=True))
                        uncached = self._output_tensor(model(inputs, use_cache=False))
                    torch.testing.assert_close(uncached, cached, atol=1e-4, rtol=1e-4)

    def test_compressed_attention_ignores_padding(self):
        config = self._tie_free_config(self.model_tester.get_config())
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            model = model_class(config).eval()
            inputs = torch.randint(1, config.vocab_size, (2, 13))
            mask = torch.ones_like(inputs)
            mask[0, :7] = 0
            mask[1, :1] = 0
            changed_padding = inputs.clone()
            changed_padding[mask == 0] = (changed_padding[mask == 0] + 1) % config.vocab_size
            for use_cache in (False, True):
                with self.subTest(model=model_class.__name__, use_cache=use_cache), torch.no_grad():
                    padded = self._output_tensor(model(inputs, attention_mask=mask, use_cache=use_cache))
                    changed = self._output_tensor(model(changed_padding, attention_mask=mask, use_cache=use_cache))
                    torch.testing.assert_close(padded[mask.bool()], changed[mask.bool()], atol=1e-5, rtol=1e-5)
                    for row, start in ((0, 7), (1, 1)):
                        unpadded = self._output_tensor(model(inputs[row : row + 1, start:], use_cache=use_cache))
                        torch.testing.assert_close(padded[row : row + 1, start:], unpadded, atol=1e-4, rtol=1e-4)

    def test_padded_chunked_decode_and_beam_state(self):
        from transformers import DynamicCache

        config = self._tie_free_config(self.model_tester.get_config())
        torch.manual_seed(0)
        model = self.model_tester.causal_lm_class(config).eval()
        inputs = torch.randint(1, config.vocab_size, (2, 13))
        mask = torch.ones_like(inputs)
        mask[0, :7] = 0
        mask[1, :1] = 0
        order = torch.tensor([1, 0, 1])
        new_tokens = torch.randint(1, config.vocab_size, (3, 3))
        with torch.no_grad():
            for operation in ("reorder", "repeat_select"):
                with self.subTest(operation=operation):
                    cache = DynamicCache(config=config)
                    # One row contains no live token in the first chunk; the other
                    # has a different partial-group boundary.
                    first = model(inputs[:, :5], attention_mask=mask[:, :5], past_key_values=cache).logits
                    second = model(inputs[:, 5:], attention_mask=mask, past_key_values=cache).logits
                    full = model(inputs, attention_mask=mask, use_cache=False).logits
                    joined = torch.cat([first, second], dim=1)
                    torch.testing.assert_close(joined[mask.bool()], full[mask.bool()], atol=1e-4, rtol=1e-4)
                    if operation == "reorder":
                        cache.reorder_cache(order)
                    else:
                        cache.batch_repeat_interleave(2)
                        cache.batch_select_indices(torch.tensor([2, 0, 3]))
                    continued_mask = torch.cat([mask[order], torch.ones_like(new_tokens)], dim=-1)
                    actual = model(new_tokens, attention_mask=continued_mask, past_key_values=cache).logits
                    for row, source_row in enumerate(order.tolist()):
                        prefix = inputs[source_row][mask[source_row].bool()].unsqueeze(0)
                        expected = model(
                            torch.cat([prefix, new_tokens[row : row + 1]], dim=-1), use_cache=False
                        ).logits[:, -3:]
                        torch.testing.assert_close(actual[row : row + 1], expected, atol=1e-4, rtol=1e-4)

    def test_precomputed_mask_preserves_compressed_token_liveness(self):
        config = self._tie_free_config(self.model_tester.get_config())
        torch.manual_seed(0)
        model = self.model_tester.causal_lm_class(config).eval()
        inputs = torch.randint(1, config.vocab_size, (2, 9))
        mask = torch.ones_like(inputs)
        mask[0, :3] = 0
        mask[1] = 0
        indices = torch.arange(inputs.shape[1])
        causal = (indices[:, None] >= indices) & (indices[:, None] - indices < config.sliding_window)
        visible = causal[None, None] & mask[:, None, None, :].bool()
        additive = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        with torch.no_grad():
            expected = model(inputs, attention_mask=mask, use_cache=False).logits
            for prepared in (additive, {"sliding_attention": additive}, visible):
                actual = model(inputs, attention_mask=prepared, use_cache=False).logits
                torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-4, rtol=1e-4)
                self.assertTrue(torch.isfinite(actual).all())

    def test_save_load_round_trip(self):
        """save→load must be exact."""
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

    def _native_state_dict(self, model, config):
        """The model's weights under the released checkpoint's names: DeepSeek-native
        module names, no `model.` prefix, per-expert `w1` (gate) / `w3` (up) / `w2`."""
        inter = config.moe_intermediate_size
        native = {}
        for name, tensor in model.state_dict().items():
            native_name = self._to_native_name(name)
            if name.endswith("mlp.experts.gate_up_proj"):
                for e, weight in enumerate(tensor):
                    base = native_name[: -len("gate_up_proj")]
                    native[f"{base}{e}.w1.weight"] = weight[:inter].contiguous()
                    native[f"{base}{e}.w3.weight"] = weight[inter:].contiguous()
            elif name.endswith("mlp.experts.down_proj"):
                for e, weight in enumerate(tensor):
                    native[f"{native_name[: -len('down_proj')]}{e}.w2.weight"] = weight.contiguous()
            else:
                native[native_name] = tensor.contiguous()
        return native

    def test_native_checkpoint_names_load(self):
        """A checkpoint in the released naming (`layers.0.attn.wq_a`, `ffn.experts.E.w1`,
        raw `hc_attn_fn`, top-level `embed` / `head`, `gate.bias_vl`, ...) loads into the
        HF-named modules through the `deepseek_v41_text` conversion mapping with no
        missing / unexpected keys, for the CausalLM wrapper AND the bare text model, and
        reproduces the source model's logits exactly."""
        from safetensors.torch import save_file

        config = self.model_tester.get_config()
        model = self.model_tester.causal_lm_class(config).eval()
        native = self._native_state_dict(model, config)
        self.assertIn("layers.0.attn.wq_a.weight", native)
        self.assertIn("layers.0.ffn.experts.3.w2.weight", native)
        self.assertIn("layers.0.ffn.gate.bias_vl", native)
        self.assertIn("layers.0.hc_attn_fn", native)
        self.assertIn("layers.1.attn.compressor.wgate.weight", native)
        self.assertFalse({k for k in native if k.startswith("model.") or "self_attn" in k or "_hc." in k})
        # the released file also carries the DSpark draft layers, the vision tower, the
        # aligner and the image delimiters: still ignored once the renames rewrite them
        native.update(
            {
                "mtp.0.attn.wq_a.weight": torch.zeros(4, 4),
                "mtp.0.ffn.experts.0.w1.weight": torch.zeros(4, 4),
                "mtp.0.hc_attn_fn": torch.zeros(3, 4),
                "vision.blocks.0.attn.wo.weight": torch.zeros(4, 4),
                "aligner.w1.weight": torch.zeros(4, 4),
                "image_start": torch.zeros(4),
            }
        )

        inputs = torch.randint(0, config.vocab_size, (1, 6))
        with tempfile.TemporaryDirectory() as tmp:
            save_file(native, os.path.join(tmp, "model.safetensors"))
            config.save_pretrained(tmp)
            loaded, info = self.model_tester.causal_lm_class.from_pretrained(tmp, output_loading_info=True)
            self.assertFalse({k: v for k, v in info.items() if v}, info)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(inputs).logits, loaded(inputs).logits))

            # the bare text model reads the same file (the `model.` prefix is the loader's)
            text_model, info = self.model_tester.base_model_class.from_pretrained(tmp, output_loading_info=True)
            self.assertEqual(set(info["unexpected_keys"]), {"lm_head.weight"}, info)
            self.assertFalse(info["missing_keys"], info)
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(model.model(inputs).last_hidden_state, text_model(inputs).last_hidden_state)
                )

    def test_router_logits_and_aux_loss(self):
        """`output_router_logits` records one pre-activation logit tensor per layer (the
        gate's `[tokens, n_routed_experts]` output, not the top-k weights) and turns on
        the Mixtral load-balancing aux loss, which is added to the LM loss scaled by
        `router_aux_loss_coef`."""
        config = self.model_tester.get_config()
        model = self.model_tester.causal_lm_class(config).eval()
        inputs = torch.randint(0, config.vocab_size, (2, 7))
        with torch.no_grad():
            plain = model(inputs, labels=inputs)
            out = model(inputs, labels=inputs, output_router_logits=True)
        self.assertIsNone(plain.aux_loss)
        self.assertEqual(len(out.router_logits), config.num_hidden_layers)
        for layer_logits in out.router_logits:
            self.assertEqual(tuple(layer_logits.shape), (2 * 7, config.n_routed_experts))
        self.assertTrue(torch.isfinite(out.aux_loss))
        self.assertGreater(out.aux_loss.item(), 0.0)
        self.assertTrue(torch.allclose(out.loss, plain.loss + config.router_aux_loss_coef * out.aux_loss))

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

    def _engram_config(self, tokenizer, layer_ids=(1, 2)):
        """The tester config with tiny engram tables on `layer_ids`, hashed through
        `tokenizer`. `vocab_size` is pinned to the tokenizer: `generate` samples the
        full vocab and the hash state's token map only covers the tokenizer."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        _, compressed_vocab = build_compressed_token_map(tokenizer)
        config = self.model_tester.get_config()
        config.vocab_size = len(tokenizer)
        config.engram_layer_ids = list(layer_ids)
        config.engram_num_embeddings = [700] * len(layer_ids)
        config.engram_vocab_size = 64
        config.engram_n_heads = 2
        config.engram_head_dim = 16
        config.engram_max_ngram_size = 4
        config.engram_pad_id = 8
        config.engram_compressed_vocab_size = compressed_vocab
        return config

    def test_engram_hash_state(self):
        """The engram n-gram hash state is a pure function of (tokenizer, config):
        rebuilt states produce identical hashes, and a DEAD token (image span) breaks
        the n-gram look-back for exactly the `max_ngram_size - 1` positions after it."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41NgramHashState

        tokenizer = tiny_word_tokenizer()
        config = self._engram_config(tokenizer)

        state = DeepseekV41NgramHashState(config)
        state.bind_tokenizer(tokenizer)
        # Non-pad ids only: pad hashes like a blocked look-back, which would let the
        # inequalities below coincide by chance.
        ids = torch.randint(0, len(tokenizer) - 1, (1, 12))
        hashes = state(ids, None, None)

        # Rebuild: multipliers and prime buckets must be reproducible.
        rebuilt = DeepseekV41NgramHashState(config)
        rebuilt.bind_tokenizer(tokenizer)
        self.assertTrue(torch.equal(hashes, rebuilt(ids, None, None)))

        # A DEAD token at position 5 changes the hashes of positions 5..8 (its
        # n-grams) but not position 9+ (whose 4-grams no longer reach it).
        # `token_mask` is live-True (False marks tokens outside any n-gram).
        live = torch.ones_like(ids, dtype=torch.bool)
        live[0, 5] = False
        masked = state(ids, live, None)
        self.assertFalse(torch.equal(hashes[0, 5:9], masked[0, 5:9]))
        self.assertTrue(torch.equal(hashes[0, 9:], masked[0, 9:]))
        self.assertTrue(torch.equal(hashes[0, :5], masked[0, :5]))

        # The look-back crosses forward calls only through the cache: chunked
        # hashing through one must equal the one-shot hashes, and without a cache the
        # second chunk starts from an empty look-back.
        from transformers import DynamicCache

        cache = DynamicCache(config=config)
        chunked = torch.cat([state(ids[:, :7], None, cache), state(ids[:, 7:], None, cache)], dim=1)
        self.assertTrue(torch.equal(hashes, chunked))
        self.assertFalse(torch.equal(hashes[:, 7:], state(ids[:, 7:], None, None)))

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

    def test_csa_crop_refusal_preserves_continuation(self):
        """Refused rollback must leave CSA and preceding sliding layers usable."""
        from transformers import DynamicCache

        tokenizer = tiny_word_tokenizer()
        config = self._tie_free_config(self._engram_config(tokenizer))
        torch.manual_seed(0)
        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]])
        next_tokens = torch.tensor([[7, 0, 1], [1, 0, 7]])
        with torch.no_grad():
            expected = model(torch.cat([inputs, next_tokens], dim=1), use_cache=False).logits[:, -3:]
            for record_past in (False, True):
                for layer_only in (False, True):
                    with self.subTest(record_past=record_past, layer_only=layer_only):
                        cache = DynamicCache(config=config)
                        if record_past:
                            cache.activate_past_recording()
                        model(inputs, past_key_values=cache, use_cache=True)
                        target = cache.layers[config.kv_source_layer_ids[0]] if layer_only else cache
                        for tokens_to_remove in (-1, inputs.shape[1] - 1):
                            with self.assertRaises(RuntimeError):
                                target.crop(tokens_to_remove)
                            self.assertEqual(
                                [layer.get_seq_length() for layer in cache.layers],
                                [inputs.shape[1]] * config.num_hidden_layers,
                            )
                        self.assertFalse(cache.is_croppable)
                        actual = model(next_tokens, past_key_values=cache, use_cache=True).logits
                        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_csa_zero_crop_preserves_continuation(self):
        """Trim-only crop preserves compressed groups and bounded engram history."""
        from transformers import DynamicCache

        tokenizer = tiny_word_tokenizer()
        config = self._tie_free_config(self._engram_config(tokenizer))
        torch.manual_seed(0)
        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        cache = DynamicCache(config=config)
        source = cache.layers[config.kv_source_layer_ids[0]]
        source.crop(0)
        self.assertEqual(source.get_seq_length(), 0)
        for tokens_to_remove in (-1, 1):
            with self.assertRaises(RuntimeError):
                source.crop(tokens_to_remove)
            self.assertEqual(source.get_seq_length(), 0)
        cache.activate_past_recording()
        inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]])
        next_tokens = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            expected = model(torch.cat([inputs, next_tokens], dim=1), use_cache=False).logits[:, -3:]
            model(inputs, past_key_values=cache, use_cache=True)
            cache.crop(0)
            self.assertEqual(
                [layer.get_seq_length() for layer in cache.layers],
                [inputs.shape[1]] * config.num_hidden_layers,
            )
            self.assertEqual(source.engram_context.shape[1], config.engram_max_ngram_size - 1)
            self.assertEqual(cache.layers[0].keys.shape[-2], config.sliding_window - 1)
            actual = model(next_tokens, past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_beam_reorder_follows_engram_history(self):
        """The engram n-gram look-back lives on the cache: `cache.reorder_cache` alone
        must permute it (beam search has no model-side hook), and a decode step after
        the reorder must match a one-shot forward of the reordered sequences (found by
        the cross-engine audit — the CSA-layer reorder tests ran with engram disabled
        and missed the history)."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41EngramHistoryLayer

        tokenizer = tiny_word_tokenizer()
        config = self._tie_free_config(self._engram_config(tokenizer))
        torch.manual_seed(0)
        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.randint(0, len(tokenizer) - 1, (2, 9))
        # Distinct look-backs per row, so a missed permutation is visible.
        inputs[1, -1] = (inputs[0, -1] + 1) % (len(tokenizer) - 1)
        next_tokens = torch.randint(0, len(tokenizer) - 1, (2, 1))
        beam_idx = torch.tensor([1, 0])
        with torch.no_grad():
            cache = model(inputs, use_cache=True).past_key_values
            layer = next(layer for layer in cache.layers if isinstance(layer, DeepseekV41EngramHistoryLayer))
            history = layer.engram_context.clone()
            self.assertEqual(tuple(history.shape), (2, config.engram_max_ngram_size - 1))

            cache.reorder_cache(beam_idx)
            self.assertTrue(torch.equal(layer.engram_context, history[beam_idx]))

            decoded = model(next_tokens, past_key_values=cache, use_cache=True).logits
            one_shot = model(torch.cat([inputs[beam_idx], next_tokens], dim=1)).logits[:, -1:]
        self.assertTrue(torch.allclose(decoded, one_shot, atol=1e-4))

        # End to end: beam search with engram layers enabled runs through the plain
        # cache reorder path (`min_new_tokens` so a random-weight EOS cannot cut it short).
        beams = model.generate(inputs, max_new_tokens=4, min_new_tokens=4, num_beams=2, do_sample=False)
        self.assertEqual(tuple(beams.shape), (2, 13))

    @staticmethod
    def _to_native_name(name):
        """HF parameter name → released-checkpoint name (the reverse of the
        `deepseek_v41_text` conversion mapping, minus the expert merge)."""
        for hf, native in (
            ("model.layers.", "layers."),
            ("model.embed_tokens.", "embed."),
            ("model.norm.", "norm."),
            ("lm_head.", "head."),
            (".self_attn.", ".attn."),
            (".mlp.", ".ffn."),
            (".input_layernorm.", ".attn_norm."),
            (".post_attention_layernorm.", ".ffn_norm."),
            (".attn_hc.fn", ".hc_attn_fn"),
            (".attn_hc.base", ".hc_attn_base"),
            (".attn_hc.scale", ".hc_attn_scale"),
            (".ffn_hc.fn", ".hc_ffn_fn"),
            (".ffn_hc.base", ".hc_ffn_base"),
            (".ffn_hc.scale", ".hc_ffn_scale"),
            (".attn.sinks", ".attn.attn_sink"),
            (".q_a_proj.", ".wq_a."),
            (".q_a_norm.", ".q_norm."),
            (".q_b_proj.", ".wq_b."),
            (".attn.kv_proj.", ".attn.wkv."),
            (".compressor.kv_proj.", ".compressor.wkv."),
            (".compressor.gate_proj.", ".compressor.wgate."),
            (".compressor.kv_norm.", ".compressor.norm."),
            (".indexer.k_proj.", ".indexer.wk."),
            (".o_a_proj.", ".wo_a."),
            (".o_b_proj.", ".wo_b."),
            (".gate.e_score_correction_bias_vl", ".gate.bias_vl"),
            (".gate.e_score_correction_bias", ".gate.bias"),
            (".shared_experts.gate_proj.", ".shared_experts.w1."),
            (".shared_experts.up_proj.", ".shared_experts.w3."),
            (".shared_experts.down_proj.", ".shared_experts.w2."),
        ):
            name = name.replace(hf, native)
        # model-level `engram_tables.<layer>.*` <- `layers.<layer>.engram.embed.*`
        return re.sub(r"^model\.engram_tables\.(\d+)\.", r"layers.\1.engram.embed.", name)

    def test_fp8_native_checkpoint_load(self):
        """Load a native-format quantized checkpoint replicating the released layout AND
        names: fp8 e4m3 weights + ue8m0 block scales for attention / shared experts /
        engram.wkv, PACKED FP4 routed experts (per expert `w1` / `w2` / `w3`: e2m1
        nibbles in int8, per-row 32-channel ue8m0 scales — the released MXFP4 [1, 32]
        block), fp8 engram tables, BF16 compressor / indexer / embed, F32 mHC params and
        gate biases. The `wo_a` grouped projection and the engram tables must survive
        the load, and the per-expert fp4 tensors must land exactly in the fused
        `gate_up_proj` / `down_proj`, for both the wrapper and bare text backbone."""
        from safetensors.torch import save_file

        from transformers import FineGrainedFP8Config
        from transformers.models.deepseek_v41 import DeepseekV41Config
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        def quantize_fp8(weight, block=32):
            out_dim, in_dim = weight.shape
            blocks = weight.float().view(out_dim // block, block, in_dim // block, block)
            amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
            scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))  # ue8m0: powers of two
            q = (blocks / scale.unsqueeze(1).unsqueeze(3)).clamp(-448, 448)
            deq = q.to(torch.float8_e4m3fn).float() * scale.unsqueeze(1).unsqueeze(3)
            return (
                q.reshape(out_dim, in_dim).to(torch.float8_e4m3fn),
                scale.to(torch.float8_e8m0fnu),  # [out_blocks, in_blocks] like the release
                deq.reshape(out_dim, in_dim),
            )

        def quantize_fp4(weight):
            # packed fp4 like the release — two e2m1 nibbles per int8 byte (even index in
            # the low nibble), one ue8m0 scale per row per 32 fp4 channels (the released
            # MXFP4 [1, 32] block: scales [2304, 160] over an unpacked in-dim of 5120)
            out_dim, in_dim = weight.shape
            groups = weight.float().view(out_dim, in_dim // 32, 32)
            amax = groups.abs().amax(-1).clamp_min(1e-4)
            scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
            q = (groups / scale.unsqueeze(-1)).clamp(-6, 6)
            grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            codes = torch.bucketize(q.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
            values = grid[codes] * torch.where(q < 0, -1.0, 1.0)
            nibbles = codes.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
            packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
            return (
                packed.reshape(out_dim, in_dim // 2).to(torch.int8),
                scale.to(torch.float8_e8m0fnu),
                (values * scale.unsqueeze(-1)).reshape(out_dim, in_dim),
            )

        tokenizer = tiny_word_tokenizer()
        _, compressed_vocab = build_compressed_token_map(tokenizer)
        config = self.model_tester.get_config()
        config.engram_layer_ids = [1]
        config.engram_num_embeddings = [700]
        config.engram_vocab_size = 64
        config.engram_n_heads = 2
        config.engram_head_dim = 64  # divisible by the engram table's 32-channel groups
        config.engram_pad_id = 8
        config.engram_compressed_vocab_size = compressed_vocab

        model = self.model_tester.causal_lm_class(config).to(torch.bfloat16).eval()
        model.model.bind_tokenizer(tokenizer)
        inter = config.moe_intermediate_size
        # the released checkpoint's quantized set: attention projections (incl. the
        # grouped wo_a), indexer.wq_b, engram.wkv and the SHARED experts are fp8-block;
        # the ROUTED experts are packed fp4. Compressor, indexer.wk / weights_proj, gate,
        # embed and head stay bf16; mHC / sinks / gate biases f32.
        fp8_suffixes = (
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_proj.weight",
            "self_attn.o_a_proj.weight",
            "self_attn.o_b_proj.weight",
            "indexer.q_b_proj.weight",
            "engram.wkv.weight",
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        )
        f32_leaves = ("fn", "base", "scale", "sinks", "e_score_correction_bias", "e_score_correction_bias_vl")
        native, dequantized = {}, {}
        for name, tensor in model.state_dict().items():
            native_name = self._to_native_name(name)
            if name.endswith("mlp.experts.gate_up_proj"):
                fused = []
                for e, weight in enumerate(tensor):
                    halves = []
                    for w_name, half in (("w1", weight[:inter]), ("w3", weight[inter:])):
                        packed, scale, deq = quantize_fp4(half)
                        base = native_name[: -len("gate_up_proj")] + f"{e}.{w_name}"
                        native[base + ".weight"], native[base + ".scale"] = packed, scale
                        halves.append(deq)
                    fused.append(torch.cat(halves, dim=0))
                dequantized[name] = torch.stack(fused).to(torch.bfloat16)
            elif name.endswith("mlp.experts.down_proj"):
                fused = []
                for e, weight in enumerate(tensor):
                    packed, scale, deq = quantize_fp4(weight)
                    base = native_name[: -len("down_proj")] + f"{e}.w2"
                    native[base + ".weight"], native[base + ".scale"] = packed, scale
                    fused.append(deq)
                dequantized[name] = torch.stack(fused).to(torch.bfloat16)
            elif "engram_tables" in name and name.endswith(".weight"):
                rows, dim = tensor.shape
                blocks = tensor.float().view(rows, dim // 32, 32)
                amax = blocks.abs().amax(-1).clamp_min(1e-4)
                scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
                q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448)
                native[native_name] = q.reshape(rows, dim).to(torch.float8_e4m3fn)
                native[native_name[: -len(".weight")] + ".scale"] = scale.to(torch.float8_e8m0fnu)
                dequantized[name] = (
                    (q.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(rows, dim).to(torch.bfloat16)
                )
            elif "engram_tables" in name and name.endswith(".weight_scale_inv"):
                continue  # the `.scale` sibling is written by the branch above (the quantizer renames it)
            elif name.endswith(fp8_suffixes):
                weight, scale, deq = quantize_fp8(tensor)
                # the checkpoint keeps the param's own name for the fp8 tensor and
                # stores the block scales as a `.scale` SIBLING of it
                native[native_name] = weight
                native[native_name[: -len(".weight")] + ".scale"] = scale
                dequantized[name] = deq.to(torch.bfloat16)
            elif name.split(".")[-1] in f32_leaves or name.endswith(("q_weight", "k_weight")):
                native[native_name] = tensor.to(torch.float32)
            else:
                native[native_name] = tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor
        # the file must be in the released naming: no HF names leak through
        self.assertFalse({k for k in native if "self_attn" in k or "gate_up_proj" in k or k.startswith("model.")})
        self.assertIn("layers.0.ffn.experts.0.w1.scale", native)
        self.assertIn("layers.0.hc_attn_fn", native)
        self.assertIn("layers.0.ffn.gate.bias_vl", native)

        with tempfile.TemporaryDirectory() as tmp:
            save_file(native, os.path.join(tmp, "model.safetensors"))
            composite = DeepseekV41Config(text_config=config.to_dict())
            # Keep the storage metadata exclusively at the top level, as in the
            # release, rather than serializing the constructor's hoisted expert_dtype.
            composite.quantization_config = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
            composite.save_pretrained(tmp)
            loaded = self.model_tester.causal_lm_class.from_pretrained(
                tmp, dtype=torch.bfloat16, quantization_config=FineGrainedFP8Config(dequantize=True)
            )
            loaded.model.bind_tokenizer(tokenizer)
            text_model, info = self.model_tester.base_model_class.from_pretrained(
                tmp,
                dtype=torch.bfloat16,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                use_cache=False,
                output_loading_info=True,
            )
            text_model.bind_tokenizer(tokenizer)
            self.assertEqual(set(info["unexpected_keys"]), {"lm_head.weight"}, info)
            self.assertFalse(info["missing_keys"], info)

        # 1. untouched tensors load verbatim (BF16/F32 modules survive the fp8 path);
        #    the gate biases come back as the fp32 buffers they are
        ref = model.state_dict()
        got = loaded.state_dict()
        for name in (
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.attn_hc.fn",
            "model.layers.1.self_attn.compressor.gate_proj.weight",
            "model.layers.0.mlp.gate.e_score_correction_bias_vl",
        ):
            self.assertTrue(torch.allclose(ref[name].float(), got[name].float(), atol=1e-3), name)
        self.assertEqual(got["model.layers.0.mlp.gate.e_score_correction_bias"].dtype, torch.float32)
        self.assertEqual(got["model.layers.0.attn_hc.fn"].dtype, torch.float32)
        # 2. fp8 tensors dequantize to exactly what we packed (scales applied once)
        for name, deq in dequantized.items():
            self.assertTrue(torch.allclose(deq.float(), got[name].float(), atol=5e-2), name)
        # 3. the fused experts: every per-expert fp4 tensor lands exactly in its slice
        for layer in range(config.num_hidden_layers):
            for proj in ("gate_up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{proj}"
                self.assertEqual(got[name].shape, ref[name].shape, name)
                self.assertTrue(torch.equal(got[name], dequantized[name]), name)
        # 4. the grouped o_a_proj specifically: dequantized and reshapeable
        o_a_proj = got["model.layers.0.self_attn.o_a_proj.weight"]
        self.assertEqual(o_a_proj.dtype, torch.bfloat16)
        o_a_proj.view(config.o_groups, -1, config.hidden_size)
        # 5. end-to-end: fp8-loaded logits track the bf16 original
        # stay inside the tiny tokenizer's range: the engram hashes map ids through
        # build_compressed_token_map, whose lookup has one row per tokenizer entry
        inputs = torch.randint(0, len(tokenizer), (2, 9))
        with torch.no_grad():
            clean = model(inputs).logits.float()
            quant = loaded(inputs).logits.float()
        self.assertLess((clean - quant).abs().max().item(), 0.35)
        # The bare backbone must detect quantization before extracting text_config.
        text_weights = text_model.state_dict()
        for name, deq in dequantized.items():
            torch.testing.assert_close(text_weights[name.removeprefix("model.")], deq, atol=5e-2, rtol=0)
        reference_state = {
            name.removeprefix("model."): dequantized.get(name, tensor)
            for name, tensor in ref.items()
            if name.startswith("model.")
        }
        reference = self.model_tester.base_model_class.from_pretrained(
            None, config=config, state_dict=reference_state, dtype=torch.bfloat16
        )
        reference.bind_tokenizer(tokenizer)
        with torch.no_grad():
            expected = reference(inputs, use_cache=False).last_hidden_state
            text_output = text_model(inputs)
        torch.testing.assert_close(text_output.last_hidden_state, expected, atol=1e-4, rtol=1e-4)
        self.assertIsNone(text_output.past_key_values)

    def test_engram_forward(self):
        """The engram layers run end-to-end: bind the tokenizer, forward, decode; an
        unbound model refuses to hash instead of hashing garbage."""
        tokenizer = tiny_word_tokenizer()
        config = self._engram_config(tokenizer)

        model = self.model_tester.causal_lm_class(config).eval()
        inputs = torch.randint(0, len(tokenizer), (2, 12))
        with self.assertRaisesRegex(ValueError, "bind_tokenizer"):
            model(inputs)
        model.model.bind_tokenizer(tokenizer)
        with torch.no_grad():
            out = model(inputs, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            dec = model(inputs[:, :1], past_key_values=out.past_key_values, use_cache=True)
            self.assertTrue(torch.isfinite(dec.logits).all())

    def test_engram_chunked_prefill_matches_one_shot(self):
        """With engram layers on, a prompt fed in two chunks must hash — and score —
        exactly like the one-shot prefill: the n-gram look-back of the second chunk's
        first `max_ngram_size - 1` positions comes from the cache."""
        from transformers import DynamicCache

        tokenizer = tiny_word_tokenizer()
        config = self._tie_free_config(self._engram_config(tokenizer, layer_ids=(1,)))
        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.randint(0, len(tokenizer) - 1, (2, 13))
        split = 7
        with torch.no_grad():
            full = model(inputs).logits
            cache = DynamicCache(config=config)
            head = model(inputs[:, :split], past_key_values=cache, use_cache=True).logits
            tail = model(inputs[:, split:], past_key_values=cache, use_cache=True).logits
        self.assertTrue(torch.allclose(full, torch.cat([head, tail], dim=1), atol=1e-4))

    def test_engram_from_pretrained_binds_tokenizer(self):
        """`from_pretrained` binds the checkpoint's own tokenizer after the meta-device
        load (the hash tables are non-persistent buffers, rebuilt by `_init_weights`);
        a checkpoint without a tokenizer loads unbound and says so."""
        tokenizer = tiny_word_tokenizer()
        config = self._engram_config(tokenizer, layer_ids=(1,))
        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.randint(0, len(tokenizer), (2, 9))
        with torch.no_grad():
            before = model(inputs).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            unbound = self.model_tester.causal_lm_class.from_pretrained(tmp)
            self.assertEqual(unbound.model.engram_hash_state.token_map.numel(), 0)
            tokenizer.save_pretrained(tmp)
            loaded = self.model_tester.causal_lm_class.from_pretrained(tmp)
        with torch.no_grad():
            after = loaded(inputs).logits
        self.assertTrue(torch.allclose(before, after, atol=1e-5))


class DeepseekV41VisionText2TextModelTester:
    """Tiny image-text-to-text tester (one image per batch row, its span at the start of
    the sequence), following `Qwen2VLVisionText2TextModelTester`. The ViT patch grid
    (4, 2) is deliberately NOT a multiple of the aligner's 3x3 window, so the aligner's
    zero-padding runs in every forward; `ceil(4/3) x ceil(2/3) = 2 x 1` aligner rows
    make a 6-token image span: [START] + [IMAGE] + [NEWLINE] + [IMAGE] + [NEWLINE] + [END].
    """

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        pad_token_id=0,
        image_token_id=100,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.is_training = is_training
        self.vocab_size = 128
        self.num_hidden_layers = 2
        self.hidden_size = 64
        self.n_vit_h, self.n_vit_w = 4, 2
        self.patch_size = 14
        self.downsample_ratio = 3
        self.n_llm_h, self.n_llm_w = 2, 1
        self.span_length = self.n_llm_h * (self.n_llm_w + 1) + 2
        self.seq_length = seq_length + self.span_length
        self.vision_config = {
            "hidden_size": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": self.patch_size,
            "downsample_ratio": self.downsample_ratio,
        }
        # One compressed source and one consumer exercise multimodal shared KV.
        # Generic embedding tests do not need Engram's tokenizer-dependent state.
        self.text_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": 4,
            "num_hidden_layers": self.num_hidden_layers,
            "compress_ratios": [2, 2],
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "max_position_embeddings": 128,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_groups": 2,
            "o_lora_rank": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "sliding_window": 8,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 3,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 2,
            "candidate_topk_blocks": 4,
            "candidate_block_size": 4,
            "num_nextn_predict_layers": 0,
            "scoring_func": "sqrtsoftplus",
            "gate_temp": 1.0,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "rms_norm_eps": 1e-20,
            "engram_layer_ids": [],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
        }

    def get_config(self):
        return DeepseekV41Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size * self.n_vit_h * self.n_vit_w, self.num_channels * self.patch_size**2], scale=1.0
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, -1] = self.pad_token_id
        attention_mask[:, -1] = 0
        input_ids[:, : self.span_length] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, self.n_vit_h, self.n_vit_w]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class DeepseekV41VisionText2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """Model tester for `DeepseekV41ForConditionalGeneration` (image-text-to-text)."""

    all_model_classes = (DeepseekV41Model, DeepseekV41ForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": DeepseekV41ForConditionalGeneration}
    _is_composite = True

    # Same text-backbone caveats as DeepseekV41ModelTest: the indexer's top-k is
    # non-differentiable, per-layer KV lengths differ, and the engram needs a tokenizer.
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = DeepseekV41VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekV41Config, has_text_modality=False)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = self.model_tester.n_vit_h * self.model_tester.n_vit_w
        inputs = {
            key: value[: batch_size * patches_per_image] if key == "pixel_values" else value[:batch_size]
            for key, value in inputs.items()
        }
        config.text_config.eos_token_id = None
        config.text_config.forced_eos_token_id = None
        config.text_config.pad_token_id = self.model_tester.pad_token_id
        return config, inputs

    @unittest.skip("V4.1's compressed KV cache has different sequence lengths in source and consumer layers.")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("The text backbone is eager-only; SDPA is supported only by the vision tower.")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    @unittest.skip(
        "V4.1's compressor stores group state on custom cache layers, which is not compatible with QuantizedCache."
    )
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("V4.1's per-layer KV lengths differ (window-capped ring + compressed entries).")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(
        "The top-level renames are anchored at `^` so they cannot match nested serialized "
        "keys; the real round trip is covered by `test_native_checkpoint_names_load_vl`."
    )
    def test_reverse_loading_mapping(self):
        pass

    def _check_attentions_for_generate(self, batch_size, attentions, prompt_length, output_length, config, **kwargs):
        # see DeepseekV41ModelTest: per-layer KV lengths differ, check shape invariants only
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), (output_length - prompt_length))
        for iter_attentions in attentions:
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertIsInstance(layer_attention, torch.Tensor)
                self.assertEqual(layer_attention.shape[0], batch_size)

    @property
    def has_attentions(self):
        return False

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """A pixel count that does not match the placeholder spans raises, with a
        message saying what does not match; multi-row batches with one image each run."""
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave every image token in the text
            patches_per_image = self.model_tester.n_vit_h * self.model_tester.n_vit_w
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-patches_per_image:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

    def test_merge_image_embeddings(self):
        """The IMAGE slots take the aligner rows in reading order, the delimiters their
        learned embeddings, and everything else keeps the token embedding."""
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = DeepseekV41Model(config).to(torch_device)
        model.eval()
        tester = self.model_tester
        pixel_values = floats_tensor(
            [tester.batch_size * tester.n_vit_h * tester.n_vit_w, 3 * tester.patch_size**2], scale=1.0
        ).to(torch_device)
        image_grid_thw = torch.tensor([[1, tester.n_vit_h, tester.n_vit_w]] * tester.batch_size, device=torch_device)
        input_ids = torch.randint(
            0, tester.vocab_size, (tester.batch_size, tester.span_length + 3), device=torch_device
        )
        input_ids[input_ids == tester.image_token_id] = tester.pad_token_id
        input_ids[:, : tester.span_length] = tester.image_token_id
        image_features = model.get_image_features(pixel_values, image_grid_thw).pooler_output
        token_embeds = model.language_model.embed_tokens(input_ids)
        merged, image_mask = model.merge_image_embeddings(
            input_ids, token_embeds.clone(), image_features, image_grid_thw
        )

        # every span position is masked, delimiters included
        self.assertTrue(torch.equal(image_mask, input_ids == tester.image_token_id))
        for row in range(tester.batch_size):
            span = merged[row, : tester.span_length]
            # layout: START, IMAGE, NEWLINE, IMAGE, NEWLINE, END
            self.assertTrue(torch.allclose(span[0], model.image_start, atol=1e-6))
            self.assertTrue(torch.allclose(span[1], image_features[row][0], atol=1e-6))
            self.assertTrue(torch.allclose(span[2], model.image_newline, atol=1e-6))
            self.assertTrue(torch.allclose(span[3], image_features[row][1], atol=1e-6))
            self.assertTrue(torch.allclose(span[4], model.image_newline, atol=1e-6))
            self.assertTrue(torch.allclose(span[5], model.image_end, atol=1e-6))
        # text positions untouched
        self.assertTrue(torch.allclose(merged[:, tester.span_length :], token_embeds[:, tester.span_length :]))

    def test_image_mask_threads_to_router_and_engram(self):
        """`image_mask` reaches the MoE router (switching to `e_score_correction_bias_vl`
        on image-span tokens) and, inverted, the engram hash state / gate mask."""
        tokenizer = tiny_word_tokenizer()
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        _, compressed_vocab = build_compressed_token_map(tokenizer)
        config = self.model_tester.get_config()
        config.text_config.engram_layer_ids = [1]
        config.text_config.engram_num_embeddings = [700]
        config.text_config.engram_vocab_size = 64
        config.text_config.engram_n_heads = 2
        config.text_config.engram_head_dim = 16
        config.text_config.engram_max_ngram_size = 4
        config.text_config.engram_pad_id = len(tokenizer) - 1
        config.text_config.engram_compressed_vocab_size = compressed_vocab
        model = DeepseekV41ForConditionalGeneration(config).to(torch_device)
        model.eval()
        model.model.bind_tokenizer(tokenizer)

        # 1. the router: a VL bias that forces expert 0 on image tokens changes the
        #    selection exactly there (the plain bias stays zero)
        gate = model.model.language_model.layers[0].mlp.gate
        with torch.no_grad():
            gate.e_score_correction_bias.zero_()
            gate.e_score_correction_bias_vl.zero_()
            gate.e_score_correction_bias_vl[0] = 100.0
            hidden = torch.randn(2, 6, 1, gate.hidden_dim, device=torch_device)
            image_mask = torch.zeros(2, 6, 1, dtype=torch.bool, device=torch_device)
            image_mask[0, :3] = True
            _, _, plain = gate(hidden.reshape(2, 6, -1), None)
            _, _, masked = gate(hidden.reshape(2, 6, -1), image_mask.reshape(2, 6))
        # the gate flattens to [B * S, top_k]: rows 0-2 are batch 0's masked positions.
        # The VL bias (100 on expert 0) forces their FIRST pick to expert 0; every
        # unmasked position keeps the plain selection.
        self.assertTrue(torch.equal(masked[:3, 0], torch.zeros_like(masked[:3, 0])))
        self.assertTrue(torch.equal(masked[3:6], plain[3:6]))  # batch 0's text rows
        self.assertTrue(torch.equal(masked[6:], plain[6:]))  # the all-text row

        # 2. the engram: the hash state and the gate receive `live & ~image_mask`
        inputs = torch.randint(0, len(tokenizer) - 1, (1, 10), device=torch_device)
        image_mask = torch.zeros(1, 10, dtype=torch.bool, device=torch_device)
        image_mask[0, 4:8] = True
        attention_mask = torch.ones(1, 10, dtype=torch.long, device=torch_device)
        attention_mask[0, 9] = 0  # a pad, DEAD too
        state = model.model.language_model.engram_hash_state
        recorded = {}
        original_forward = state.forward

        def spy(input_ids, token_mask, past_key_values):
            recorded["token_mask"] = token_mask
            return original_forward(input_ids, token_mask, past_key_values)

        state.forward = spy
        recorded_layers = {}
        layer = model.model.language_model.layers[1]
        original_layer_forward = layer.forward

        def layer_spy(hidden_streams, pre_mix, engram_rows, token_mask, *args, **kwargs):
            recorded_layers.setdefault("token_mask", []).append(token_mask)
            return original_layer_forward(hidden_streams, pre_mix, engram_rows, token_mask, *args, **kwargs)

        layer.forward = layer_spy
        try:
            with torch.no_grad():
                model(input_ids=inputs, attention_mask=attention_mask, image_mask=image_mask)
        finally:
            state.forward = original_forward
            layer.forward = original_layer_forward
        expected = attention_mask.bool() & ~image_mask
        self.assertTrue(torch.equal(recorded["token_mask"], expected))
        for token_mask in recorded_layers["token_mask"]:
            self.assertTrue(torch.equal(token_mask, expected))

    def test_image_inputs_require_one_chunk_prefill(self):
        """The reference asserts image spans are prefilled in one chunk (`start_pos == 0`);
        feeding images into a non-empty cache refuses instead of silently mis-scattering."""
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekV41ForConditionalGeneration(config).to(torch_device)
        model.eval()
        input_dict = copy.deepcopy(input_dict)
        with torch.no_grad():
            first = model(
                input_ids=input_dict["input_ids"],
                attention_mask=input_dict["attention_mask"],
                use_cache=True,
            )
        with self.assertRaisesRegex(ValueError, "prefilled in one chunk"):
            model(
                input_ids=input_dict["input_ids"][:, -1:],
                attention_mask=input_dict["attention_mask"],
                pixel_values=input_dict["pixel_values"],
                image_grid_thw=input_dict["image_grid_thw"],
                past_key_values=first.past_key_values,
            )

    @staticmethod
    def _to_native_name_vl(name):
        """HF parameter name -> released-checkpoint name for the VL tree (the reverse of
        the `deepseek_v41` conversion mapping): `model.visual.*` <- `vision.*` (with the
        attention's `qkv`/`proj` under their native `wqkv`/`wo`), `model.aligner.*` <-
        `aligner.*`, `model.language_model.*` <- the text names, `lm_head.weight` <-
        `head.weight`."""
        if name.startswith("model.visual."):
            # the vision tree keeps its native names, except the reused Qwen2-VL
            # attention modules (`qkv` / `proj` are the native `wqkv` / `wo`)
            return "vision." + name[len("model.visual.") :].replace("attn.qkv.", "attn.wqkv.").replace(
                "attn.proj.", "attn.wo."
            )
        if name.startswith("model.aligner."):
            return "aligner." + name[len("model.aligner.") :]
        for param in ("image_start", "image_end", "image_newline"):
            if name == f"model.{param}":
                return param
        if name == "lm_head.weight":
            return "head.weight"
        # the text backbone: the shared text mapping on the language-model subtree
        native = DeepseekV41ModelTest._to_native_name(name.replace("model.language_model.", "model."))
        return native.removeprefix("model.")

    def test_native_checkpoint_names_load_vl(self):
        """A checkpoint in the released naming (`vision.blocks.0.attn.wqkv.weight`,
        `aligner.w1.weight`, top-level `image_start`, the text names, ...) loads into
        `DeepseekV41ForConditionalGeneration` through the `deepseek_v41` conversion
        mapping with no missing / unexpected keys and exact logits."""
        from safetensors.torch import save_file

        config, _ = self.model_tester.prepare_config_and_inputs()
        model = DeepseekV41ForConditionalGeneration(config).to(torch_device)
        model.eval()
        native = {}
        for name, tensor in model.state_dict().items():
            native_name = self._to_native_name_vl(name)
            if name.endswith("mlp.experts.gate_up_proj"):
                inter = config.text_config.moe_intermediate_size
                for e, weight in enumerate(tensor):
                    base = native_name[: -len("gate_up_proj")]
                    native[f"{base}{e}.w1.weight"] = weight[:inter].contiguous()
                    native[f"{base}{e}.w3.weight"] = weight[inter:].contiguous()
            elif name.endswith("mlp.experts.down_proj"):
                for e, weight in enumerate(tensor):
                    native[f"{native_name[: -len('down_proj')]}{e}.w2.weight"] = weight.contiguous()
            else:
                native[native_name] = tensor.contiguous()
        self.assertIn("vision.blocks.0.attn.wqkv.weight", native)
        self.assertIn("vision.blocks.0.attn.wo.weight", native)
        self.assertIn("vision.blocks.0.mlp.w1.weight", native)
        self.assertIn("vision.norm.weight", native)
        self.assertIn("aligner.w1.weight", native)
        self.assertIn("image_start", native)
        self.assertIn("image_newline", native)
        self.assertIn("layers.0.attn.wq_a.weight", native)
        self.assertIn("embed.weight", native)
        self.assertIn("norm.weight", native)
        self.assertIn("head.weight", native)
        self.assertFalse({k for k in native if k.startswith("model.") or "self_attn" in k or "_hc." in k})
        # the released file also carries the DSpark draft layers: still ignored
        native.update({"mtp.0.attn.wq_a.weight": torch.zeros(4, 4)})

        inputs = self.model_tester.prepare_config_and_inputs_for_common()[1]
        with tempfile.TemporaryDirectory() as tmp:
            save_file({k: v.cpu() for k, v in native.items()}, os.path.join(tmp, "model.safetensors"))
            config.save_pretrained(tmp)
            loaded, info = DeepseekV41ForConditionalGeneration.from_pretrained(tmp, output_loading_info=True)
            loaded = loaded.to(torch_device).eval()
            self.assertFalse({k: v for k, v in info.items() if v}, info)
            with torch.no_grad():
                before = model(**inputs).logits
                after = loaded(**inputs).logits
            self.assertTrue(torch.equal(before, after))

            # the bare VL backbone reads the same file (the `model.` prefix is the loader's)
            base, info = DeepseekV41Model.from_pretrained(tmp, output_loading_info=True)
            base = base.to(torch_device).eval()
            self.assertEqual(set(info["unexpected_keys"]), {"lm_head.weight"}, info)
            self.assertFalse(info["missing_keys"], info)
            with torch.no_grad():
                self.assertTrue(torch.equal(model.model(**inputs).last_hidden_state, base(**inputs).last_hidden_state))

    def _heterogeneous_image_inputs(self):
        tester = self.model_tester
        input_ids = torch.full((3, 16), 3, dtype=torch.long, device=torch_device)
        # Zero images; one 1x1 image; two adjacent, unequal images ending the prompt.
        input_ids[1, 12:] = tester.image_token_id
        input_ids[2, 5:] = tester.image_token_id
        grids = torch.tensor([[1, 1, 1], [1, 4, 2], [1, 2, 5]], device=torch_device)
        pixels = floats_tensor([19, 3 * tester.patch_size**2]).to(torch_device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixels,
            "image_grid_thw": grids,
        }

    def test_heterogeneous_image_expansion(self):
        model = DeepseekV41ForConditionalGeneration(self.model_tester.get_config()).to(torch_device).eval()
        inputs = self._heterogeneous_image_inputs()
        with torch.no_grad():
            expected = model(**inputs).logits
            for repeats in (2, [1, 3, 2]):
                for embed_only in (False, True):
                    current = dict(inputs)
                    ids = current.pop("input_ids")
                    if embed_only:
                        current["inputs_embeds"] = model.get_input_embeddings()(ids)
                        ids = None
                    expanded_ids, expanded = model._expand_inputs_for_generation(
                        expand_size=repeats, input_ids=ids, **current
                    )
                    actual = model(input_ids=expanded_ids, **expanded).logits
                    repeat_counts = torch.tensor(
                        [repeats] * 3 if isinstance(repeats, int) else repeats, device=torch_device
                    )
                    torch.testing.assert_close(
                        actual, expected.repeat_interleave(repeat_counts, dim=0), atol=1e-5, rtol=1e-5
                    )

    def test_heterogeneous_beam_generation(self):
        model = DeepseekV41ForConditionalGeneration(self.model_tester.get_config()).to(torch_device).eval()
        inputs = self._heterogeneous_image_inputs()
        options = {
            "num_beams": 3,
            "num_return_sequences": 2,
            "max_new_tokens": 3,
            "do_sample": False,
            "eos_token_id": None,
            "pad_token_id": 0,
            "bad_words_ids": [[self.model_tester.image_token_id]],
        }
        with torch.no_grad():
            batched = model.generate(**inputs, **options)
            individual = []
            for row, image_slice, patch_slice in (
                (0, slice(0, 0), slice(0, 0)),
                (1, slice(0, 1), slice(0, 1)),
                (2, slice(1, 3), slice(1, 19)),
            ):
                current = {
                    key: value[row : row + 1]
                    for key, value in inputs.items()
                    if key in ("input_ids", "attention_mask")
                }
                if row:
                    current["image_grid_thw"] = inputs["image_grid_thw"][image_slice]
                    current["pixel_values"] = inputs["pixel_values"][patch_slice]
                individual.append(model.generate(**current, **options))
            self.assertTrue(torch.equal(batched, torch.cat(individual)))

    def test_adjacent_image_layout_and_feature_outputs(self):
        model = DeepseekV41Model(self.model_tester.get_config()).to(torch_device).eval()
        inputs = self._heterogeneous_image_inputs()
        with torch.no_grad():
            outputs = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            self.assertEqual(outputs.last_hidden_state.shape, (19, self.model_tester.vision_config["hidden_size"]))
            self.assertEqual([part.shape for part in outputs.pooler_output], [(1, 64), (2, 64), (2, 64)])
            original = model.get_input_embeddings()(inputs["input_ids"])
            merged, mask = model.merge_image_embeddings(
                inputs["input_ids"], original, outputs.pooler_output, inputs["image_grid_thw"]
            )
            expected_spans = (
                (1, 12, [model.image_start, outputs.pooler_output[0][0], model.image_newline, model.image_end]),
                (
                    2,
                    5,
                    [
                        model.image_start,
                        outputs.pooler_output[1][0],
                        model.image_newline,
                        outputs.pooler_output[1][1],
                        model.image_newline,
                        model.image_end,
                    ],
                ),
                (
                    2,
                    11,
                    [
                        model.image_start,
                        outputs.pooler_output[2][0],
                        outputs.pooler_output[2][1],
                        model.image_newline,
                        model.image_end,
                    ],
                ),
            )
            for row, start, values in expected_spans:
                torch.testing.assert_close(
                    merged[row, start : start + len(values)], torch.stack(values), atol=0, rtol=0
                )
            torch.testing.assert_close(merged[~mask], original[~mask], atol=0, rtol=0)

    def test_image_prefill_engram_cache_continuation_and_input_variants(self):
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        tokenizer = tiny_word_tokenizer(tuple(f"word{i}" for i in range(127)))
        _, compressed_vocab = build_compressed_token_map(tokenizer)
        config, inputs = self.prepare_config_and_inputs_for_generate(batch_size=1)
        config.text_config.engram_layer_ids = [1]
        config.text_config.engram_num_embeddings = [700]
        config.text_config.engram_vocab_size = 64
        config.text_config.engram_n_heads = 2
        config.text_config.engram_head_dim = 16
        config.text_config.engram_max_ngram_size = 4
        config.text_config.engram_pad_id = 127
        config.text_config.engram_compressed_vocab_size = compressed_vocab
        model = DeepseekV41ForConditionalGeneration(config).to(torch_device).eval()
        model.model.bind_tokenizer(tokenizer)
        # End prefill with an image, then resume text: the old image mask must not
        # route the next token through VL biases or mark it DEAD for Engram.
        span_length = self.model_tester.span_length
        inputs["input_ids"] = inputs["input_ids"][:, :span_length]
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["image_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        continuation = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        with torch.no_grad():
            model.model.language_model.layers[0].mlp.gate.e_score_correction_bias_vl[0] = 100
            embeddings = model.get_input_embeddings()(inputs["input_ids"])
            ordinary = model(**inputs, use_cache=False).logits
            torch.testing.assert_close(model(**inputs, inputs_embeds=embeddings, use_cache=False).logits, ordinary)
            without_ids = {key: value for key, value in inputs.items() if key != "input_ids"}
            with self.assertRaisesRegex(ValueError, "engram layers require"):
                model(**without_ids, inputs_embeds=embeddings)
            with self.assertRaisesRegex(ValueError, "engram layers require"):
                model.generate(**without_ids, inputs_embeds=embeddings, max_new_tokens=1)

            full = dict(inputs)
            full["input_ids"] = torch.cat((inputs["input_ids"], continuation), -1)
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            full["image_mask"] = torch.nn.functional.pad(inputs["image_mask"], (0, continuation.shape[1]), value=False)
            expected = model(**full, use_cache=False).logits[:, span_length:]
            output = model(**inputs, use_cache=True)
            kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
            kwargs["use_cache"] = True
            ids = inputs["input_ids"]
            chunks = []
            for token in continuation.unbind(-1):
                kwargs = model._update_model_kwargs_for_generation(output, kwargs)
                ids = torch.cat((ids, token[:, None]), -1)
                prepared = model.prepare_inputs_for_generation(ids, next_sequence_length=1, **kwargs)
                output = model(**prepared)
                chunks.append(output.logits)
            torch.testing.assert_close(torch.cat(chunks, 1), expected, atol=1e-5, rtol=1e-5)
            options = {"max_new_tokens": 3, "do_sample": False, "bad_words_ids": [[config.image_token_id]]}
            generated = model.generate(**inputs, **options)
            self.assertTrue(torch.equal(generated, model.generate(**inputs, inputs_embeds=embeddings, **options)))
            self.assertTrue(torch.equal(generated, model.generate(**inputs, use_cache=False, **options)))

    def test_native_fp8_conversion_preserves_image_features(self):
        from transformers import FineGrainedFP8Config
        from transformers.quantizers.auto import AutoHfQuantizer

        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch.bfloat16).eval()
            pixels, grids = inputs["pixel_values"].cpu(), inputs["image_grid_thw"].cpu()
            with torch.no_grad():
                expected = model.get_image_features(pixels, grids)
            quantizer = AutoHfQuantizer.from_config(
                FineGrainedFP8Config(
                    weight_block_size=(32, 32), scale_fmt="ue8m0", expert_dtype="fp4", dequantize=False
                ),
                pre_quantized=True,
            )
            # Execute native module replacement without CUDA kernels: image modules
            # must remain usable because the release provides no FP8 scales for them.
            quantizer._process_model_before_weight_loading(model)
            with torch.no_grad():
                actual = model.get_image_features(pixels, grids)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0)
            for actual_rows, expected_rows in zip(actual.pooler_output, expected.pooler_output):
                torch.testing.assert_close(actual_rows, expected_rows, atol=0, rtol=0)

    def test_native_quantized_composite_loading(self):
        from safetensors.torch import save_file

        from transformers import FineGrainedFP8Config

        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekV41ForConditionalGeneration(config).to(torch.bfloat16).eval()
        native = {}
        reference_state = dict(model.state_dict())
        quantized_suffixes = (
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_proj.weight",
            "self_attn.o_a_proj.weight",
            "self_attn.o_b_proj.weight",
            "indexer.q_b_proj.weight",
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        )
        f32_leaves = ("fn", "base", "scale", "sinks", "e_score_correction_bias", "e_score_correction_bias_vl")

        def pack_fp4(weight):
            groups = weight.float().reshape(weight.shape[0], -1, 32)
            scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 6)))
            normalized = groups / scale[..., None]
            codes = torch.bucketize(normalized.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
            values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])[codes] * normalized.sign()
            nibbles = codes.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)
            packed = (nibbles[..., ::2] | (nibbles[..., 1::2] << 4)).reshape(weight.shape[0], -1).to(torch.int8)
            return (
                packed,
                scale.to(torch.float8_e8m0fnu),
                (values * scale[..., None]).reshape_as(weight).to(weight.dtype),
            )

        for name, tensor in model.state_dict().items():
            native_name = self._to_native_name_vl(name)
            if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                is_gate_up = name.endswith("gate_up_proj")
                prefix = native_name.rsplit(".", 1)[0]
                restored = []
                for expert, weight in enumerate(tensor):
                    parts = weight.chunk(2, 0) if is_gate_up else (weight,)
                    restored_parts = []
                    for projection, part in zip(("w1", "w3") if is_gate_up else ("w2",), parts):
                        packed, scale, dequantized = pack_fp4(part)
                        native[f"{prefix}.{expert}.{projection}.weight"] = packed
                        native[f"{prefix}.{expert}.{projection}.scale"] = scale
                        restored_parts.append(dequantized)
                    restored.append(torch.cat(restored_parts))
                reference_state[name] = torch.stack(restored)
            elif name.endswith(quantized_suffixes):
                rows, columns = tensor.shape
                blocks = tensor.float().reshape(rows // 32, 32, columns // 32, 32)
                scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-4) / 448)))
                quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
                native[native_name] = quantized.reshape_as(tensor)
                native[native_name.removesuffix(".weight") + ".scale"] = scale.to(torch.float8_e8m0fnu)
                reference_state[name] = (
                    (quantized.float() * scale[:, None, :, None]).reshape_as(tensor).to(tensor.dtype)
                )
            else:
                native[native_name] = tensor.float() if name.rsplit(".", 1)[-1] in f32_leaves else tensor
        # Construct the dequantized reference with the same load dtype: a blanket
        # `.to(bfloat16)` also rounds nonpersistent RoPE buffers, unlike from_pretrained.
        model = DeepseekV41ForConditionalGeneration.from_pretrained(
            None, config=copy.deepcopy(config), state_dict=reference_state, dtype=torch.bfloat16
        )
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(
                {name: value.contiguous() for name, value in native.items()},
                os.path.join(directory, "model.safetensors"),
            )
            loaded, info = DeepseekV41ForConditionalGeneration.from_pretrained(
                directory,
                dtype=torch.bfloat16,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                output_loading_info=True,
            )
            self.assertFalse({key: value for key, value in info.items() if value}, info)
            with torch.no_grad():
                cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
                torch.testing.assert_close(
                    loaded(**cpu_inputs).logits, model(**cpu_inputs).logits, atol=1e-4, rtol=1e-4
                )

    def test_vision_norm_dtype_and_meta_loading(self):
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41VisionRMSNorm

        norm = DeepseekV41VisionRMSNorm(64).to(dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.1, 2, 64))
        values = torch.linspace(-1000, 1000, 192).reshape(3, 64).to(torch.bfloat16)
        reference = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(norm(values), (reference * norm.weight.float()).to(values.dtype), atol=0, rtol=0)
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekV41ForConditionalGeneration(config).eval()
        cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = DeepseekV41ForConditionalGeneration.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded(**cpu_inputs).logits, model(**cpu_inputs).logits, atol=0, rtol=0)
            loaded = DeepseekV41ForConditionalGeneration.from_pretrained(directory, dtype=torch.bfloat16).eval()
            for module in loaded.model.visual.modules():
                if isinstance(module, DeepseekV41VisionRMSNorm):
                    self.assertEqual(module.weight.dtype, torch.bfloat16)
            with torch.no_grad():
                actual = loaded.get_image_features(cpu_inputs["pixel_values"], cpu_inputs["image_grid_thw"])
                expected = model.to(torch.bfloat16).get_image_features(
                    cpu_inputs["pixel_values"], cpu_inputs["image_grid_thw"]
                )
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-2, rtol=2e-2)

    def test_vision_parity_with_reference(self):
        """Independent functional form of reference-impl/vision.py, including 2D RoPE and channel-major unfold."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Aligner, DeepseekV41VisionModel

        config = self.model_tester.get_config()
        vision = DeepseekV41VisionModel(config.vision_config).eval()
        aligner = DeepseekV41Aligner(config).eval()
        height, width = 7, 5
        patches = torch.randn(height * width, 3 * config.vision_config.patch_size**2)
        head_dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads
        inv_freq = 1.0 / (
            config.vision_config.rope_theta ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
        )
        hpos = torch.arange(height)[:, None].expand(height, width)
        wpos = torch.arange(width)[None, :].expand(height, width)
        phases = (torch.stack((hpos, wpos), -1).reshape(-1, 2, 1) * inv_freq).flatten(1)
        cos, sin = phases.cos()[:, None], phases.sin()[:, None]

        def norm(x, weight):
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            return (weight.float() * normalized).to(x.dtype)

        def rotate(x):
            first, second = x.float().chunk(2, dim=-1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)

        with torch.no_grad():
            x = torch.nn.functional.linear(patches, vision.patch_embed.proj.weight, vision.patch_embed.proj.bias)
            for block in vision.blocks:
                qkv = torch.nn.functional.linear(
                    norm(x, block.norm1.weight), block.attn.qkv.weight, block.attn.qkv.bias
                )
                q, k, v = [
                    part.reshape(-1, config.vision_config.num_attention_heads, head_dim) for part in qkv.chunk(3, -1)
                ]
                attention = (
                    torch.nn.functional.scaled_dot_product_attention(
                        rotate(q).transpose(0, 1), rotate(k).transpose(0, 1), v.transpose(0, 1)
                    )
                    .transpose(0, 1)
                    .reshape_as(x)
                )
                x = x + torch.nn.functional.linear(attention, block.attn.proj.weight, block.attn.proj.bias)
                gate, up = torch.nn.functional.linear(norm(x, block.norm2.weight), block.mlp.w1.weight).chunk(2, -1)
                x = x + torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, block.mlp.w2.weight)
            reference_features = norm(x, vision.norm.weight)
            # Construct the reference channel-major windows independently of F.unfold.
            ratio = config.vision_config.downsample_ratio
            grid = reference_features.reshape(height, width, -1)
            grid = torch.nn.functional.pad(grid, (0, 0, 0, -width % ratio, 0, -height % ratio))
            windows = torch.stack(
                [
                    grid[h : h + ratio, w : w + ratio].permute(2, 0, 1).flatten()
                    for h in range(0, grid.shape[0], ratio)
                    for w in range(0, grid.shape[1], ratio)
                ]
            )
            reference_rows = torch.nn.functional.linear(
                torch.nn.functional.gelu(torch.nn.functional.linear(windows, aligner.w1.weight, aligner.w1.bias)),
                aligner.w2.weight,
                aligner.w2.bias,
            )
            for implementation in ("eager", "sdpa"):
                vision.set_attn_implementation(implementation)
                outputs = vision(patches, grid_thw=torch.tensor([[1, height, width]]))
                torch.testing.assert_close(outputs.last_hidden_state, reference_features, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    aligner(outputs.last_hidden_state, height, width), reference_rows, atol=1e-5, rtol=1e-5
                )


@require_torch_accelerator
@slow
class DeepseekV41IntegrationTest(unittest.TestCase):
    """End-to-end checks on the published DeepSeek-V4.1-Flash checkpoint (476 GiB on disk:
    fp8 attention, packed-fp4 routed experts, two ~98 GB fp8 engram tables), through the
    standard loading path — the fp8 quantizer keeps the weights quantized on CUDA and the
    engram tables stay in host RAM when no accelerator can hold them. The expected tokens
    were reproduced on 4×H100 (with CPU offload), 8×H100 and, under expert parallelism,
    8×H200; they match the eager fp32 reference harness token-for-token. Run manually::

        RUN_SLOW=1 pytest tests/models/deepseek_v41/test_modeling_deepseek_v41.py::DeepseekV41IntegrationTest -s
        RUN_SLOW=1 torchrun --nproc-per-node 8 -m pytest ... -k expert_parallel
    """

    model_id = "deepseek-ai/DeepSeek-V4.1-Flash"
    # The repo ships no chat template; this is `encoding.encode_messages([user], "chat")`
    # for "Say hello and tell me what model you are." (bos, <|User|>, ..., <|Assistant|>, </think>).
    prompt_ids = [0, 128803, 63006, 44388, 305, 4575, 678, 1205, 2645, 440, 477, 16, 128804, 128822]
    expected_ids = [19923, 3, 342, 4571, 22651, 4374, 1465, 14, 411, 7703, 22896, 5572, 513, 22651, 4374, 1465]
    expected_text = "Hello! I'm DeepSeek, an AI assistant created by DeepSeek"

    def _generate(self, model):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = torch.tensor([self.prompt_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=16, do_sample=False)
        new = out[0, inputs.shape[1] :].tolist()
        self.assertEqual(tokenizer.decode(new), self.expected_text)
        self.assertEqual(new, self.expected_ids)

    @require_torch_multi_accelerator
    def test_v41_flash_fp8_generation(self):
        """`device_map="auto"` over the available accelerators: fp8 / fp4 weights kept
        quantized (dequantize=False on CUDA), engram tables excluded from placement."""
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        self._generate(model)

    @require_torch_multi_accelerator
    def test_v41_flash_expert_parallel_generation(self):
        """Expert parallelism (one process per accelerator, launched with torchrun):
        `base_model_ep_plan` shards the fp4 experts along the expert axis and the engram
        tables along their embedding dim. Needs >= 141 GB per rank on 8 ranks."""
        import os

        if "WORLD_SIZE" not in os.environ:
            self.skipTest("launch with torchrun to run the expert-parallel test")
        from transformers.distributed.configuration_utils import DistributedConfig

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            distributed_config=DistributedConfig(tp_size=int(os.environ["WORLD_SIZE"]), enable_expert_parallel=True),
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = torch.tensor([self.prompt_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=16, do_sample=False)
        # The DeepGEMM / dynamic-activation-quant kernels are not bit-identical to the
        # Triton path: check the answer, not the exact ids.
        text = tokenizer.decode(out[0, inputs.shape[1] :], skip_special_tokens=True)
        self.assertTrue(text.startswith("Hello! I'm"), text)
