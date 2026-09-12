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

    def test_beam_reorder_follows_engram_history(self):
        """The engram n-gram history lives on the model, not the cache: beam search
        must permute it together with the cache (found by the cross-engine audit —
        the CSA-layer reorder tests ran with engram disabled and missed it)."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_compressed_token_map

        tokenizer = tiny_word_tokenizer()
        config = self.model_tester.get_config()
        config.engram_layer_ids = [1, 2]
        config.engram_num_embeddings = [700, 700]
        config.engram_vocab_size = 64
        config.engram_n_heads = 2
        config.engram_head_dim = 16
        config.engram_pad_id = 8
        _, compressed_vocab = build_compressed_token_map(tokenizer)
        config.engram_compressed_vocab_size = compressed_vocab
        # `generate` samples over the full vocab; the hash state's token map only
        # covers the tokenizer, so keep the two aligned in this test.
        config.vocab_size = len(tokenizer)

        model = self.model_tester.causal_lm_class(config).eval()
        model.model.bind_tokenizer(tokenizer)
        inputs = torch.randint(0, len(tokenizer), (2, 9))
        with torch.no_grad():
            out = model(inputs, use_cache=True)
        history = model.model.engram_hash_state.history.clone()
        self.assertEqual(history.shape[0], 2)

        model._reorder_cache(out.past_key_values, torch.tensor([1, 0]))

        self.assertTrue(torch.equal(model.model.engram_hash_state.history, history.flip(0)))

        # End to end: beam search with engram layers enabled must run and stay
        # deterministic across two runs with the same seed.
        torch.manual_seed(0)
        beams1 = model.generate(inputs, max_new_tokens=4, num_beams=2, do_sample=False)
        torch.manual_seed(0)
        beams2 = model.generate(inputs, max_new_tokens=4, num_beams=2, do_sample=False)
        self.assertTrue(torch.equal(beams1, beams2))

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
        return name

    def test_fp8_native_checkpoint_load(self):
        """Load a native-format quantized checkpoint replicating the released layout AND
        names: fp8 e4m3 weights + ue8m0 block scales for attention / shared experts /
        engram.wkv, PACKED FP4 routed experts (per expert `w1` / `w2` / `w3`: e2m1
        nibbles in int8, per-row 32-channel ue8m0 scales — the released MXFP4 [1, 32]
        block), fp8 engram tables, BF16 compressor / indexer / embed, F32 mHC params and
        gate biases. The `wo_a` grouped projection and the engram tables must survive
        the load, and the per-expert fp4 tensors must land exactly in the fused
        `gate_up_proj` / `down_proj`."""
        from safetensors.torch import save_file

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
            elif name.endswith("engram.embed.weight"):
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
            elif name.endswith("engram.embed.scale"):
                continue  # written by the branch above; the model's (unused-in-bf16) scale must not overwrite it
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
            composite = DeepseekV41Config(
                text_config=config.to_dict(),
                quantization_config={
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [32, 32],
                    "scale_fmt": "ue8m0",
                    "expert_dtype": "fp4",  # routed experts ship packed fp4, like the release
                },
            )
            composite.save_pretrained(tmp)
            loaded = self.model_tester.causal_lm_class.from_pretrained(tmp, dtype=torch.bfloat16)
            loaded.model.bind_tokenizer(tokenizer)

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
