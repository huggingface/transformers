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

    def test_beam_reorder_follows_engram_history(self):
        """The engram n-gram look-back lives on the cache: `cache.reorder_cache` alone
        must permute it (beam search has no model-side hook), and a decode step after
        the reorder must match a one-shot forward of the reordered sequences (found by
        the cross-engine audit — the CSA-layer reorder tests ran with engram disabled
        and missed the history)."""
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41EngramHistoryLayer

        tokenizer = tiny_word_tokenizer()
        config = self._tie_free_config(self._engram_config(tokenizer))
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
        # cache reorder path.
        beams = model.generate(inputs, max_new_tokens=4, num_beams=2, do_sample=False)
        self.assertEqual(tuple(beams.shape), (2, 13))

    def test_fp8_native_checkpoint_load(self):
        """Load a native-format quantized checkpoint replicating the released layout:
        fp8 e4m3 weights + ue8m0 block scales for attention/shared experts/engram.wkv,
        PACKED FP4 routed experts (e2m1 nibbles in int8, per-row 32-channel ue8m0
        scales — the released MXFP4 [1, 32] block), fp8 engram tables, BF16 compressor/indexer/embed, F32 mHC params.
        Found by the cross-engine audit: the `wo_a` grouped projection and the engram
        tables must survive the load, and the fp4 experts must unpack exactly."""
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
        native, dequantized = {}, {}
        for name, param in model.named_parameters():
            native_name = name
            for a, b in [
                ("model.layers.", "layers."),
                ("model.embed.", "embed."),
                ("model.norm.", "norm."),
                ("lm_head.", "head."),
            ]:
                native_name = native_name.replace(a, b)
            # the released checkpoint's quantized set: attention projections (incl.
            # the grouped wo_a), indexer.wq_b, engram.wkv and the SHARED experts are
            # fp8-block; the ROUTED experts are packed fp4 (e2m1 nibbles in an int8
            # container, one ue8m0 scale per row per 16 channels). Compressor,
            # indexer.wk/weights_proj, gate, embed and head stay bf16/f32.
            is_fp8 = (
                param.ndim == 2
                and any(
                    name.endswith(f".{suffix}")
                    for suffix in (
                        "attn.wq_a.weight",
                        "attn.wq_b.weight",
                        "attn.wkv.weight",
                        "attn.wo_a.weight",
                        "attn.wo_b.weight",
                        "indexer.wq_b.weight",
                        "engram.wkv.weight",
                        "w1.weight",
                        "w2.weight",
                        "w3.weight",
                    )
                )
                and ".experts." not in name
            )
            if name.endswith("engram.embed.weight"):
                rows, dim = param.shape
                blocks = param.float().view(rows, dim // 32, 32)
                amax = blocks.abs().amax(-1).clamp_min(1e-4)
                scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
                q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448)
                native["layers.1.engram.embed.weight"] = q.reshape(rows, dim).to(torch.float8_e4m3fn)
                native["layers.1.engram.embed.scale"] = scale.to(torch.float8_e8m0fnu)
                dequantized[name] = (
                    (q.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(rows, dim).to(torch.bfloat16)
                )
            elif ".experts." in name and param.ndim == 2:
                # routed experts: packed fp4 like the release — two e2m1 nibbles per
                # int8 byte (even index in the low nibble), one ue8m0 scale per row
                # per 32 fp4 channels (the released MXFP4 [1, 32] block: scales
                # [2304, 160] over an unpacked in-dim of 5120)
                out_dim, in_dim = param.shape
                groups = param.float().view(out_dim, in_dim // 32, 32)
                amax = groups.abs().amax(-1).clamp_min(1e-4)
                scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
                q = (groups / scale.unsqueeze(-1)).clamp(-6, 6)
                grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
                codes = torch.bucketize(q.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
                values = grid[codes] * torch.where(q < 0, -1.0, 1.0)
                nibbles = codes.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
                packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
                native[native_name] = packed.reshape(out_dim, in_dim // 2).to(torch.int8)
                native[native_name[: -len(".weight")] + ".scale"] = scale.to(torch.float8_e8m0fnu)
                dequantized[name] = (values * scale.unsqueeze(-1)).reshape(out_dim, in_dim).to(torch.bfloat16)
            elif is_fp8:
                weight, scale, deq = quantize_fp8(param.data)
                # the checkpoint keeps the param's own name for the fp8 tensor and
                # stores the block scales as a `.scale` SIBLING of it
                native[native_name] = weight
                native[native_name[: -len(".weight")] + ".scale"] = scale
                dequantized[name] = deq.to(torch.bfloat16)
            elif name.endswith("engram.embed.scale"):
                # the checkpoint's table scale was written by the branch above;
                # the model's (unused-in-bf16) scale param must not overwrite it
                continue
            else:
                native[native_name] = param.data.to(torch.bfloat16) if param.dtype.is_floating_point else param.data
        # F32 tensors stay F32 (the checkpoint's mHC / sink / bias dtypes)
        for name, param in model.named_parameters():
            if name.split(".")[-1] in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
                "attn_sink",
                "bias",
                "bias_vl",
                "q_weight",
                "k_weight",
            ):
                native_name = name
                for a, b in [("model.layers.", "layers.")]:
                    native_name = native_name.replace(a, b)
                native[native_name] = param.data.to(torch.float32)

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

        # 1. untouched tensors load verbatim (BF16/F32 modules survive the fp8 path)
        ref = dict(model.named_parameters())
        got = dict(loaded.named_parameters())
        for name in ("model.embed.weight", "model.layers.0.ffn.gate.weight", "model.layers.0.hc_attn_fn"):
            self.assertTrue(torch.allclose(ref[name].float(), got[name].float(), atol=1e-3), name)
        # 2. fp8 tensors dequantize to exactly what we packed (scales applied once)
        for name, deq in dequantized.items():
            self.assertTrue(torch.allclose(deq.float(), got[name].float(), atol=5e-2), name)
        # 3. the grouped wo_a specifically: dequantized and reshapeable
        wo_a = got["model.layers.0.attn.wo_a.weight"]
        self.assertEqual(wo_a.dtype, torch.bfloat16)
        wo_a.view(config.o_groups, -1, config.hidden_size)
        # 4. end-to-end: fp8-loaded logits track the bf16 original
        # stay inside the tiny tokenizer's range: the engram hashes map ids through
        # build_compressed_token_map, whose lookup has one row per tokenizer entry
        inputs = torch.randint(0, len(tokenizer), (2, 9))
        with torch.no_grad():
            clean = model(inputs).logits.float()
            quant = loaded(inputs).logits.float()
        self.assertLess((clean - quant).abs().max().item(), 0.35)

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
            self.assertIsNone(unbound.model.engram_hash_state.token_map)
            tokenizer.save_pretrained(tmp)
            loaded = self.model_tester.causal_lm_class.from_pretrained(tmp)
        with torch.no_grad():
            after = loaded(inputs).logits
        self.assertTrue(torch.allclose(before, after, atol=1e-5))
