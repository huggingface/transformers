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
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DeepseekV4Model,
        FineGrainedFP8Config,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class DeepseekV4ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV4Model

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Standard CausalLMModelTester knobs — override the parent's positional defaults.
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.num_hidden_layers = 2
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 64
        self.max_position_embeddings = 64
        # V4-only knobs.
        self.head_dim = 32
        self.partial_rotary_factor = 8 / 32  # qk_rope_head_dim=8 / head_dim=32
        self.q_lora_rank = 32
        self.o_groups = 2
        self.o_lora_rank = 16
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        # All "moe" (no "hash_moe") so inputs_embeds-only generation tests in
        # CausalLMModelTest exercise the model without hitting the hash router's
        # input_ids requirement. A dedicated test covers the hash path.
        self.mlp_layer_types = ["moe", "moe"]
        self.layer_types = ["heavily_compressed_attention", "compressed_sparse_attention"]
        self.sliding_window = 8
        self.hc_mult = 2
        self.hc_sinkhorn_iters = 3
        self.hc_eps = 1.0e-6
        self.index_n_heads = 2
        self.index_head_dim = 16
        self.index_topk = 2
        self.num_nextn_predict_layers = 0
        self.scoring_func = "sqrtsoftplus"
        self.routed_scaling_factor = 1.5
        self.swiglu_limit = 10.0
        self.rope_theta = 10000.0
        self.compress_rope_theta = 160000.0
        self.attention_bias = False
        self.attention_dropout = 0.0


@require_torch
class DeepseekV4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV4ModelTester

    # Indexer parameters only influence the argmax over compressed positions (``topk``),
    # which is non-differentiable — their gradients flow through a separate objective in
    # the upstream training recipe, not the main causal-LM loss.
    test_all_params_have_gradient = False

    # No SequenceClassification / TokenClassification / QA heads on V4.
    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    @unittest.skip(
        "V4's `DeepseekV4GroupedLinear` uses `torch.bmm` for the per-group matmul; "
        "torchao's Float8Tensor only fast-paths `F.linear` (bmm needs the optional `mslk` "
        "kernel) so the quantized-TP path fails. A custom V4 FP8 path will land later."
    )
    def test_tp_generation_quantized(self):
        pass

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        # V4 layers with a Compressor attend to extra pooled positions, so the KV
        # length varies per layer. We only check the shape invariants: batched, same
        # number-of-heads and query-length; the KV-length axis may differ across layers.
        import torch  # noqa: PLC0415

        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), (output_length - prompt_length))
        for _, iter_attentions in enumerate(attentions):
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertIsInstance(layer_attention, torch.Tensor)
                self.assertEqual(layer_attention.shape[0], batch_size)
                self.assertEqual(layer_attention.shape[1], config.num_attention_heads)

    @unittest.skip(
        "V4's rotary uses per-layer-type inv_freq buffers (Gemma3 pattern); the common test calls forward without `layer_type` and reads `.inv_freq`, neither of which apply."
    )
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip(
        "V4's rotary uses per-layer-type rope_parameters; the common test sets a flat dict and skips for multi-layer-type rotaries."
    )
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_hidden_states_output(self):
        # V4 layers emit a 4D ``[B, S, hc_mult, hidden]`` tensor — the hc_mult streams
        # are only collapsed at the top of the model via ``hc_head``. The common
        # ``test_hidden_states_output`` assumes ``(batch, seq, hidden)``; we re-run the
        # same check but accept the extra HC axis, and we additionally assert the final
        # (post-hc_head) ``last_hidden_state`` has the standard 3D shape.
        import torch  # noqa: PLC0415

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            seq_len = inputs_dict["input_ids"].shape[1]
            for layer_h in hidden_states:
                # Accept either the collapsed (3D) post-head shape or the per-layer 4D shape.
                if layer_h.ndim == 3:
                    self.assertEqual(layer_h.shape, (inputs_dict["input_ids"].shape[0], seq_len, config.hidden_size))
                else:
                    self.assertEqual(
                        layer_h.shape,
                        (inputs_dict["input_ids"].shape[0], seq_len, config.hc_mult, config.hidden_size),
                    )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # Every V4 layer is sliding-window, so the cache is length-bounded to
        # ``sliding_window`` instead of the full ``seq_length`` the parent tester expects.
        # We also accept the compressed-segment positions that ``DeepseekV4Attention``
        # appends on compress layers (they live beyond the window on the keys axis).
        import torch  # noqa: PLC0415

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.head_dim
        for layer in past_key_values.layers:
            keys, values = layer.keys, layer.values
            self.assertIsInstance(keys, torch.Tensor)
            self.assertEqual(keys.shape[0], batch_size)
            self.assertEqual(keys.shape[1], num_kv_heads)
            self.assertEqual(keys.shape[3], head_dim)
            self.assertEqual(keys.shape, values.shape)

    @unittest.skip(
        reason=(
            "V4's conversion mapping is two-pass: a structural prefix rename "
            "(``layers.X.attn.`` → ``model.layers.X.self_attn.``) runs first, then specific in-prefix "
            "renames operate on the already-prefixed HF-form keys (``model.layers.X.self_attn.compressor.norm.`` "
            "→ ``...compressor.kv_norm.``). This split is load-bearing for save / load round-tripping — "
            "any single-pass ordering loses information in either direction (the general prefix rule "
            "and a specific in-prefix rule both want to match the same upstream key, and one of the "
            "two directions ends up with the general rule stealing the match). The base "
            "``test_reverse_loading_mapping`` checks every source pattern against the *upstream-form* "
            "serialized keys, so the Pass 2 patterns (written in HF form) inherently can't satisfy "
            "that invariant. The actual round-trip is exercised by ``test_save_load``."
        )
    )
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip(
        reason=(
            "V4's compressor pools windows of ``compress_rate`` consecutive tokens *before* the "
            "attention mask is applied — left-padding shifts the window boundaries so pad tokens "
            "get folded into the pooled KV entries, and the resulting logits diverge from the "
            "unpadded run by design (same fundamental limitation as RecurrentGemma)."
        )
    )
    def test_left_padding_compatibility(self):
        pass

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # V4's per-layer hidden states carry an extra ``hc_mult`` dim (Hyper-Connection
        # parallel streams). We skip the exact seq-length assertion the base tester does,
        # because assisted-decoding feeds arbitrary draft-token batches in, and just
        # sanity-check batch / hidden dims.
        import torch  # noqa: PLC0415

        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)


@require_torch
@require_torch_accelerator
@slow
class DeepseekV4IntegrationTest(unittest.TestCase):
    """End-to-end check on the published DeepSeek-V4-Flash checkpoint.

    Loads the real 43-layer FP8 weights, dequantizes on the fly via
    :class:`FineGrainedFP8Config`, and greedy-generates a continuation of a fixed
    prompt. The forward path that this test covers is everything past the typical
    tiny-config tests can reach: the per-layer FP8 dequant in
    ``update_weight_conversions``, the ``compress_ratios → layer_types`` config
    translation (sliding / CSA / HCA), the ``coff=2`` overlap-window pooling on CSA
    layers and the indexer's inner pool, the per-head Q rescale in
    :class:`DeepseekV4Attention`, the YaRN-blended ``compress_rope_theta`` in the
    compressor, the trailing-rope partial-RoPE convention, and the cross-layer
    Hyper-Connection signal propagation. Any regression in those would tip
    generation back into a single-token collapse or pure ``<EOS>`` output (the
    failure modes we hit while landing the architecture).

    Marked ``@slow`` because the checkpoint is ~700 GB on disk and only loadable
    on a multi-GPU host (``device_map="auto"`` plus FP8 dequant materializes the
    weights in bf16). Run manually with::

        RUN_SLOW=1 pytest tests/models/deepseek_v4/test_modeling_deepseek_v4.py::DeepseekV4IntegrationTest -k generation -s
    """

    model_id = "deepseek-ai/DeepSeek-V4-Flash"
    prompt = "Pipeline parallelism in ai is "

    def test_v4_flash_fp8_generation(self):
        # ``dequantize=True`` so we can run on bf16-only kernels (needed for the
        # ``grouped_mm`` path the routed experts hit). Eager attention so we
        # exercise the same forward we tune the rest of the V4 modeling around.
        quantization_config = FineGrainedFP8Config(dequantize=True)
        config = AutoConfig.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
        )

        inputs = tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        # Snapshot of greedy-decoded text. The exact continuation is deterministic
        # under ``do_sample=False`` for a fixed prompt — if this snapshot drifts,
        # something in the V4 forward / RoPE / Q-rescale / HC stack changed.
        expected = (
            "Pipeline parallelism in ai is  driven by three key factors: the exponential increase in data "
            "size, the development of increasingly powerful computational techniques (especially deep "
            "learning), to handle this data, and the availability of massive computational resources on "
            "which to run these methods, all of which are are  well aligned with  trends in  industry, "
            " academia and  research"
        )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        self.assertEqual(decoded, expected)

    def test_v4_flash_fp8_chat_seven_prompts(self):
        """Chat-templated greedy generation across 7 prompts of varying length.

        Covers: short factual recall (1, 2), translation (3), code generation (4),
        out-of-distribution recall (5), open-ended creative writing (6), and a
        long-context summarisation (7, 234 input tokens — exercises the HCA path
        since input >> ``compress_rates['heavily_compressed_attention']`` = 128).
        Each completion is a fixed snapshot of the current greedy output. If any
        snapshot drifts, something changed in: per-layer-type RoPE selection
        (sliding ``main`` vs CSA / HCA ``compress``), the CSA / HCA per-query
        ``block_bias`` causal mask, the Hyper-Connection Sinkhorn projection or
        the residual mixing direction, or the fp32 promotion in the MoE path.
        """

        def _chat(prompt: str) -> str:
            # V4-Flash chat-mode template (canonical form in ``encoding/encoding_dsv4.py``
            # on the model repo). ``</think>`` after ``<｜Assistant｜>`` skips the
            # reasoning block and goes straight to the answer.
            return f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>"

        long_prompt = (
            "Please read the following extended passage carefully and then provide a concise "
            "three-sentence summary that captures the main themes and the most important details. "
            'Be precise and avoid restating the wording; paraphrase. Passage: "It is a truth '
            "universally acknowledged, that a single man in possession of a good fortune, must be "
            "in want of a wife. However little known the feelings or views of such a man may be "
            "on his first entering a neighbourhood, this truth is so well fixed in the minds of "
            "the surrounding families, that he is considered the rightful property of some one or "
            "other of their daughters. 'My dear Mr. Bennet,' said his lady to him one day, 'have "
            "you heard that Netherfield Park is let at last?' Mr. Bennet replied that he had not. "
            "'But it is,' returned she; 'for Mrs. Long has just been here, and she told me all "
            "about it.' Mr. Bennet made no answer. 'Do not you want to know who has taken it?' "
            "cried his wife impatiently. 'You want to tell me, and I have no objection to hearing "
            "it.' This was invitation enough.\""
        )

        cases: list[tuple[str, str]] = [
            (
                "The capital of France is",
                "The capital of France is Paris.",
            ),
            (
                "List the first ten prime numbers:",
                "The first ten prime numbers are:\n\n2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
            ),
            (
                "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
                '"Le rapide renard brun saute par-dessus le chien paresseux."',
            ),
            (
                "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
                (
                    "Here's a Python function that returns the nth Fibonacci number:\n\n"
                    "## Method 1: Iterative (Most Efficient)\n\n"
                    '```python\ndef fibonacci(n):\n    """\n    Returns the nth Fibonacci number.\n    \n'
                    "    Args:\n        n: Non-negative integer (0-indexed: fib(0)=0, fib(1)="
                ),
            ),
            (
                "What are the three properties of the UE8M0 scale factor format?",
                (
                    "Based on the standard naming convention for fixed-point data types "
                    '(where "U" stands for unsigned, "E" for exponent, and "M" for mantissa), '
                    "the **UE8M0** format has the following three properties:\n\n"
                    "1.  **Unsigned (U):** The value is always non-negative"
                ),
            ),
            (
                'Write a short story that begins with: "Once upon a time, in a forest far away, there lived a..."',
                (
                    "Once upon a time, in a forest far away, there lived a squirrel named Pip who was "
                    "terrified of the dark. While other squirrels slept soundly in their dreys as the "
                    "moon rose, Pip would lie awake, his tiny heart hammering against his ribs. The "
                    "rustle of leaves was a monster’s breath"
                ),
            ),
            (
                long_prompt,
                (
                    "The opening establishes a societal assumption that wealthy single men are naturally "
                    "seeking wives, making them prime targets for local families with eligible daughters. "
                    "Mrs. Bennet eagerly informs her indifferent husband that Netherfield Park has been "
                    "leased, hoping to spark his interest in the new tenant. Mr. Bennet’s dry, reluctant engagement"
                ),
            ),
        ]

        quantization_config = FineGrainedFP8Config(dequantize=True)
        config = AutoConfig.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
        )

        for i, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(prompt_index=i):
                inputs = tokenizer(_chat(prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0, inputs.input_ids.size(1) :]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                self.assertEqual(completion, expected)
