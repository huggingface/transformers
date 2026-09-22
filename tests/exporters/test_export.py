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
import functools
import inspect
import itertools
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from parameterized import parameterized

from transformers import GenerationConfig, set_seed
from transformers.exporters.components import Component, ComponentRole
from transformers.exporters.configs import AotiConfig, TensorrtConfig
from transformers.exporters.decompose import decompose_for_generation, decompose_multimodal, is_multimodal
from transformers.exporters.exporter_aoti import AotiExporter
from transformers.exporters.exporter_dynamo import _VARLEN_ATTENTION_PATHS, DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.exporter_openvino import OpenVINOConfig, OpenVINOExporter
from transformers.exporters.exporter_tensorrt import TensorrtExporter
from transformers.exporters.utils import (
    cast_leaf_tensors,
    get_leaf_tensors,
    module_device,
    module_dtype,
    precompute_export_inputs,
)
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch_greater_or_equal,
    require_torch_tensorrt,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)


# ──────────────────────────── skip lists ────────────────────────────
#
# A single mapping ``EXPORT_SKIPS[scope][model_class_name] = reason`` drives every skip.
# ``scope`` is a dotted path that narrows from broad (``"all"`` — every backend, every variant)
# to specific (``"onnx.generate"``, ``"onnx.dynamic"``, ``"openvino"``, …). At test time
# ``_should_skip`` walks the scopes that match the current ``(backend, generate, dynamic)``
# triple and returns ``True`` as soon as the model is found in any of them. Reasons live next
# to the model name so the "why" travels with the entry.
#
# Adding a new skip: pick the most specific scope that applies and add a ``"Name": "reason"``
# entry. Add a new scope key if the existing ones don't fit.


EXPORT_SKIPS: dict[str, dict[str, str]] = {
    # Every backend, every variant.
    "all": {
        "VideoMAEForPreTraining": (
            "Computes loss even when `return_loss=False`, hitting a data-dependent guard in "
            "`mse_loss`. TODO: skip loss when labels aren't provided."
        ),
        "OpenAIPrivacyFilterModel": (
            "`get_correct_experts_implementation` defaults to `eager` because the model is "
            "sensitive to accumulation order. Eager experts forward iterates over "
            "`expert_hit.nonzero()` (data-dependent shape). Users can opt into "
            "`set_experts_implementation('batched_mm')` to export."
        ),
        "OpenAIPrivacyFilterForTokenClassification": (
            "Same root cause as `OpenAIPrivacyFilterModel` — eager experts implementation."
        ),
        "GlmImageModel": (
            "Vision attention does a data-dependent chunked split (`torch.split(..., lengths.tolist())` "
            "over `cu_seqlens`), which hits `GuardOnDataDependentSymNode: u0 > 1` — it needs the shared "
            "vision-attention export patch, and even with it the export runs long (further guards / slow "
            "symbolic lowering). Not worth the model-specific export support for a diffusers-pipeline "
            "model. TODO: revisit on demand."
        ),
        "GlmImageForConditionalGeneration": "Same as `GlmImageModel`.",
    },
    # Every backend, generate path only.
    "generate": {
        "Blip2ForConditionalGeneration": (
            "`generate()` delegates to the inner language model without calling top-level "
            "`forward()`, so `decompose_prefill_decode` can't capture inputs. "
            "TODO: route generate through top-level `forward()`."
        ),
        "InstructBlipForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "InstructBlipVideoForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "Kosmos2ForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "RecurrentGemmaForCausalLM": (
            "Stores recurrent/conv state as module attributes (not a `Cache` object); "
            "`torch.export` can't carry that state between calls. "
            "TODO: refactor to a cache-based SSM pattern (like Mamba/Mamba2)."
        ),
        "MoshiForConditionalGeneration": (
            "Its audio kwargs reach no graph: the dynamo drive dies in `_validate_model_kwargs` (`The "
            "following model_kwargs are not used by the model: ['moshi_audio_codes', 'user_audio_codes']`) "
            "and the ONNX one on a device mismatch (`X1 and X2 must have the same device type. X1: cpu X2: "
            "cuda`) — not the rank mismatch this entry used to claim. Only the *merged* decode, and only "
            "without a static cache: measured with this lifted, 34 of 36 variants pass and those two fail. "
            "TODO: carry a model's own per-step audio kwargs through the decomposition, the way "
            "`PerceptionLMForConditionalGeneration` needs for its video path — the same shape of gap."
        ),
        "DiaForConditionalGeneration": (
            "Decodes several audio codebooks at once, so its decoder inputs carry a channel axis "
            "(`decoder_input_ids` is 3-D) and its `decoder_attention_mask` is shaped to match. The runtime "
            "builds the decoder's causal mask from the cache — 2-D positions into a `[batch, 1, q, kv]` "
            "mask — which is the right thing for every other encoder-decoder and the wrong rank here "
            "(`upper bound and lower bound inconsistent with step sign`). TODO: shape the decoder mask "
            "from the graph's own declared rank, the way `_mask_feed` already does for mixed attention."
        ),
        "Gemma3nForConditionalGeneration": (
            "Its text model takes an extra `per_layer_inputs` tensor that the multi-modal decomposition does "
            "not carry, so the captured call raises `TypeError` on `self.language_model(...)` "
            "(`modeling_gemma3n.py:2148`) — immediately, on all three backends, before any export runs. The "
            'old reason here (prefill returning only `logits`, "same shape as Voxtral") no longer applies: '
            "Voxtral now passes and its entry is gone. Measured with this lifted: 18 of 36 variants pass, 11 "
            "reach the runtime drive. TODO: let the decomposition carry a model's extra per-layer inputs."
        ),
        "VibeVoiceForConditionalGeneration": (
            "Generation uses two forward calls with different input shapes (prefill + noise scheduler); "
            "`decompose_prefill_decode` can't capture the full generate path reliably, causing flaky "
            "CUDAGraphs / export failures. TODO: handle in a follow-up PR."
        ),
    },
    # Every backend, dynamic-shape only.
    "dynamic": {
        "Sam2Model": (
            "`torch.export` of the Hiera vision backbone under dynamic shapes exceeds the 10-minute "
            "test timeout (12 attention blocks × 3 Q-pool stage transitions on symbolic H/W). Backend-"
            "agnostic — the torch.export step itself overruns, so every backend hits it."
        ),
        "Sam2VisionModel": (
            "torch 2.13's constraint solver raises `NotImplementedError` from `solve_univariate_inequality` "
            "on the Hiera window-partition guard `Eq(s/32 - (s/4)//8, 0)` (a `FloorDiv` in a rational "
            "equation); tracing itself succeeds. ONNX + ORT also overrun the 1000s timeout at ~7.5 min."
        ),
        "SeamlessM4TForSpeechToSpeech": (
            "The Conformer speech encoder is non-causal, so `sdpa_attention_forward` evaluates "
            "`q_length > 1 and attention_mask is None and is_causal`; under dynamic shapes `q_length > 1` "
            "is a `SymBool` and Python's `and` returns it as the first falsy operand, so SDPA raises "
            "`argument 'is_causal' must be bool, not SymBool`. Static shapes work. TODO: handle on the "
            "exporter side, see https://github.com/huggingface/transformers/pull/46196#discussion_r3717333141"
        ),
        "SeamlessM4TForSpeechToText": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
        "SeamlessM4Tv2ForSpeechToSpeech": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
        "SeamlessM4Tv2ForSpeechToText": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
    },
    # Generate path, dynamic-shape only. Backend-agnostic (it's in the shared decomposition).
    "generate.dynamic": {
        "ReformerModelWithLMHead": (
            "Carries LSH state as `past_buckets_states` (a list of tuples) plus `num_hashes` / "
            "`next_sequence_length` kwargs instead of a `Cache`, so the runtime — which feeds "
            "`past_key_values` / `cache_params` — can't satisfy the exported signature "
            "(`kwarg keyword mismatch`). TODO: refactor Reformer onto a `Cache` subclass."
        ),
    },
    # Generate path, the *runtime* half only: these export fine, and the export assertions still run —
    # what fails is driving the exported graphs through `generate`.
    "generate.runtime": {
        "KyutaiSpeechToTextForConditionalGeneration": (
            "Encodes its audio window-by-window inside `prepare_inputs_for_generation` — slicing "
            "`input_values` by a moving `current_window`, running the codec model with its own "
            "`encoder_past_key_values` and `padding_cache`, and copying the new tokens in-place — so "
            "driving the exported graphs takes that model-specific loop, not the generic one (the runtime "
            "never consumes `input_values`, and the eager side's codec state has no exported counterpart)."
        ),
        "MiniMaxForCausalLM": (
            "`MiniMaxCache` keeps two state containers: `layers`, which the traced cache fills only for the "
            "attention layers, and a separate `linear_cache` list for the lightning-attention state. The "
            "invented-layer half of this is now answerable — the export metadata records which layers the "
            "traced cache really had, so the extra `LinearAttentionLayer`s `generate` pre-sizes from "
            "`config.layer_types` can be dropped (measured: doing that leaves the 19 non-generate export "
            "variants passing). What remains is `linear_cache`: it is not layer-shaped, so "
            "`materialize_cache_layers` cannot fill it, and driving `generate` walks a `linear_cache` leaf "
            "path into an empty list (`TypeError: 'NoneType' object is not subscriptable`). TODO: give the "
            "lightning layers a real `LinearAttentionLayer` (dropping `linear_cache`); a config-aware "
            "pre-size alone is not enough."
        ),
        "xLSTMForCausalLM": (
            "`xLSTMCache` is not a full `Cache`: it keeps its state in `rnn_state` with no `layers` list, and "
            "lacks API `generate` expects (`is_compileable`), so every step of building and driving it "
            "surfaces as the next `AttributeError`. TODO: bring the class up to the `Cache` API rather than "
            "special-case it in the runtime."
        ),
        "DeepseekV4ForCausalLM": (
            "Its HCA/CSA cache layers hold dict-keyed state that the trace bakes into the input tree spec, "
            "and a fresh cache cannot present it. The dict *keys* do match — `DeepseekV4HCACache` declares "
            "`compressor` and `DeepseekV4CSACache` adds `indexer` in `__init__` — but their values are "
            "`None` until the compressor first fires, where the traced spec carries tensors "
            "(`buffer_kv`/`buffer_gate`/`compressed_kv`/`overlap_*`, 19 leaves), so the leaf *count* differs. "
            "Those tensors are buffered source tokens and emitted compression entries, i.e. real prefill "
            "state, so pre-creating them as zeros would feed the window tokens the model never saw; the "
            "graph is specialized to a mid-compression state (`entry_count: {'compressor': 1, 'indexer': 1}`) "
            "that its tensor-only write-back cannot advance either. Recording the traced context and "
            "restoring it onto a built cache was tried and does not help — the leaf count is the blocker, "
            "not the counters. ExportArtifacts itself passes, and so do the static-cache generate variants."
        ),
        "CsmForConditionalGeneration": (
            "Generates a *frame* at a time: `input_ids` is `[batch, sequence, codebooks]` and each step runs "
            "the backbone then the depth decoder to fill the codebooks, so `generate`'s loop cannot append "
            "the next token (`torch.cat([input_ids, next_tokens[:, None]], dim=-1)` sees 3 dims and 2). "
            "Driving it needs the model's own two-stage loop, not a generic one; the graphs themselves "
            "export and match eager (the non-generate variants cover them)."
        ),
        "HiggsAudioV2ForConditionalGeneration": (
            "Its `prepare_inputs_for_generation` does per-step surgery no generic loop reproduces: it "
            "counts how many audio ids the cache already holds, masks those out, and in decode drops "
            "`input_ids` entirely to pass only the last audio-codebook row. The runtime feeds the generic "
            "text+kwargs step instead, so generation diverges from the first token. ExportArtifacts itself and the "
            "per-component parity still run."
        ),
        "XLMWithLMHeadModel": (
            "Its `prepare_inputs_for_generation` appends a mask token to `input_ids` every step and builds a "
            "`langs` tensor from `config.lang_id`, so the graph takes a per-step input only that model can "
            "produce (and a step is one token wider than `generate`'s). ExportArtifacts itself is covered by the "
            "non-generate variants."
        ),
        "XLNetLMHeadModel": (
            "Its `prepare_inputs_for_generation` builds a fresh `perm_mask` and `target_mapping` for every "
            "step and appends a dummy token, so a decode step is three tokens wide over `mems` rather than "
            "one over a `Cache`. Those tensors are model-specific per-step inputs the graph declares but no "
            "generic runner can synthesize. The model is already on the deprecation list in "
            "`_supports_default_dynamic_cache`; export itself is covered by the non-generate variants."
        ),
        "BltForCausalLM": (
            "Reads `past_key_values.self_attention_cache`, i.e. wants an `EncoderDecoderCache` pair, but "
            "`config.is_encoder_decoder` is False so `generate` builds a plain `DynamicCache`. Handing it a "
            "pair of fresh `DynamicCache`s gets past the attribute error and then mismatches the input tree "
            "spec, because the traced pair's halves are not both empty. TODO: derive the pair's shape from "
            "the trace rather than guessing it."
        ),
        "RwkvForCausalLM": (
            "Carries its fixed-size state as a plain tensor list under its own `state` kwarg and output "
            "field — not a `Cache` under `past_key_values`/`cache_params` — so the runtime's cache plumbing "
            "(runner choice, feed, write-back, propagation through `generate`) has no counterpart: every "
            "decode step re-picks the prefill graph and trips its baked prompt-length guard. TODO: teach "
            "`cache_input`/`forward` the `state` kwarg, or port RWKV onto a `Cache` subclass."
        ),
    },
    # The runtime drives these, but not from a *merged* decode — every other variant is served.
    "dynamo.generate.runtime.multi_token": {
        "VoxtralRealtimeForConditionalGeneration": (
            "Streams its audio alongside the text — `generate` embeds `input_features` once outside the loop "
            "and hands each step the window its own tokens span — so the runtime drives it through the "
            "embedder component and a `past_seen * downsample_factor` slice (`_STREAMING_EMBEDDERS`). That "
            "works for every variant but the merged decode, which folds `downsample_factor` audio rows into "
            "the feature axis: reshaping a symbolic-length axis into `(n, k)` makes torch decide view-vs-copy "
            "on whether `n` is 1, so the trace bakes `Ne(frames//4, 1)` and the single-token steps `generate` "
            "makes violate it (`Guard failed: encoder_inputs_embeds.size()[1] // 4 != 1`). The graph exports "
            "fine; only the drive trips.\n"
            "Dynamo only, and not because the other backends serve it better: the guard is a torch-level "
            "shape assertion that lives in the `ExportedProgram`, and ONNX's lowered graph reshapes from the "
            "runtime shape instead — measured, its merged decode drives and matches ids. Two fixes were tried "
            "and measured not to help: making the `inputs_embeds += audio_embeds` broadcast explicit, and "
            "spelling the reshape's sizes out instead of `-1` (the branch is in "
            "`_reshape_view_helper_core_alg`, not in how the size is written). The non-merged variants do "
            "drive, because the split keeps a `prefill` graph whose decode is traced at length 1."
        ),
    },
    # The runtime drives these, but not from a *merged* decode — every other variant is served.
    "generate.runtime.multi_token": {
        "ClvpForCausalLM": (
            "Its merged decode graph carries a deferred assert the prompt cannot satisfy. Captured on "
            "continuation steps only (cache 3, query 2, mask 5), the export derives "
            "`attention_mask.size()[1] >= 4`; the prompt step is 3 wide, so driving the prompt through that "
            "same graph trips it. Measured: nothing records it statically — `range_constraints` is empty and "
            "it is a graph assert — so it is discoverable only by running the exported graph. The export "
            "itself is fine, and so is the model with a separate prefill graph; it is one-graph mode it "
            "cannot do. TODO: verify after export that the decode graph serves the prompt and keep the "
            "prefill when it does not, rather than deciding from the model's shape alone "
            "(`_needs_prefill_graph`)."
        ),
    },
    "generate.multi_token": {
        "ZayaForCausalLM": (
            "Its merged decode graph specializes the query axis instead of keeping it symbolic, which is the "
            "one thing this variant exists to avoid: the decode program comes back with `input_ids` at a "
            "static `(1, 2)` while `attention_mask` stays `(1, s53)`, so driving it with `generate`'s "
            "single-token steps trips the input-constraint check (`Guard failed: -1 + input_ids.size()[1] == "
            "1`). `TORCH_LOGS=+dynamic` traces the specializing guard (`Eq(s64, 2)`, from `expand` / "
            "`infer_size`) through the router's `router_hidden_states[:, -seq_length:]` "
            "(`modeling_zaya.py:460`) into the mixer and down to `update_conv_state`, which `copy_`s a "
            "fixed-`conv_kernel_size` buffer from a slice whose width follows the step. The other hybrids "
            "(bamba, jamba, lfm2, nemotron_h) share that cache helper and their multi-token variants pass, "
            "so the router slice is what compounds it here. Fails identically on dynamo and ONNX, and "
            "predates the metadata work (measured on both). Every other zaya variant passes."
        ),
        "ZambaForCausalLM": (
            "Its hand-copied mixer runs the selective scan per head with the associative path deliberately "
            "off ('Old model: only when user request it explicitly'), so the sequential scan unrolls and "
            "bakes the query length. The rest of the family (mamba / falcon_mamba / jamba) traces "
            "length-generically (the associative scan + its `initial_states`); aligning "
            "zamba's per-head mixer with mamba's would lift this."
        ),
        "ProphetNetForCausalLM": (
            'The eager model itself refuses the merged capture: its forward asserts "`use_cache` is only '
            'supported for `decoder_input_ids` of length 1" (`modeling_prophetnet.py`), so a 2-token '
            "continuation-from-past cannot even run, let alone trace. The assert is load-bearing, not "
            "defensive: in that branch `position_ids` is a single `(1, 1)` tensor and both ngram masks are "
            "`None`, so two new tokens would share a position embedding and not attend to each other. "
            "Lifting it is a refactor of the ngram mask machinery, and would also unblock prompt-lookup and "
            "assisted decoding, which hit this same assert today (measured) — so it is not export-only."
        ),
        "ReformerModelWithLMHead": (
            "Chunked local attention assumes a chunk-aligned query length; the merged multi-token query "
            "(seq 2) mismatches the chunked key axis (`size 2 vs 6`). "
            "Same chunked-attention limitation as the `onnx.generate` skip."
        ),
        "VibeVoiceForConditionalGeneration": (
            "Classifier-free guidance runs `forward()` twice per generated token — the conditional branch "
            "and the unconditional one, each with its own cache of a different length — so the captured "
            "calls interleave the two branches. `_merge_decode_calls` then merges a conditional decode step "
            "with an unconditional call, mismatching the query and cache axes (`size 5 vs 3` in attention). "
            "Single-token static generate is fine (it captures a conditional decode step). "
            "TODO: make the capture branch-aware."
        ),
    },
    # Multi-token decode capture on ExecuTorch: the SSM associative scan (what keeps the query axis
    # symbolic under export) has no ExecuTorch lowering and the runtime has no loop
    # primitive to lower it to, so those exports keep the sequential scan — which unrolls and pins the
    # merged decode to the traced step length. torch.export runs the scan natively; ONNX lowers it to a
    # dynamic-trip-count `Loop` (`_translate_associative_scan`).
    "executorch.generate.multi_token": {
        "MambaForCausalLM": "Sequential selective scan bakes the query length (no ExecuTorch associative_scan).",
        "FalconMambaForCausalLM": "Same as `MambaForCausalLM`.",
        "JambaForCausalLM": "Same as `MambaForCausalLM`.",
    },
    # ONNX, every variant.
    "onnx": {
        "CHMv2ForDepthEstimation": (
            "`run_decompositions` retraces through aot_autograd which emits a `detach_(alias(...))` "
            "pair the functional-graph assertion rejects (independent of any source `.detach()` — "
            "verified). Torch export works. TODO: file upstream `torch.export` issue."
        ),
        "PixioModel": ("Lowering exceeds the 10-minute test timeout."),
        "PixioBackbone": "Same `timeout` failure as `PixioModel`.",
    },
    # ONNX, generate path only.
    "onnx.generate": {
        "ReformerModelWithLMHead": (
            "Chunked local attention exports a Constant idx that exceeds the cached-keys axis "
            "length under static decode (prefill+1 token, seq=17 vs chunked axis of 16). The same "
            "computation stays symbolic under dynamic so ORT can't pre-validate it. The other "
            "three Reformer-local-attn ONNX variants pass."
        ),
    },
    # ONNX, driving the exported graphs through `generate` only — they export and run standalone.
    "onnx.generate.runtime": {
        "ProphetNetForConditionalGeneration": (
            "Exports and drives fine on dynamo and ExecuTorch; only the ONNX runtime can't feed it. The "
            "prefill session declares the encoder state as `encoder_last_hidden_state` while the runtime's "
            "feed carries no encoder entry under that name at all (`Required inputs "
            "(['encoder_last_hidden_state']) are missing from input feed`), so generation dies on the first "
            "call. Other encoder-decoders (bart) pass the same variant and dynamo names the same input "
            "`encoder_outputs_last_hidden_state` and works, so this is prophetnet-specific IO naming on our "
            "side, not a model limit. TODO: name the encoder input the way the runtime looks it up, then "
            "drop this skip."
        ),
    },
    # ONNX, dynamic-shape only.
    "onnx.dynamic": {
        "GroundingDinoModel": (
            "Same `detach_(alias(...))` retrace bug as CHMv2, but only triggered under dynamic "
            "shapes — `aot_autograd`'s decomposition pipeline emits the detach itself (verified "
            "by guarding all three modeling-side detaches with `if self.training`). Static works."
        ),
        "GroundingDinoForObjectDetection": "Same as `GroundingDinoModel`.",
        "MMGroundingDinoModel": "Same as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same as `GroundingDinoModel`.",
        "BigBirdModel": ("Lowering exceeds the 10-minute test timeout under dynamic shapes."),
        "BigBirdForCausalLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForPreTraining": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `BigBirdModel`.",
        "DonutSwinModel": "Same `timeout` failure as `BigBirdModel`.",
        "DonutSwinForImageClassification": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerSwinModel": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerSwinBackbone": "Same `timeout` failure as `BigBirdModel`.",
        "Mask2FormerModel": "Same `timeout` failure as `BigBirdModel`.",
        "Mask2FormerForUniversalSegmentation": "Same `timeout` failure as `BigBirdModel`.",
        "SwinModel": "Same `timeout` failure as `BigBirdModel`.",
        "SwinBackbone": "Same `timeout` failure as `BigBirdModel`.",
        "SwinForImageClassification": "Same `timeout` failure as `BigBirdModel`.",
        "SwinForMaskedImageModeling": "Same `timeout` failure as `BigBirdModel`.",
        "Swinv2Model": "Same `timeout` failure as `BigBirdModel`.",
        "Swinv2Backbone": "Same `timeout` failure as `BigBirdModel`.",
        "Swinv2ForImageClassification": "Same `timeout` failure as `BigBirdModel`.",
        "Swinv2ForMaskedImageModeling": "Same `timeout` failure as `BigBirdModel`.",
    },
    # ExecuTorch — lowering failures grouped by root cause; see the first entry of each
    # `Same ... as` chain for the full description.
    "executorch": {
        "Qwen3ASRForConditionalGeneration": (
            "Its `.pte` loads until ExecuTorch fails to allocate a tensor: `getTensorDataPtr() failed: 0x21` "
            "(`MemoryAllocationFailed`), surfaced as `execute() 0x12`. Not our sizing — the tensor is "
            "`[64, 128, 32]` and the largest planned one in that program is ~2M elements — and not "
            "adjustable from here: the Python runtime's `load_method` takes no allocator. The export itself "
            "is fine (it stopped failing once `dim_order_from_stride` could order a data-dependent stride)."
        ),
        "Siglip2VisionModel": (
            "`aten::_upsample_bilinear2d_aa.out` refuses its own output at run time: "
            "`Check failed (out.size(2) == output_size[0])`. The portable kernel checks the extent it was "
            "handed against the one it computes, and the two disagree once the axis is dynamic."
        ),
        "Siglip2ForImageClassification": "Same `_upsample_bilinear2d_aa` output-extent check as `Siglip2VisionModel`.",
        "JetMoeModel": (
            "MoE and mixture-of-attention route tokens with a data-dependent `inputs.split(expert_size)`, "
            "whose sizes come from the gate's `expert_size.tolist()` — unbacked scalars. What rejects them "
            "is not the memory planner (`_fix_range_constraints` bounds unbacked dims, and the planner "
            "copes) but EXIR's edge-dialect *arg validator* in `to_edge_transform_and_lower`, which needs a "
            "concrete int for every `split_with_sizes_copy` size: `InternalError: Could not extract "
            "specialized integer from data-dependent expression`. Note the failure CI actually reports is "
            "`IndexError: tuple index out of range` at `modeling_jetmoe.py` — `_patch_unbacked_split` "
            "intercepts the split first and hands back a 1-tuple that the per-expert loop then indexes. "
            "The routing can't be precomputed outside the graph (it is recomputed per layer from that "
            "layer's hidden states), and `@use_experts_implementation` can't host it: every "
            "`ExpertsInterface` entry is fixed to an MLP signature over `gate_up_proj`/`down_proj`, while "
            "`JetMoeMoA` straddles `map()`/`reduce()` with attention in between. A verified export-only "
            "rewrite does exist — the rows are already sorted by expert, so a masked dense pass is the same "
            "value at static shapes (measured 2.4e-7 against eager) — but it costs `num_experts`x the GEMM "
            "FLOPs of the split. TODO: land that if JetMoe ever needs to deploy; it would also clear the "
            "`_can_compile_fullgraph = False` this same `.tolist()` forces. Exports fine on "
            "torch.export/ONNX (dynamic dim at runtime)."
        ),
        "JetMoeForCausalLM": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "JetMoeForSequenceClassification": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "Lfm2VlForConditionalGeneration": (
            "Its NaViT-style packer sizes the vision stack from the number of patches each image really "
            "has, so those extents are unbacked, and the trace stops inside torch's own `slice` "
            "decomposition on a question no reasoning can settle: `GuardOnDataDependentSymNode: Could not "
            "guard on data-dependent expression u88 < 0` at `_decomp/decompositions.py:782 in "
            "slice_forward` — the normalization that asks whether the index is negative. This is a *trace* "
            "failure, not a memory-planning one: it never reaches `to_edge`, and note that unbacked does "
            "not mean unplannable here, since `_fix_range_constraints` bounds unbacked dims (which is why "
            "qwen3_asr's `.nonzero()`-packed length exports fine). Measured independent of "
            "`_patch_unbacked_split` — dropping that patch reproduces the identical guard. dynamo and ONNX "
            "carry both models under dynamic shapes (measured), decomposing the slice differently. Lifting "
            "this needs the data dependence gone — the per-image geometry precomputed outside the graph, "
            "the way the grid VLMs feed `cu_seqlens` / `window_index` — not a change of backend."
        ),
        "Lfm2VlModel": "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`.",
        "MiniCPMV4_6ForConditionalGeneration": (
            "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`, same `slice_forward` guard "
            "(measured, at `u84 < 0`)."
        ),
        "MiniCPMV4_6Model": "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`.",
        "FlavaModel": (
            "The interleaved text/image/multimodal encoder streams make XNNPACK's disjoint-set partitioner "
            "emit partitions that form a dependency cycle once fused (`Invalid partition, found dependency "
            "cycles`). The single-stream sub-models (image/text/multimodal/codebook) export fine."
        ),
        "FlavaForPreTraining": "Same fused-partition dependency cycle as `FlavaModel` (wraps it).",
        "PPDocLayoutV3ForObjectDetection": (
            "A single detection head applied at every decoder layer and tied to the encoder head is "
            "duplicated by the constant-dedup pass; `_unsafe_adjust_original_program` then deletes the "
            "shared target once and raises `KeyError` on the next copy while stripping delegated params."
        ),
    },
    "executorch.generate": {},
    "executorch.dynamic": {
        "MaskFormerForInstanceSegmentation": (
            "Lowering does not finish: >1000s inside sympy / `symbolic_shapes`, measured on an idle machine "
            "(so not sweep contention). The time is symbolic-shape reasoning over the graph's dynamic axes, "
            "not compute."
        ),
        "Qwen3_5ForCausalLM": "Same >1000s symbolic-shape lowering as `MaskFormerForInstanceSegmentation`.",
        "Qwen3NextForCausalLM": "Same >1000s symbolic-shape lowering as `MaskFormerForInstanceSegmentation`.",
        # Timeouts, not lowering defects: windowed-attention vision stacks re-partition every window on a
        # symbolic H/W, and the lowering alone outruns the test budget. Measured in the ExecuTorch sweep of
        # 2026-08-25 (maskformer at the 1000s mark); the rest of the Swin family, `efficientnet` and
        # `hrm_text` timed out in the sweep before it and share the shape. Skipped rather than re-measured —
        # a run that only ever ends in a timeout costs the whole budget to tell us nothing.
        "MaskFormerSwinModel": "Lowering exceeds the test timeout under dynamic shapes.",
        "MaskFormerSwinBackbone": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "SwinModel": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "SwinBackbone": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "SwinForImageClassification": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "SwinForMaskedImageModeling": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "Swinv2Model": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "Swinv2Backbone": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "Swinv2ForImageClassification": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "Swinv2ForMaskedImageModeling": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "DonutSwinModel": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "DonutSwinForImageClassification": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "EfficientNetModel": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "EfficientNetForImageClassification": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "HrmTextModel": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "HrmTextForCausalLM": "Same `timeout` failure as `MaskFormerSwinModel`.",
        "Mask2FormerModel": ("Lowering exceeds the 10-minute test timeout under dynamic shapes."),
        "Mask2FormerForUniversalSegmentation": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForPreTraining": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForCausalLM": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `Mask2FormerModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `Mask2FormerModel`.",
        "GroundingDinoModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "GroundingDinoForObjectDetection": "Same `timeout` failure as `Mask2FormerModel`.",
        "MMGroundingDinoModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "MMGroundingDinoForObjectDetection": "Same `timeout` failure as `Mask2FormerModel`.",
        "Sam2VisionModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "Swin2SRModel": (
            "ExecuTorch plans its arena ahead of time from per-dimension upper bounds, and the windowed "
            "attention's compound `Mod`/`FloorDiv` extents (window padding plus the cyclic shift) leave "
            "`ConstraintBasedSymShapeEvalPass` with no bound at all, so each such dim takes the cap floor of "
            "1024 -- including the ones whose traced value is `window_size ** 2` = 4. The plan comes out at "
            "466 GiB, dominated by the delegated `window_partition`/`window_reverse` and cosine-attention "
            "buffers (the latter planned `(13312, 4, 1024, 1024)` against a real `(13312, 4, 4, 4)`), and "
            "`load_program` dies allocating it -- that request being just *under* the runner's RAM, the OS "
            "admits it and the worker is OOM-killed rather than raising. Only the ahead-of-time plan is too "
            "big: `executorch.static` runs end to end, and torch.export and ONNX both pass under dynamic "
            "shapes because they allocate from the real shapes at run time. Tightening our own caps only "
            "makes the failure clean (arena 1.7 GiB, load then fails 0x21 on a single tensor). TODO: lift by "
            "bounding a reshape's factor dims jointly in the planner, or by passing `dynamic_shapes` that "
            "keep height/width static."
        ),
        "Swin2SRForImageSuperResolution": (
            "Same 466 GiB windowed-attention arena as `Swin2SRModel` -- its upsampler head adds nothing to the plan."
        ),
        "TimesformerModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "TimesformerForVideoClassification": "Same `timeout` failure as `Mask2FormerModel`.",
    },
    "executorch.static": {
        "SplinterForPreTraining": (
            "`aten::nonzero.out` cannot size its output under a static-shape export: the extent is "
            "data-dependent, so `resize_tensor` refuses it (`op_nonzero.cpp`). The dynamic variant passes."
        ),
        "MusicFlamingoForConditionalGeneration": (
            "Same data-dependent `aten::nonzero.out` resize as `SplinterForPreTraining`; its audio encoder "
            "additionally hits XNNPACK declining to propagate shapes (`xnn_status_invalid_parameter`)."
        ),
        "MusicFlamingoModel": "Same data-dependent `aten::nonzero.out` resize as `MusicFlamingoForConditionalGeneration`.",
        "PaddleOCRVLForConditionalGeneration": (
            "Its image encoder's `aten::view_copy.out` fails `check_view_copy_args` at run time — the view's "
            "target extent is not the one the planned output carries once the axis is static."
        ),
        "Wav2Vec2BertModel": (
            "Its conv feature extractor reshapes on the stacked floor-divisions its own stride chain "
            "produces (`((((s//4)+1)//2)+1)//2 …`), which ExecuTorch's lowering cannot satisfy: "
            "`RuntimeError: shape '[4*s99, 16, …]' is invalid`. Re-measured — this was recorded as a "
            "timeout, but it fails outright, well inside the limit."
        ),
        "Wav2Vec2BertForCTC": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForSequenceClassification": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForAudioFrameClassification": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForXVector": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "GroundingDinoModel": (
            "Static-shape export raises `KeyError: 'bbox_embed.1.layers.0.weight'`: the per-decoder-layer "
            "bbox-embed head is shared/tied, so the constant-dedup pass duplicates it and "
            "`_unsafe_adjust_original_program` deletes the shared target once then KeyErrors on the next "
            "copy (same shared-detection-head issue as `PPDocLayoutV3ForObjectDetection`). The dynamic "
            "variant is skipped for `timeout` above."
        ),
        "GroundingDinoForObjectDetection": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
        "MMGroundingDinoModel": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
    },
}


# ──────────────────────────── ONNX optimization toggles ────────────────────────────
# Not "skips" — these select whether `onnxscript` optimisation runs for a given model.
# Same scope-keyed shape as ``EXPORT_SKIPS`` for symmetry.


# Model classes whose ExecuTorch export must skip the backend partitioner
# (`ExecutorchConfig(partition=False)`), keyed by scope like `EXPORT_SKIPS`. XNNPACK's partitioner can claim
# subgraphs its own compiler then refuses at *method load*, which reads as an unrelated `0x21`/`0x14` when
# the program is run. An entry here says "this graph lowers, but only to the portable kernels" — the model
# is still exported and still run, just without delegation, so it keeps real coverage instead of a
# tolerated load failure.
#
# A candidate belongs here only once lowering it undelegated is shown to *run*. The test passing is not
# enough on its own: a tolerated load failure passes too, so check that the class's
# `ExecuTorch runtime limitation tolerated` warning is gone as well. An XNNPACK refusal alone proves
# nothing — the ModernVBert family's vision encoder and sam3_lite_text fail to load with `0x21` whether
# delegated or not (that code is an arena the plan cannot allocate, not a refusal), and sam3_lite_text goes
# on to fail at execute with `0x12` once undelegated.
# Per-op partitioner configs to withhold from XNNPACK, keyed like `EXECUTORCH_DISABLE_PARTITION` and
# preferred over it: withholding one config leaves that op to the portable kernels and keeps every other op
# delegated, where disabling the partitioner costs the whole graph its acceleration. An entry belongs here
# once *removing that one config* is shown to make the program run — measured by re-exporting with each of
# the 52 configs withheld in turn and keeping the ones that come back with no tolerance warning. Where no
# single config suffices (univnet and perceiver: all 52 refused individually), the coarse
# `EXECUTORCH_DISABLE_PARTITION` below is still the only way past.
EXECUTORCH_PARTITION_EXCLUDE: dict[str, dict[str, tuple[str, ...]]] = {
    # Dynamic shapes only — the static variants lower and run fully delegated.
    "dynamic": {
        # The vision encoder's inputs are dynamic on every axis, and XNNPACK gives up propagating shapes
        # through `unsqueeze_copy` (`Propagating input shapes failed with code:
        # xnn_status_invalid_parameter`). `_patch_unsqueeze` cannot reach these: they come from
        # decompositions, below any Python-level patch, so the config has to be withheld instead.
        "MuseGlimmerForConditionalGeneration": ("UnsqueezeCopyConfig",),
        "MuseGlimmerModel": ("UnsqueezeCopyConfig",),
    },
    # Both shape variants.
    "all": {
        # Rank-7 activations from the location-variable convolution — `(2, 16, 7, 256, 1, 1, 1)`, past the
        # 6 dimensions XNNPACK can define. Every config claiming an op that touches them must be withheld.
        "UnivNetModel": ("CloneDimOrderConfig", "PermuteConfig", "UnsqueezeCopyConfig", "ViewCopyConfig"),
    },
    # Static shapes only — the dynamic variants lower and run fully delegated.
    "static": {
        # XNNPACK claims `aten.view_copy` and its compiler then refuses the partition it claimed (`0x1` at
        # method load). Withholding `ViewCopyConfig` alone lets these run; each of the other 51 configs
        # changes nothing, so it is that op pattern and not the graph. Both families share the GatedDeltaNet
        # backbone the refusal comes from.
        "Qwen3_5Model": ("ViewCopyConfig",),
        "Qwen3_5TextModel": ("ViewCopyConfig",),
        "Qwen3_5ForCausalLM": ("ViewCopyConfig",),
        "Qwen3_5ForConditionalGeneration": ("ViewCopyConfig",),
        "Qwen3_5ForSequenceClassification": ("ViewCopyConfig",),
        "Qwen3_5ForTokenClassification": ("ViewCopyConfig",),
        "Qwen3_5TextForSequenceClassification": ("ViewCopyConfig",),
        "Qwen3NextModel": ("ViewCopyConfig",),
        "Qwen3NextForCausalLM": ("ViewCopyConfig",),
        "Qwen3NextForQuestionAnswering": ("ViewCopyConfig",),
        "Qwen3NextForSequenceClassification": ("ViewCopyConfig",),
        "Qwen3NextForTokenClassification": ("ViewCopyConfig",),
        "Qwen3_5MoeModel": ("ViewCopyConfig",),
        "Qwen3_5MoeTextModel": ("ViewCopyConfig",),
        "Qwen3_5MoeForCausalLM": ("ViewCopyConfig",),
        "Qwen3_5MoeForConditionalGeneration": ("ViewCopyConfig",),
        "OlmoHybridModel": ("ViewCopyConfig",),
        "OlmoHybridForCausalLM": ("ViewCopyConfig",),
        "PerceiverModel": ("ViewCopyConfig",),
        # The flow head's rank-heavy activations: XNNPACK cannot define them, and the three configs that
        # claim the ops touching them have to be withheld together (measured — no smaller set runs).
        "PerceiverForOpticalFlow": ("CloneConfig", "PermuteConfig", "ViewCopyConfig"),
    },
}


EXECUTORCH_DISABLE_PARTITION: dict[str, dict[str, str]] = {
    # Static shapes only — the dynamic variants lower and run delegated.
    "static": {
        "PerceiverForMultimodalAutoencoding": (
            "The only entry left that no per-op exclusion reaches: withholding all 52 partitioner configs "
            "still does not get this program running, because what it needs is a kernel ExecuTorch does not "
            "ship (`aten::rand_like.out`, from the `torch.bernoulli` masking its preprocessor runs at "
            "inference). Disabling the partitioner turns the delegate's `0x1` into that missing-kernel "
            "`0x14`, which is tolerated and reported — the honest end state until the kernel exists."
        ),
    },
}


ONNX_DISABLE_OPTIMIZE: dict[str, dict[str, str]] = {
    # Disable for every variant.
    "all": {
        "LayoutLMv2Model": (
            "Detectron2 FPN backbone — onnxscript optimizer drops initializers still referenced "
            "by nodes, producing an invalid graph for ORT."
        ),
        "LayoutLMv2ForSequenceClassification": "Same as `LayoutLMv2Model`.",
        "LayoutLMv2ForTokenClassification": "Same as `LayoutLMv2Model`.",
        "LayoutLMv2ForQuestionAnswering": "Same as `LayoutLMv2Model`.",
        "YolosModel": (
            "Optimizer takes >6 min on the YOLOS detection graph (many small Concat/Slice nodes). "
            "`optimize=False` exports in 2s. TODO: revisit when onnxscript's optimizer improves."
        ),
        "YolosForObjectDetection": "Same as `YolosModel`.",
        "PixioModel": "Same dense-small-node optimizer slowdown as YOLOS (~100–290s).",
        "SegGptModel": "Same dense-small-node optimizer slowdown as YOLOS.",
        "SegGptForImageSegmentation": "Same dense-small-node optimizer slowdown as YOLOS.",
    },
    # Disable for dynamic-shape only — static benefits from optimisation.
    "dynamic": {
        "ProphetNetModel": (
            "Onnxscript's `SplitToSequence` constant-folding trips `'NoneType' object has no "
            "attribute 'ndim'` under dynamic shapes. Static works after the vectorized "
            "`ngram_attention_bias` rewrite."
        ),
        "ProphetNetForConditionalGeneration": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ProphetNetDecoder": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ProphetNetForCausalLM": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ZoeDepthForDepthEstimation": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "LlavaNextVideoModel": (
            "Same `SplitToSequence` folding crash as `ProphetNetModel` — here from the per-image "
            "`torch.split` of the anyres video features."
        ),
        "LlavaNextVideoForConditionalGeneration": "Same `SplitToSequence` issue as `LlavaNextVideoModel`.",
    },
}


# Parameterization for export tests: runs once with dynamic=True and once with dynamic=False.
DYNAMIC_EXPORT_PARAMS = parameterized.expand(
    [(False,), (True,)],
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}",
)

# Generation export tests run the product of three axes: shape dynamism, single- vs multi-token decode
# capture, and the generation config used for the capture.
_EXPORT_SHAPE_MODES = [False, True]  # dynamic=False (static shapes) / dynamic=True
# `multi_token_decode=False` captures the classic single-token decode (its query axis specializes to 1;
# the separate `prefill` graph serves the prompt); `True` merges two decode steps so the query axis stays
# symbolic and one graph serves prefill and decode.
_EXPORT_DECODE_MODES = [False, True]  # multi_token_decode
# `generation_config=None` is the model's own config (growing `DynamicCache`);
# `cache_implementation="static"` exports against a `StaticCache`. Every cache runs under every shape
# mode — under dynamic shapes a static cache still keeps a symbolic (resizable) size, it just writes at
# fixed positions. The static entry declares `max_cache_len` explicitly — a static-cache export's
# contract: the runtime is handed the same generation config the model was exported with and builds the
# cache it declares; without it, the capture sizes the cache from its own internal token count and the
# runtime from the caller's, and backends that freeze the traced length reject the mismatch.
# `max_cache_len` must fit every tester's prompt + new tokens: `generate` silently grows an under-sized
# static cache, and it grows it *differently* in each phase — the capture adds its own internal token count,
# the runtime the caller's — so the graph bakes one length and the runtime builds another (gpt_bigcode and
# minimax prompt at ~127 and ~151, and were off by exactly the one-token difference). Multi-modal prompts
# with image tokens run ~80 long, so this sits well clear of every tester.
# Both entries declare `use_cache=True`: an exported decode graph is only useful with a cache, and a model
# whose own config disables caching (bart's standalone decoder) would otherwise be captured cacheless —
# re-feeding the whole growing sequence every step, which a frozen-shape graph can't serve at all.
_EXPORT_GENERATION_CONFIGS = [
    GenerationConfig(use_cache=True),
    GenerationConfig(cache_implementation="static", max_cache_len=256, use_cache=True),
]

GENERATE_EXPORT_PARAMS = parameterized.expand(
    [
        (dynamic, multi_token, config)
        for dynamic, multi_token, config in itertools.product(
            _EXPORT_SHAPE_MODES, _EXPORT_DECODE_MODES, _EXPORT_GENERATION_CONFIGS
        )
        # A merged multi-token decode under static shapes would freeze its query axis at 2 — a graph no
        # decode step could ever run.
        if not (multi_token and not dynamic)
    ],
    name_func=lambda f, _, p: (
        f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}"
        + ("_multi_token" if p.args[1] else "")
        + (f"_{p.args[2].cache_implementation}_cache" if p.args[2].cache_implementation else "")
    ),
)


def _needs_static_cache(generation_config) -> bool:
    """True if `generation_config` requests a cache the model must explicitly support (a static impl).
    Such variants only run on models that can (see the `_can_compile_fullgraph` gate in the tests)."""
    return generation_config is not None and generation_config.cache_implementation is not None


# Maximum time (in seconds) for a single export test before it is killed.
EXPORT_TEST_TIMEOUT = 1000

# Minimum torch version the exporters target — older releases lack `torch.export` features the
# exporters rely on, so the export sweep is skipped (not failed) below this. Sourced from the
# exporter itself so the test and the runtime check can't drift apart.
MIN_EXPORT_TORCH_VERSION = DynamoExporter.min_versions["torch"]


@functools.lru_cache(maxsize=1)
def _inductor_toolchain_problem() -> str | None:
    """Why Inductor's C++ compiler cannot build what Inductor emits, or `None` when it can.

    A compiled export needs a toolchain that takes those flags, and one that does not fails every
    AOTInductor test in the sweep the same way (`-std=c++20` on a pre-gcc-10 compiler is the usual
    reason). Asking the compiler once, and skipping, keeps an environment problem from reading as
    hundreds of model failures — but it has to *say* so, or a whole sweep skips and looks like it ran.
    """
    from torch._inductor.cpp_builder import get_cpp_compiler

    compiler = get_cpp_compiler()
    try:
        probe = subprocess.run(
            [compiler, "-std=c++20", "-x", "c++", "-E", "-"],
            input="",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"{compiler} could not be run ({type(error).__name__})"
    if probe.returncode == 0:
        return None
    return f"{compiler} rejects the flags Inductor emits: {probe.stderr.strip().splitlines()[0]}"


if (_TOOLCHAIN_PROBLEM := _inductor_toolchain_problem()) is not None:
    warnings.warn(
        f"Every AOTInductor export test will be skipped: {_TOOLCHAIN_PROBLEM}. Point `CXX` at a compiler "
        "that accepts `-std=c++20` to run them.",
        stacklevel=2,
    )

require_inductor_toolchain = unittest.skipUnless(
    _TOOLCHAIN_PROBLEM is None, f"Inductor cannot compile here: {_TOOLCHAIN_PROBLEM}"
)


# ──────────────────────────── helpers ────────────────────────────


def disable_hub_kernels(test_fn):
    """Force `is_kernels_available()` to `False` for the duration of an export test.

    ExportArtifacts must trace the pure-PyTorch path, never a Hub kernel (`mamba-ssm`, `causal-conv1d`, …): those
    need optional deps (`einops`, triton, …) and aren't exportable anyway. Kernels load lazily on the first
    (eager) forward — outside the exporter's own trace-time patch — so the whole test is wrapped. With
    `is_kernels_available()` False, `lazy_load_kernel` short-circuits to `None` and the fallback runs.
    """

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        from transformers.integrations import hub_kernels
        from transformers.utils import import_utils

        # `lazy_load_kernel` gates on `hub_kernels`'s own binding; patch the canonical def too.
        targets = [(hub_kernels, "is_kernels_available"), (import_utils, "is_kernels_available")]
        saved = [(obj, name, getattr(obj, name)) for obj, name in targets]
        for obj, name in targets:
            setattr(obj, name, lambda *args, **kwargs: False)
        try:
            return test_fn(*args, **kwargs)
        finally:
            for obj, name, original in saved:
                setattr(obj, name, original)

    return wrapper


def _clean_inputs_for_export(inputs_dict, config):
    """Strip None values and export-incompatible keys from an inputs dict. Mutates config in-place."""
    inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}
    for key in ("labels", "future_values", "return_loss", "target_values"):
        inputs_dict.pop(key, None)
    config.return_loss = False
    return inputs_dict


# A line ExecuTorch's C++ side wrote, as opposed to anything else sharing stderr.
_EXECUTORCH_LOG_LINE = re.compile(r"\[\w+\.cpp:\d+\]")
# Where each component's ExecuTorch log is written, one file per label. Under `-n auto` every worker shares
# stderr, so a sweep's log interleaves and cannot be attributed to the test that produced it — which is the
# whole point of keeping it. A file per label keeps them separable.
_EXECUTORCH_LOG_DIR = Path(os.environ.get("TRANSFORMERS_EXECUTORCH_LOG_DIR", "executorch_logs"))


def _write_executorch_log(label: str, log: str) -> str | None:
    """Write one component's ExecuTorch log beside the others, and return where it went."""
    lines = [line for line in log.splitlines(keepends=True) if _EXECUTORCH_LOG_LINE.search(line)]
    if not lines:
        return None
    _EXECUTORCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^\w.-]+", "_", label) + ".log"
    path = _EXECUTORCH_LOG_DIR / filename
    path.write_text("".join(lines))
    return str(path)


@contextmanager
def _capturing_executorch_log():
    """Capture what ExecuTorch writes to stderr, and yield a reader for it.

    ExecuTorch logs from C++ — the operator registry and `method.cpp` write straight to fd 2 — so
    `contextlib.redirect_stderr`, which only rebinds `sys.stderr`, sees none of it. The fd itself has to be
    redirected.

    Needs `-s` (`--capture=no`) to see anything: under pytest's own fd capture ExecuTorch emits no log at
    all — not to this sink and not to pytest's, so a sweep that wants the kernel names in the tolerance
    warnings has to disable capture. Redirecting to a plain file is otherwise fine; the emptiness is
    ExecuTorch's, not this redirect's.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile("w+b") as sink:
        os.dup2(sink.fileno(), 2)

        def read_log() -> str:
            """Read while the redirect is still up: a caller handling the exception has not left the block
            yet, so the log cannot be handed over only on exit."""
            sys.stderr.flush()
            here = sink.tell()
            sink.seek(0)
            text = sink.read().decode("utf-8", "replace")
            sink.seek(here)
            return text

        try:
            yield read_log
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            sink.seek(0)
            # Nothing another writer put on stderr is swallowed; ExecuTorch's own lines are held back,
            # because a tolerated failure files them per label rather than interleaving them into the run.
            passthrough = sink.read().decode("utf-8", "replace").splitlines(keepends=True)
            sys.stderr.write("".join(line for line in passthrough if not _EXECUTORCH_LOG_LINE.search(line)))


def _executorch_log_detail(log: str) -> str:
    """The actionable part of an ExecuTorch failure log: which kernel is missing, else its last complaint.

    A bare error code says only which phase failed. The kernel name says whether it is a platform ceiling or
    something the export should have avoided emitting (a graph reaching lowering with an `argsort` in it asks
    the portable registry for `aten::sort.values`, which it does not ship, while `aten::topk.values` — the
    same computation — it does).
    """
    missing = re.findall(r"Missing operator: \[\d+\] (\S+)", log) or re.findall(r"kernel '([^']+)' not found", log)
    if missing:
        return f"missing kernel(s) {sorted(set(missing))}"
    complaints = [line.strip() for line in log.splitlines() if re.match(r"\[\w+\.cpp:\d+\]", line.strip())]
    # Prefer the line that says *why* over the one that says where it gave up: a delegate refusal logs its
    # `xnn_status_*` first and then a generic `CALL_DELEGATE execute failed` last, and only the first is
    # actionable.
    # `method.cpp` dumps one `arg N with type id` line per operand after a kernel failure; they are the
    # last lines but say nothing about the cause, so they never win.
    complaints = [line for line in complaints if not re.search(r"arg \d+ with type id", line)]
    causes = [
        line for line in complaints if re.search(r"xnn_status|Internal Error|Attempted to resize|Check failed", line)
    ]
    # The *first* cause, not the last: a kernel logs the specific complaint
    # (`tensor_util_portable.cpp: 4 input tensors have different dim orders`) before the wrapper check that
    # gave up on it (`op_where.cpp: tensors_have_same_dim_order(...)`), and only the first names the reason.
    return (causes or complaints)[0] if (causes or complaints) else ""


@contextmanager
def _tolerating_executorch_limits(label: str):
    """Swallow only the failures where ExecuTorch itself refuses to run an otherwise valid program — a named
    code from the *load* or *execute* phase, or a `bad_alloc` (`_is_executorch_runtime_limit`). A transformers
    export defect surfaces earlier as a `torch.export` error, or later as an output mismatch.

    Anything about how *we* fed the method stays visible: `set_inputs` fails when we hand it something it
    never declared, and a missing input means the decomposition produced a component we cannot feed. A model
    that genuinely needs an exception belongs in `EXPORT_SKIPS`, argued, where it can be seen.
    """
    with _capturing_executorch_log() as executorch_log:
        try:
            yield
        except (RuntimeError, MemoryError) as error:
            if not _is_executorch_runtime_limit(error):
                # A visible failure needs its log too: ExecuTorch's exception carries only the code, while
                # the kernel and the shapes it refused live in the C++ log this context captured. Dropping it
                # here left `0x12` (never tolerated, so never written) reported as a bare code.
                log = executorch_log()
                detail = _executorch_log_detail(log)
                written = _write_executorch_log(label, log)
                if not (detail or written):
                    raise
                raise type(error)(
                    f"{error}\n[executorch] {label}: {detail}" + (f" (full log: {written})" if written else "")
                ) from error
            # A tolerated failure still reports the test as passed, so say so — otherwise a green run is
            # indistinguishable from one where the program actually ran. The log detail is what makes the
            # warning actionable: the error code alone cannot tell a platform ceiling from an op the export
            # should not have emitted.
            log = executorch_log()
            detail = _executorch_log_detail(log)
            written = _write_executorch_log(label, log)
            warnings.warn(
                f"{label}: ExecuTorch runtime limitation tolerated; this test passes without running the "
                f"program — add it to `EXECUTORCH_DISABLE_PARTITION` if lowering it undelegated runs "
                f"instead: {str(error).strip().splitlines()[0]}"
                + (f" [{detail}]" if detail else "")
                + (f" (full log: {written})" if written else ""),
                stacklevel=2,
            )


# ExecuTorch runtime error codes that mean "the export is valid (it produced a loadable program) but
# ExecuTorch's own portable runtime / XNNPACK backend can't service it" — a runtime limitation, not a
# transformers export defect (which surfaces earlier as a `torch.export` error or later as an output
# mismatch). Each is a code whose own definition (`runtime/core/error.h`) attributes it to the runtime:
# load 0x14 `OperatorMissing` (the registry has no kernel for an op the program needs) and 0x21
# `MemoryAllocationFailed` (the arena cannot be allocated); execute 0x10 `NotSupported` (the backend
# declines the operation in this context — XNNPACK cannot resize a static tensor to the runtime shape).
# 0x1 `Internal` is generic ("an internal error occurred"), so it counts at *execute* only: by then
# `set_inputs` has accepted the feed, and XNNExecutor reports a delegate refusal this way — the exception
# carries only the code (the `xnn_status_*` detail goes to ExecuTorch's own log), so the phase is the whole
# signal. At *load* the same code is as easily a malformed program of ours, so it does not count there.
# 0x12 `InvalidArgument` never counts. It shows up as a kernel refusing to resize its own output ("Attempted
# to resize a static tensor. Expected shape (2, 2, 32), but received (2, 1, 32)" from `tensor_impl.cpp`, via
# `aten::embedding.out`), which happens when a dynamic axis reaches lowering without the bound that would let
# the planner size it for the largest shape. That is a fixable defect on our side of the export, not a
# platform ceiling, so it stays visible. Only failures from `execute()` itself count: the same codes also come out of
# `set_inputs()`, but binding the runtime inputs is *our* side of the contract — it fails when we hand the
# method something it never declared (feeding an fp32 cache to a half-precision program did exactly that,
# and reading it as a backend limitation hid the bug across every MoE model), so those have to stay visible.
_ET_LOAD_LIMIT_CODES = {"0x14", "0x21"}
_ET_EXECUTE_LIMIT_CODES = {"0x1", "0x10"}


def _is_executorch_runtime_limit(exc):
    """True if ``exc`` is a known ExecuTorch runtime limitation (missing kernel / arena / kernel bug)."""
    msg = str(exc)
    if isinstance(exc, MemoryError) or "bad_alloc" in msg:
        return True
    load = re.search(r"Failed to load method forward, error: 0x:?([0-9a-fA-F]+)", msg)
    if load and f"0x{load.group(1)}" in _ET_LOAD_LIMIT_CODES:
        return True
    execute = re.search(r"execute\(\) failed with error 0x([0-9a-fA-F]+)", msg)
    if not execute:
        return False
    code = f"0x{execute.group(1)}"
    return code in _ET_EXECUTE_LIMIT_CODES


def _onnx_optimize_enabled(model_class, dynamic: bool) -> bool:
    """Return whether onnxscript optimisation should run for this model under this shape mode.

    Mirrors ``_should_skip``'s scope walk on ``ONNX_DISABLE_OPTIMIZE`` — ``"all"`` always
    applies; ``"dynamic"`` adds the dynamic-only entries.
    """
    name = model_class.__name__
    scopes = ["all", "dynamic" if dynamic else "static"]
    return not any(name in ONNX_DISABLE_OPTIMIZE.get(scope, {}) for scope in scopes)


def _executorch_partition_exclude(model_class, dynamic: bool) -> tuple[str, ...]:
    """The partitioner configs to withhold for this class, by the same scope walk as
    ``_executorch_partition_enabled``. Empty means hand the partitioner everything."""
    name = model_class.__name__
    excluded: tuple[str, ...] = ()
    for scope in ("all", "dynamic" if dynamic else "static"):
        excluded += EXECUTORCH_PARTITION_EXCLUDE.get(scope, {}).get(name, ())
    return excluded


def _executorch_partition_enabled(model_class, dynamic: bool) -> bool:
    """Return whether the ExecuTorch export may hand subgraphs to the backend's partitioner.

    Mirrors ``_onnx_optimize_enabled``'s scope walk on ``EXECUTORCH_DISABLE_PARTITION`` — ``"all"``
    always applies; ``"dynamic"`` / ``"static"`` add the entries for that shape variant.
    """
    name = model_class.__name__
    scopes = ["all", "dynamic" if dynamic else "static"]
    return not any(name in EXECUTORCH_DISABLE_PARTITION.get(scope, {}) for scope in scopes)


def needs_half_precision_export(model) -> bool:
    """Whether `model` exercises a kernel that only runs in half precision, so the export test builds it in
    half precision rather than fp32. Two such kernels: grouped-mm MoE experts (`config._experts_implementation`
    resolves to `"grouped_mm"`; the eager/batched paths are fp32-fine) and the vision/audio varlen flash
    attention (the forwards patched in `_VARLEN_ATTENTION_PATHS`, matched by full module-qualified class path so
    a generic name like `VisionAttention` can't collide). Everything else exports fine — and more faithfully —
    in fp32."""
    if getattr(getattr(model, "config", None), "_experts_implementation", None) == "grouped_mm":
        return True
    return any(
        f"{type(module).__module__}.{type(module).__qualname__}.forward" in _VARLEN_ATTENTION_PATHS
        for module in model.modules()
    )


# ──────────────────────────── mixins ────────────────────────────


def _assert_openvino_output_names(case, runtime, actual: dict, expected: dict) -> None:
    """Every leaf eager returns is accounted for — as an output, or as folded state.

    An OpenVINO export turns each round-tripped cache tensor into an internal variable the plugin keeps
    between calls, so the graph returns logits and the cache stays behind its `Assign` sinks. Reading it
    back off those sinks is what lets this compare the whole set rather than excusing what is missing.
    """
    case.assertTrue(actual, "OpenVINO outputs are empty.")
    runner = getattr(runtime, "runner", None)
    folded = runner.state_tensors() if runner is not None and runner.owns_state else {}
    case.assertEqual(set(actual) | set(folded), set(expected))


class ExportTesterMixin:
    """Mixin providing non-generative export tests for Dynamo, ONNX, and ExecuTorch backends.

    Mixed into [`ModelTesterMixin`] so every model test class that inherits from it
    automatically runs these export tests against all entries in `all_model_classes`.

    Expected attributes provided by [`ModelTesterMixin`]:
    - `all_model_classes` — iterable of model class objects to test.
    - `model_tester` — object with `prepare_config_and_inputs_for_common()` (and optionally
      `prepare_config_and_inputs_for_model_class()`).
    - `test_torch_exportable` — bool; set to `False` to skip all export tests for the model.
    - `_prepare_for_class(inputs_dict, model_class)` — adjusts inputs per model class.

    Tests are parameterised over `dynamic=True` / `dynamic=False` via `DYNAMIC_EXPORT_PARAMS`.
    Multi-modal models (detected by `is_multimodal`) are automatically decomposed and each
    submodule is tested independently.
    """

    def _skip_if_not_exportable(self):
        """Skip the test if the model architecture is not exportable."""
        if not self.test_torch_exportable:
            self.skipTest(reason="Model architecture is not Dynamo exportable/traceable")

        with open(inspect.getfile(self.all_model_classes[0]), encoding="utf-8") as f:
            source_code = f.read()
            # TODO: add use_experts_implementation support to remaining MoE models
            if "for expert" in source_code and "use_experts_implementation" not in source_code:
                self.skipTest(reason="Model architecture uses eager MoE implementation which is not torch exportable")

    def _should_skip(
        self,
        model_class,
        generate=False,
        dynamic=False,
        backend=None,
        multi_token=False,
        generation_config=None,
        runtime=False,
    ):
        """Return True if this model class should be skipped for export tests.

        Walks the scopes in ``EXPORT_SKIPS`` from broad to specific that match the current test —
        ``"all"`` always applies, ``"generate"`` only for generate tests, ``"dynamic"`` / ``"static"``
        for that shape variant, ``"generate.multi_token"`` for the merged multi-token decode capture, and
        ``"generate.runtime"`` for driving the exported graphs through `generate` (the export itself still
        runs — use it when a model exports fine and only the runtime cannot serve it), and
        ``"generate.runtime.multi_token"`` for a model the runtime drives fine *except* from a merged decode.
        The runtime scope carries the shape variant too (``"generate.runtime.dynamic"`` /
        ``"generate.runtime.static"``), so an entry can say "only the dynamic drive fails" instead of gating
        every variant — and, backend-prefixed, "only this backend's dynamic drive". Every one of these
        also exists ``"<backend>."``-prefixed (``"onnx.generate.multi_token"``, …) to skip on one backend
        only, plus the bare ``"<backend>"`` for that whole backend. Also skips static-cache variants
        (a ``generation_config`` requesting one) on models that can't compile fullgraph — they don't
        support a static cache.
        """
        if _needs_static_cache(generation_config) and not model_class._can_compile_fullgraph:
            return True
        name = model_class.__name__
        scopes = ["all"]
        if generate:
            scopes.append("generate")
            if dynamic:
                scopes.append("generate.dynamic")
            if multi_token:
                scopes.append("generate.multi_token")
            if runtime:
                scopes.append("generate.runtime")
                if multi_token:
                    scopes.append("generate.runtime.multi_token")
                scopes.append("generate.runtime.dynamic" if dynamic else "generate.runtime.static")
        scopes.append("dynamic" if dynamic else "static")
        if backend:
            scopes += [backend] + [f"{backend}.{scope}" for scope in scopes if scope != "all"]
        matched = next(
            ((scope, EXPORT_SKIPS[scope][name]) for scope in scopes if name in EXPORT_SKIPS.get(scope, {})), None
        )
        if matched is None:
            return False
        # A gated entry reports as passed rather than skipped (the test returns early), so name it and its
        # argued reason in the run's warning summary.
        warnings.warn(
            f"{name} is gated by EXPORT_SKIPS[{matched[0]!r}]; this test passes without exporting: {matched[1]}",
            stacklevel=2,
        )
        return True

    def _prepare_export_model_and_inputs(self, model_class, backend, device=torch_device):
        """Create model and forward inputs ready for export.

        ``device`` defaults to ``torch_device``; the ExecuTorch tests pass ``"cpu"`` since that
        backend targets CPU anyway, keeping any pre-trace forward off the GPU (a device-side
        assert during tracing would otherwise poison the whole xdist worker's CUDA context).

        Returns:
            Dict of `{name: Component}` — one entry per component.
        """
        if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval()
        # Use half precision only when the model has a half-precision-only kernel — the vision varlen flash
        # attention or grouped-mm MoE experts (see `needs_half_precision_export`); everything else stays fp32
        # (realistic, and avoids spurious dtype mismatches). The half type is per-backend: fp16 for ONNX
        # (ORT has no bf16 kernels for many ops), bf16 for torch.export/ExecuTorch (flash + grouped_mm need it).
        half_dtype = torch.float16 if backend == "onnx" else torch.bfloat16
        dtype = half_dtype if needs_half_precision_export(model) else torch.float32
        model = model.to(device, dtype)
        set_model_for_less_flaky_test(model)

        inputs_dict = cast_leaf_tensors(inputs_dict, dtype=module_dtype(model), device=module_device(model))

        if is_multimodal(model):
            return decompose_multimodal(model, inputs_dict)
        return {"model": Component("model", model, inputs_dict, ComponentRole.MODEL)}

    def _collect_eager_outputs(self, components):
        """Run eager forward for each component and return a ``{name: leaf_tensors}`` dict."""
        eager_outputs = {}
        for name, component in components.items():
            model, inputs = component.module, component.inputs
            with torch.no_grad():
                set_seed(1234)
                eager_outputs[name] = get_leaf_tensors(model(**copy.deepcopy(inputs)))
                assert eager_outputs[name], f"Eager outputs are empty for {name}."
        return eager_outputs

    def _assert_generate_matches_eager(
        self, components, exported, backend, generation_config, dynamic, multi_token_decode
    ):
        """Wrap the exported components in `backend`'s `ModelRunner`s, reassemble them into the
        `generate`-driving runtime via `ExportedGenerator.from_runners` — the same artifacts-plus-configs
        path a deployment would use (`generation_config` is the one the components were exported with, the
        runtime's cache contract; `None` means the model's own defaults) — and assert it generates like the
        eager model. Runs both on the runner's device with greedy decoding. fp32 models must match token
        ids exactly; half-precision models (the varlen-attention VLM families and grouped-mm MoEs, see
        `needs_half_precision_export`) compare per-step scores at the dtype-calibrated tolerance and ids
        only until the first near-tie — export re-rounds ops by ~2^-8, and a tiny random model's argmax
        legitimately flips on ties that small, while a real wiring bug (wrong cache / mask / positions)
        shows up as systematic score divergence which the closeness check catches regardless. Covers
        decoder-only text, VLMs (including M-RoPE, whose 4-axis position ids the runtime rebuilds
        config-only in `_prepare_position_ids_for_generation`) and encoder-decoder models (encoder +
        decoder-step graphs).

        A single-token export wires its `prefill` graph as the generator's dedicated prefill runner, so the
        `decode` graph only ever sees query=1 steps — which is what lets the *static-shape* variants run
        parity too (over a static cache, every step reproduces the frozen shapes). A multi-token export has
        one text graph serving both, exercising the single-graph path (dynamic shapes only)."""
        from transformers.exporters import ExportedGenerator
        from transformers.exporters.decompose import (
            _MODALITY_SPECS,
            _STREAMING_EMBEDDERS,
        )

        model = components["decode"].module
        if not dynamic and "embed_tokens" in components:
            # A multi-modal model embeds its text in a graph of its own, captured on the *prompt* — under
            # static shapes that graph is specialized to the prompt's length and cannot serve the 1-token
            # decode steps the loop makes (`Guard failed: input_ids.size()[1] == 39`). The static-shape
            # exports themselves are still asserted above; driving them needs a length-generic embedder.
            return

        wanted = {
            "decode",
            "prefill",
            "encoder",
            "embed_tokens",
            *(spec[0] for spec in _MODALITY_SPECS),
            *(spec[0] for spec in _STREAMING_EMBEDDERS.values()),
        }
        # The runners themselves, not the per-graph runtimes: `from_runners` assembles the generation
        # loop out of `ModelRunner`s, and a single-graph runtime is an `ExportedModel` *wrapping* one.
        runners = {name: exported[name].runtime().runner for name in components if name in wanted}
        runtime = ExportedGenerator.from_runners(runners, model.config, model.generation_config)
        device = runtime.device
        model = model.to(device)
        inputs = self.prepare_config_and_inputs_for_generate()[1]
        inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor) and k != "labels"}
        # the half-precision models (`needs_half_precision_export`) need their float inputs cast the same
        # way the export path casts them, or the eager side hits its own tower with fp32 `pixel_values`
        inputs = cast_leaf_tensors(inputs, dtype=module_dtype(model), device=device)

        # Called exactly like a normal model: the same generate inputs go to both, and no hand-rolled cache
        # — the runtime builds the cache the exported graph needs, static or growing (`_prepare_cache_for_generation`).
        # `eos_token_id=-1` keeps both running the full `max_new_tokens` so the ids compare directly.
        # The same capture generation config goes to both sides, exactly as it went to the export's own
        # generate — the runtime builds the cache it declares.
        gen_kwargs = {
            "do_sample": False,
            "eos_token_id": -1,
            "max_new_tokens": 2,
            "min_new_tokens": 2,
            "output_scores": True,
            "return_dict_in_generate": True,
            "generation_config": generation_config,
            "disable_compile": True,
        }
        eager_out = model.generate(**inputs, **gen_kwargs)
        try:
            exported_out = runtime.generate(**inputs, **gen_kwargs)
        except (RuntimeError, MemoryError) as e:
            # A portable kernel refusing the step's shapes mid-run is this backend's ceiling, the same one
            # the component checks absorb (`_tolerating_executorch_limits`) — the graphs themselves are asserted
            # above. Only the *execute* phase counts: a failure while binding inputs means the runtime fed
            # something the method never declared, which is our bug and must stay visible.
            if backend == "executorch" and _is_executorch_runtime_limit(e):
                return
            raise
        # Both sides are pinned to exactly `min_new_tokens` steps, so a runtime that produced a different
        # number of them is a wiring failure of its own -- and one the `zip` below would quietly absorb.
        self.assertEqual(
            len(exported_out.scores), len(eager_out.scores), "exported runtime generated a different number of steps"
        )
        # Ids step by step, while the two runs stay on the same prefix. This is a wiring check — numeric
        # fidelity is asserted per component above — so it puts no score bar on an fp32 export: kernel
        # drift is model-specific, and a tiny random model's top-2 gaps are small enough that argmax flips
        # on it while saying nothing. Half precision, whose rounding scale is knowable, compares scores.
        half = module_dtype(model) in (torch.float16, torch.bfloat16)
        atol = rtol = 1.6e-2
        tie_threshold = 2 * atol if half else 5e-3
        start = eager_out.sequences.shape[1] - len(eager_out.scores)
        for step, (eager_scores, exported_scores) in enumerate(zip(eager_out.scores, exported_out.scores)):
            if half:
                torch.testing.assert_close(exported_scores, eager_scores, atol=atol, rtol=rtol)
            eager_ids = eager_out.sequences[:, start + step].tolist()
            exported_ids = exported_out.sequences[:, start + step].tolist()
            if exported_ids == eager_ids:
                continue
            # They picked different tokens. Only eager being on a coin flip excuses that, so say which it
            # was -- and stop either way: the two runs now carry different prefixes, and every later step
            # would be comparing different continuations rather than the same one.
            top2 = eager_scores.float().topk(2, dim=-1).values
            self.assertLess(
                (top2[:, 0] - top2[:, 1]).min().item(),
                tie_threshold,
                f"exported run picked {exported_ids} where eager picked {eager_ids} at step {step}, and eager "
                "was confident about it",
            )
            break

    def _check_outputs_close(self, actual, expected, atol, rtol, check_device=True):
        """Assert outputs are close, allowing up to 5% element-level mismatch.

        For bf16/fp16 outputs the fp32-calibrated tolerance is far too tight — export re-rounds ops (fusion,
        reordered reductions), which perturbs half-precision values by ~2^-8. Widen to the dtype's rounding
        scale so genuine bugs (systematic, larger drift) still fail while benign bf16 noise passes.
        """
        if any(t.dtype in (torch.bfloat16, torch.float16) for t in expected.values()):
            atol, rtol = max(atol, 1.6e-2), max(rtol, 1.6e-2)
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, check_device=check_device)
        except AssertionError as e:
            mismatched_percentage = re.findall(r"Mismatched elements: (\d+) / (\d+)", str(e))
            if mismatched_percentage:
                mismatched, total = map(int, mismatched_percentage[0])
                if mismatched / total < 0.05:
                    return  # allow up to 5%
            raise e

    # ──────────────────── torch.export tests ─────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_torch_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """ExportArtifacts each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="dynamo"):
                continue

            components = self._prepare_export_model_and_inputs(model_class, "dynamo")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, f"Exported outputs are empty for {name}.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_precomputed_inputs_match_eager(self):
        """`prepare_for_export` must not change what the model computes.

        The preparers replace data-dependent work (`cu_seqlens`, vision `position_ids`, interpolation
        indices, ...) with tensors derived from the config, and the model then skips its own branch. Those
        tensors therefore have to equal what the model would have computed itself: a preparer that derives
        them differently -- say at the wrong merge size -- yields an export that quietly disagrees with the
        model it came from.
        """
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, backend="dynamo"):
                continue

            components = self._prepare_export_model_and_inputs(model_class, "dynamo")
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    with torch.no_grad():
                        set_seed(1234)
                        without_precompute = get_leaf_tensors(model(**copy.deepcopy(inputs)))

                    config = getattr(model, "config", None)
                    if config is None:
                        continue  # a bare module (lm_head) has no config and nothing to precompute

                    with torch.no_grad():
                        precomputed_inputs = precompute_export_inputs(config, copy.deepcopy(inputs))

                    if not set(precomputed_inputs) - set(inputs):
                        continue  # nothing is precomputed for this component

                    with torch.no_grad():
                        set_seed(1234)
                        with_precompute = get_leaf_tensors(model(**precomputed_inputs))

                    self.assertTrue(with_precompute, f"Outputs are empty for {name}.")
                    # exact: a preparer reproduces the model's own tensors, so any drift is a wrong
                    # precompute rather than numerical noise
                    self.assertEqual(with_precompute.keys(), without_precompute.keys())
                    for key, expected in without_precompute.items():
                        torch.testing.assert_close(
                            with_precompute[key],
                            expected,
                            atol=0,
                            rtol=0,
                            msg=lambda more, key=key: f"{name}: precomputed inputs change `{key}`\n{more}",
                        )

    # ────────────────────── AOTInductor tests ────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_inductor_toolchain
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_aoti_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """Compile each model class with AOTInductor and verify outputs match eager within tolerance.

        The graph is the one `test_torch_export` already sweeps, so what this adds is the lowering: a
        model whose graph traces cleanly can still have an op Inductor cannot generate kernels for, or a
        symbolic shape it refuses to compile against — neither of which a traced program ever hits.
        """
        self._skip_if_not_exportable()

        exporter = AotiExporter()
        config = AotiConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="aoti"):
                continue

            components = self._prepare_export_model_and_inputs(model_class, "aoti")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, f"Compiled outputs are empty for {name}.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ─────────────────────── TensorRT tests ──────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_torch_tensorrt
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_tensorrt_export(self, dynamic, atol=1e-3, rtol=1e-3):
        """Compile each model class with TensorRT and verify outputs match eager within tolerance.

        The graph is the one `test_torch_export` sweeps, so what this covers is the conversion: which
        subgraphs TensorRT will build engines for, and whether what they compute still matches. The
        tolerance is looser than the traced backends' because an engine is free to reassociate the
        arithmetic it fuses.
        """
        self._skip_if_not_exportable()

        exporter = TensorrtExporter()
        config = TensorrtConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="tensorrt"):
                continue

            components = self._prepare_export_model_and_inputs(model_class, "tensorrt")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, f"Converted outputs are empty for {name}.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_onnx_export(self, dynamic):
        """ExportArtifacts each model class to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="onnx"):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize)

            components = self._prepare_export_model_and_inputs(model_class, "onnx")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    onnx_outputs = output.runtime()(**inputs)
                    self.assertTrue(onnx_outputs, f"ONNX outputs are empty for {name}.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export(self, dynamic):
        """ExportArtifacts each model class to ExecuTorch, run it, and verify output count matches eager."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="executorch"):
                continue

            # Per class: a graph whose delegate refuses its own partitioner's claim lowers undelegated.
            config = ExecutorchConfig(
                dynamic=dynamic,
                partition=_executorch_partition_enabled(model_class, dynamic),
                partition_exclude=_executorch_partition_exclude(model_class, dynamic),
            )

            # Trace on CPU: XNNPACK targets CPU, and CPU tracing yields device-consistent graphs.
            # Tracing on CUDA surfaces per-model device bugs — models create in-`forward` tensors
            # (arange/zeros/sinusoids) without `device=`, which default to CPU and then mismatch a
            # CUDA model (`FakeTensor Device Propagation ... cuda:0, cpu`). The exporter *can* take a
            # CUDA model, but the suite exercises the canonical CPU-traced path.
            components = self._prepare_export_model_and_inputs(model_class, "executorch", device="cpu")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    # Building the runner stays *inside* the tolerance: loading the method is where
                    # ExecuTorch reports a missing kernel or an oversized arena.
                    with _tolerating_executorch_limits(f"{model_class.__name__}/{name}"):
                        outputs = output.runtime()(**inputs)
                        tensors = [t for t in outputs.values() if isinstance(t, torch.Tensor)]
                        self.assertEqual(len(tensors), len(eager_outputs[name]))


class ExportGenerateTesterMixin(ExportTesterMixin):
    """Mixin providing generation-aware export tests for torch.export, ONNX, and ExecuTorch backends.

    Inherits ``ExportTesterMixin`` for the shared exportability gate / skip logic / input prep, and
    is mixed into a model test class alongside ``GenerationTesterMixin``.

    Required attributes on the host class (in addition to those from ``ExportTesterMixin``):
    - ``all_generative_model_classes`` — iterable of generative model class objects to test.
    - ``prepare_config_and_inputs_for_generate()`` — returns ``(config, inputs_dict)`` suitable
      for ``model.generate()``.

    Each generative model is decomposed into prefill and decode components via
    :func:`decompose_prefill_decode`.  Multi-modal models additionally decompose the prefill
    stage into individual submodules via :func:`decompose_multimodal`.
    """

    def _prepare_export_generate_model_and_inputs(
        self, model_class, backend, device=torch_device, generation_config=None, multi_token_decode=False
    ):
        """Decompose a generative model into exportable components.

        For multi-modal models: decomposes the prefill stage into individual submodules plus the decode stage.
        For decoder-only models: returns prefill and decode components.

        ``device`` defaults to ``torch_device``; the ExecuTorch tests pass ``"cpu"`` so the
        ``generate()`` call inside :func:`decompose_for_generation` runs on CPU — a device-side
        assert there (e.g. a VLM ``masked_scatter`` size mismatch) would otherwise poison the
        xdist worker's CUDA context and cascade to every later test on it.

        ``generation_config`` is forwarded to the ``generate()`` capture (default: the model's own).
        Pass one with ``cache_implementation="static"`` to export against a fixed-size ``StaticCache``.

        ``multi_token_decode`` captures the ``decode`` component with a multi-token query axis
        (continuation-from-past, or a plain prefill when the cache is empty) instead of the classic
        single-token step — see :func:`decompose_for_generation`.

        Returns:
            Dict of `{name: Component}` — one entry per component.
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval()
        # Use half precision only when the model has a half-precision-only kernel — the vision varlen flash
        # attention or grouped-mm MoE experts (see `needs_half_precision_export`); everything else stays fp32
        # (realistic, and avoids spurious dtype mismatches). The half type is per-backend: fp16 for ONNX
        # (ORT has no bf16 kernels for many ops), bf16 for torch.export/ExecuTorch (flash + grouped_mm need it).
        half_dtype = torch.float16 if backend == "onnx" else torch.bfloat16
        dtype = half_dtype if needs_half_precision_export(model) else torch.float32
        model = model.to(device, dtype)
        set_model_for_less_flaky_test(model)

        inputs_dict = cast_leaf_tensors(inputs_dict, dtype=module_dtype(model), device=module_device(model))

        return decompose_for_generation(
            model, inputs_dict, generation_config=generation_config, multi_token_decode=multi_token_decode
        )

    # ──────────────────── torch.export tests ─────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    # `atol` is looser than the non-generate check's 1e-4 because a component here can come back near zero:
    # t5gemma2's `encoder_last_hidden_state` sits at ~1e-3 while its intermediates are O(1), so plain fp32
    # accumulation lands up to 1.8e-4 apart — 17% of the value, yet ordinary in absolute terms, and it flips
    # this comparison run to run (measured 6.6%/7.1%/8.1%/10.1%/10.6% of elements over, and not seedable:
    # the variance survives pinning both weights and inputs). A real wiring bug — wrong cache, mask or
    # positions — diverges by orders of magnitude more, and the id-parity check below still guards it.
    def test_torch_export_generate(self, dynamic, multi_token_decode, generation_config, atol=5e-4, rtol=1e-4):
        """ExportArtifacts prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="dynamo",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue
            components = self._prepare_export_generate_model_and_inputs(
                model_class, "dynamo", generation_config=generation_config, multi_token_decode=multi_token_decode
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)
                    exported[name] = output

            # End-to-end id-parity over both cache kinds, whenever the graphs can serve `generate`'s loop:
            # via the dedicated `prefill` graph (always under dynamic shapes, under static ones only with
            # a static cache, whose frozen shapes reproduce every step), or via the multi-token decode
            # serving prefill and decode from one graph.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend="dynamo",
                    multi_token=multi_token_decode,
                    generation_config=generation_config,
                    runtime=True,
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "dynamo", generation_config, dynamic, multi_token_decode
                    )

    # ────────────────────── AOTInductor tests ────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_inductor_toolchain
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_aoti_export_generate(self, dynamic, multi_token_decode, generation_config, atol=5e-4, rtol=1e-4):
        """Compile the components a generation loop drives, and drive them.

        Compiling each component is one question (the graphs a loop needs are not the single forward
        `test_aoti_export` covers -- a decode step over a cache, a modality tower, an embedder); driving
        the compiled set through `generate` is the other, and it is the one that catches a package whose
        compiled-in shapes cannot serve the step the loop actually makes.
        """
        self._skip_if_not_exportable()

        exporter = AotiExporter()
        config = AotiConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="aoti",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue
            components = self._prepare_export_generate_model_and_inputs(
                model_class, "aoti", generation_config=generation_config, multi_token_decode=multi_token_decode
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, "Compiled outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)
                    exported[name] = output

            # Same gate as the traced sweep: the loop runs wherever the exported graphs can serve it.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend="aoti",
                    multi_token=multi_token_decode,
                    generation_config=generation_config,
                    runtime=True,
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "aoti", generation_config, dynamic, multi_token_decode
                    )

    # ─────────────────────── TensorRT tests ──────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_torch_tensorrt
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_tensorrt_export_generate(self, dynamic, multi_token_decode, generation_config, atol=1e-3, rtol=1e-3):
        """Convert the components a generation loop drives, and drive them.

        TensorRT holds an engine to the shape range it was built for, where `torch.export` treats that
        range as advisory — so the variants that ask a graph for a length it never traced (a growing cache
        is empty at the first step) are this backend's ceiling rather than a bug, and the ledger records
        them per variant.
        """
        self._skip_if_not_exportable()

        exporter = TensorrtExporter()
        config = TensorrtConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="tensorrt",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue
            components = self._prepare_export_generate_model_and_inputs(
                model_class, "tensorrt", generation_config=generation_config, multi_token_decode=multi_token_decode
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = output.runtime()(**copy.deepcopy(inputs))
                        self.assertTrue(exported_outputs, "Converted outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)
                    exported[name] = output

            # Same gate as the traced sweep: the loop runs wherever the exported graphs can serve it.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend="tensorrt",
                    multi_token=multi_token_decode,
                    generation_config=generation_config,
                    runtime=True,
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "tensorrt", generation_config, dynamic, multi_token_decode
                    )

    # ──────────────────────── ONNX tests ─────────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_onnx_export_generate(self, dynamic, multi_token_decode, generation_config):
        """ExportArtifacts prefill and decode stages to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="onnx",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize, external_data=False)

            components = self._prepare_export_generate_model_and_inputs(
                model_class, "onnx", generation_config=generation_config, multi_token_decode=multi_token_decode
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    onnx_outputs = output.runtime()(**inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))
                    exported[name] = output

            # End-to-end id-parity (text and VLM) — see the dynamo call site for the gate.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend="onnx",
                    multi_token=multi_token_decode,
                    generation_config=generation_config,
                    runtime=True,
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "onnx", generation_config, dynamic, multi_token_decode
                    )

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_openvino_export(self, dynamic):
        """ExportArtifacts each model class to OpenVINO IR and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="openvino"):
                continue

            exporter = OpenVINOExporter()
            config = OpenVINOConfig(dynamic=dynamic)

            components = self._prepare_export_model_and_inputs(model_class, "openvino")
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    runtime = output.runtime()
                    ov_outputs = runtime(**inputs)
                    _assert_openvino_output_names(self, runtime, ov_outputs, eager_outputs[name])

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_openvino_export_generate(self, dynamic, multi_token_decode, generation_config):
        """ExportArtifacts prefill and decode stages to OpenVINO IR and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="openvino",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue

            exporter = OpenVINOExporter()
            config = OpenVINOConfig(dynamic=dynamic)

            components = self._prepare_export_generate_model_and_inputs(
                model_class, "openvino", generation_config=generation_config, multi_token_decode=multi_token_decode
            )
            eager_outputs = self._collect_eager_outputs(components)

            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    runtime = output.runtime()
                    ov_outputs = runtime(**inputs)
                    _assert_openvino_output_names(self, runtime, ov_outputs, eager_outputs[name])

    # ──────────────────── ExecuTorch tests ───────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export_generate(self, dynamic, multi_token_decode, generation_config):
        """ExportArtifacts prefill and decode stages to ExecuTorch, run each, and verify output count matches eager."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class,
                generate=True,
                dynamic=dynamic,
                backend="executorch",
                multi_token=multi_token_decode,
                generation_config=generation_config,
            ):
                continue

            # Per class: a graph whose delegate refuses its own partitioner's claim lowers undelegated.
            config = ExecutorchConfig(
                dynamic=dynamic,
                partition=_executorch_partition_enabled(model_class, dynamic),
                partition_exclude=_executorch_partition_exclude(model_class, dynamic),
            )

            components = self._prepare_export_generate_model_and_inputs(
                model_class,
                "executorch",
                device="cpu",
                generation_config=generation_config,
                multi_token_decode=multi_token_decode,
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                model, inputs = component.module, component.inputs
                with self.subTest(f"{model_class.__name__}/{name}"):
                    output = exporter.export(model, inputs, config=config)
                    # Building the runner stays *inside* the tolerance: loading the method is where
                    # ExecuTorch reports a missing kernel or an oversized arena.
                    with _tolerating_executorch_limits(f"{model_class.__name__}/{name}"):
                        outputs = output.runtime()(**inputs)
                        tensors = [t for t in outputs.values() if isinstance(t, torch.Tensor)]
                        self.assertEqual(len(tensors), len(eager_outputs[name]))
                        # Only a component that ran is handed to the generate drive below.
                        exported[name] = output

            # End-to-end id-parity (text and VLM). Multi-token decode works on ExecuTorch because
            # `_fix_range_constraints` bounds the otherwise-unbounded sequence dim (XNNPACK can't size a
            # static tensor from an unbounded extent). Runs on CPU (device="cpu" above), matching the
            # runner's device. See the dynamo call site for the gate.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend="executorch",
                    multi_token=multi_token_decode,
                    generation_config=generation_config,
                    runtime=True,
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "executorch", generation_config, dynamic, multi_token_decode
                    )
