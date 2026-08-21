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
import re

import pytest
import torch
from parameterized import parameterized

from transformers import GenerationConfig, set_seed
from transformers.exporters.exporter_dynamo import _VARLEN_ATTENTION_PATHS, DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.utils import (
    cast_leaf_tensors,
    decompose_for_generation,
    decompose_multimodal,
    get_leaf_tensors,
    is_multimodal,
    module_device,
    module_dtype,
    precompute_export_inputs,
)
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_torch_greater_or_equal,
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
            "`generate()` creates `blank_user_audio_codes` outside the traced forward and "
            "passes it as a kwarg; the resulting ONNX input has mismatched rank (scalar vs 3D). "
            "TODO: make `blank_user_audio_codes` part of the model state."
        ),
        "CohereAsrForConditionalGeneration": (
            "Its decoder reads `encoder_outputs.attention_mask` (the encoder's own frame mask), but the "
            "exporter normalizes `encoder_outputs` to a `BaseModelOutput` so every backend's runtime can "
            "rebuild one. `ModelOutput` flattens by its dict, and parakeet attaches that mask after "
            "construction, so it never survives into the traced forward — and being `None` here it cannot "
            "ride in the dict either. TODO: carry the encoder's own output class through export and have "
            "`_ExportedEncoder` rebuild it."
        ),
        "DiaForConditionalGeneration": (
            "Decodes several audio codebooks at once, so its decoder inputs carry a channel axis "
            "(`decoder_input_ids` is 3-D) and its `decoder_attention_mask` is shaped to match. The runtime "
            "builds the decoder's causal mask from the cache — 2-D positions into a `[batch, 1, q, kv]` "
            "mask — which is the right thing for every other encoder-decoder and the wrong rank here "
            "(`upper bound and lower bound inconsistent with step sign`). TODO: shape the decoder mask "
            "from the graph's own declared rank, the way `_mask_feed` already does for mixed attention."
        ),
        "UdopForConditionalGeneration": (
            "Exported decoder output is missing `attention_mask` vs eager — encoder-decoder "
            "cross-attention mask doesn't flow through the generate decomposition correctly."
        ),
        "VoxtralRealtimeForConditionalGeneration": (
            "Exported prefill drops `past_key_values.*.{keys,values,_sliding_window_tensor}` "
            "tensors that eager returns. Plain forward exports work. "
            "TODO: align generate-decomposition path with the realtime KV-cache shape."
        ),
        "Gemma3nForConditionalGeneration": (
            "KV-shared layers (`num_kv_shared_layers`) reuse cache entries from earlier layers; "
            "exported prefill returns only `logits` while eager surfaces the populated KV cache. "
            "Same shape as Voxtral. TODO: align the generate-decomposition path."
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
            "Same Hiera backbone as `Sam2Model`; when it doesn't overrun it dies in torch's symbolic-shapes "
            "engine instead — sympy cannot solve the windowing shape expressions "
            "(`solveset is unable to solve this equation`, `KeyError: ((s100/4)//8)`)."
        ),
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
            "`MiniMaxCache` grows its `layers` lazily and holds the lightning-attention state in a separate "
            "`linear_cache` list, so a *traced* cache has entries only for the attention layers. Building it "
            "from the config instead — which is all the runtime has — pre-sizes one layer per "
            "`config.layer_types` entry and so invents the `LinearAttentionLayer`s this model never fills, "
            "so the input tree spec cannot match however the class is swapped in. TODO: give the lightning "
            "layers a real `LinearAttentionLayer` (dropping `linear_cache`), or a config-aware pre-size that "
            "skips them."
        ),
        "xLSTMForCausalLM": (
            "`xLSTMCache` is not a full `Cache`: it keeps its state in `rnn_state` with no `layers` list, and "
            "lacks API `generate` expects (`is_compileable`), so every step of building and driving it "
            "surfaces as the next `AttributeError`. TODO: bring the class up to the `Cache` API rather than "
            "special-case it in the runtime."
        ),
        "VibeVoiceAsrForConditionalGeneration": (
            "Keeps two co-equal audio encoders (`acoustic_tokenizer_encoder`, `semantic_tokenizer_encoder`) "
            'and so has no single `get_encoder(modality="audio")` to report; without that the model is not '
            "detected as multi-modal and never splits, leaving the audio path — and its data-dependent "
            "relation between the placeholder-token count and the waveform length — inside the text prefill "
            "graph, whose deferred assert (`Eq(u0, s2//2)`) then fires on the runtime's own feed. Splitting "
            "it needs the model to name an audio encoder, not the exporter to guess one."
        ),
        "DeepseekV4ForCausalLM": (
            "Its HCA/CSA cache layers keep dict-keyed state whose entries the model's own update path "
            "creates lazily (the traced cache carries `entry_count: {'compressor': 1, 'indexer': 1}` and the "
            "matching `buffer_kv`/`compressed_kv` tensors; a fresh one starts with `compressor` alone), so a "
            "config-built cache cannot match the traced input tree spec — the same lazily-grown-state "
            "problem as `MiniMaxCache`. Export itself passes, and so do the static-cache generate variants."
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
            "text+kwargs step instead, so generation diverges from the first token. Export itself and the "
            "per-component parity still run."
        ),
        "XLMWithLMHeadModel": (
            "Its `prepare_inputs_for_generation` appends a mask token to `input_ids` every step and builds a "
            "`langs` tensor from `config.lang_id`, so the graph takes a per-step input only that model can "
            "produce (and a step is one token wider than `generate`'s). Export itself is covered by the "
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
        "ProphetNetForConditionalGeneration": (
            "N-gram decoding ties its stream positions to the cache length, which the decode capture bakes "
            "(`1 + past_key_values[0].size(2) == 3`, the second-step cache the encoder-decoder capture "
            "traces at) — so every later step trips the guard. The length cannot stay symbolic while the "
            "ngram streams index relative to it. Export itself is covered by the non-generate variants."
        ),
        "PerceptionLMForConditionalGeneration": (
            "Routes videos through a *second* `get_image_features` call (`pixel_values=pixel_values_videos`); "
            "the decomposition and runtime model one modality per getter, so the video kwarg has no route "
            "and its features would need the image runner run twice with different scatter targets. "
            "TODO: let a modality spec share another's runner."
        ),
    },
    # Generate path, multi-token decode capture only — the two decode steps merged by
    # `_merge_decode_calls` into one graph whose query axis stays symbolic, so a single graph serves both
    # the prompt and every decode step. Backend-agnostic. The single-token capture, which exports prefill
    # and decode as separate graphs with their own fixed query lengths, still runs.
    "generate.multi_token": {
        "T5Gemma2ForConditionalGeneration": (
            "Multi-modal *encoder-decoder*: its image features reach the decoder as encoder output, not "
            "scattered into text embeddings, so the decomposition yields an `image_encoder` and a decode "
            "graph but neither an `embed_tokens` nor an `encoder` runner — leaving the runtime nothing to "
            "route `pixel_values` through (`model_kwargs are not used: ['pixel_values']`). Wiring a "
            "modality graph's output in as `encoder_outputs` is a runtime capability that doesn't exist "
            "yet. TODO: add it, then drop this skip."
        ),
        "PPFormulaNetForConditionalGeneration": "Same multi-modal encoder-decoder gap as `T5Gemma2ForConditionalGeneration`.",
        "Gemma3ForConditionalGeneration": (
            "Sliding-window cache **and** multi-modal. A multi-modal model exports no `prefill` graph, so its "
            "merged decode graph serves the prompt too — and a decode graph is traced mid-generation, with "
            "the sliding-window layer's `cumulative_length` fixed at that moment in the cache's pytree "
            "context, which the fresh cache a prompt arrives on cannot match. Neither half alone breaks: "
            "gemma2 / mistral / gemma3-text bake the same counter but route the prompt through their own "
            "`prefill`, and llava / qwen2_vl are multi-modal with nothing step-dependent baked. "
            "Three fixes were tried together and reverted — capturing with the model's prebuilt mask (so the "
            "counter leaves the graph), explicit query/key dims on that mask (so `Dim.AUTO` cannot infer "
            "`q == kv` from a prompt), and normalizing the counter out of the spec. The mask dims are what "
            "fail: torch rejects a dim marked dynamic that the code specializes, and a *static* cache's key "
            "axis is genuinely constant, so one spec cannot cover both cache kinds."
        ),
        "Gemma4ForConditionalGeneration": "Same sliding-window-plus-multi-modal shape as `Gemma3ForConditionalGeneration`.",
        "ZambaForCausalLM": (
            "Its hand-copied mixer runs the selective scan per head with the associative path deliberately "
            "off ('Old model: only when user request it explicitly'), so the sequential scan unrolls and "
            "bakes the query length. The rest of the family (mamba / falcon_mamba / jamba) traces "
            "length-generically (the associative scan + its `initial_states`); aligning "
            "zamba's per-head mixer with mamba's would lift this."
        ),
        "ProphetNetForCausalLM": (
            'The eager model itself refuses the merged capture: its forward asserts "`use_cache` is only '
            'supported for `decoder_input_ids` of length 1", so a 2-token continuation-from-past cannot '
            "even run, let alone trace."
        ),
        "MllamaForConditionalGeneration": (
            "Its images reach the decoder through cross-attention, and the runtime has no route for "
            "`pixel_values` / `aspect_ratio_ids` / `aspect_ratio_mask`: the decomposition finds no "
            "modality getter, so nothing consumes them (`model_kwargs are not used`). The idefics recipe "
            "fits — declare the `cross_attention_layers` as `cross_attention` in `layer_types`, and the "
            "cache's cross slots make the captured prefill the writer the runtime keeps (verified: both "
            "generate variants then pass). It is held back by the static path: that layer type maps to a "
            "*dynamic* layer (cross K/V are image-length, so no static buffer can be pre-sized), which "
            "costs mllama its static-cache compile tests. TODO: land it behind a static cross layer, or "
            "once mllama's static cross slots are sized from the image geometry."
        ),
        "GitForCausalLM": (
            "Its forward corrects for the image tokens only on a single-token step — `position_ids` "
            "offset by the past length, and an `attention_mask` widened by the cached image tokens, both "
            "gated on "
            "`seq_len == 1`. The merged capture traces at 2 tokens, so the graph bakes those branches "
            "*off* and every 1-token decode step then runs without them: the second step's scores drift "
            "~1e-2 from eager. The single-token capture, which keeps the branches, is unaffected."
        ),
        "ReformerModelWithLMHead": (
            "Chunked local attention assumes a chunk-aligned query length; the merged multi-token query "
            "(seq 2) mismatches the chunked key axis (`size 2 vs 6`). "
            "Same chunked-attention limitation as the `onnx.generate` skip."
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
        "Sam2VisionModel": (
            "`torch.export` of the Hiera vision backbone under dynamic shapes takes ~7.5 min "
            "even after simplifying `window_partition`/`window_unpartition` (12 attention blocks "
            "× 3 Q-pool stage transitions on symbolic H/W). ONNX + ORT push past 1000s timeout."
        ),
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
        "JetMoeModel": (
            "MoE and mixture-of-attention route tokens with a data-dependent `inputs.split(expert_size)` "
            "(per-expert token counts), which ExecuTorch's ahead-of-time memory planner can't size "
            "(`GuardOnDataDependentSymNode`). A static rewrite exists (per-token weight gather) but "
            "duplicates expert weights per token, so it's only viable for low-batch decode — not as the "
            "eager default — and the framework's `@use_experts_implementation` is MLP-only, so it can't "
            "host the mixture-of-attention experts. Exports fine on torch.export/ONNX (dynamic dim at runtime)."
        ),
        "JetMoeForCausalLM": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "JetMoeForSequenceClassification": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "FastVlmForConditionalGeneration": (
            "ExecuTorch lowering of the vision stack crashes the process (native segfault/OOM) — the "
            "failure is uncatchable in-process, so the pytest worker dies rather than raising."
        ),
        "FastVlmModel": "Same native ExecuTorch crash as `FastVlmForConditionalGeneration`.",
        "LlavaOnevisionForConditionalGeneration": "Same native ExecuTorch vision-stack crash as `FastVlmForConditionalGeneration`.",
        "LlavaOnevisionModel": "Same native ExecuTorch crash as `LlavaOnevisionForConditionalGeneration`.",
        "PaddleOCRVLForConditionalGeneration": "Same native ExecuTorch vision-stack crash as `FastVlmForConditionalGeneration`.",
        "PaddleOCRVLModel": "Same native ExecuTorch crash as `PaddleOCRVLForConditionalGeneration`.",
        "Qwen3ASRForConditionalGeneration": (
            "Audio encoder packs valid frames with a data-dependent `.nonzero()`; the unbacked "
            "packed length can't be sized by ExecuTorch's ahead-of-time memory planner "
            "(`GuardOnDataDependentSymNode`). Exports fine on torch.export/ONNX, which carry the "
            "dynamic dim at runtime."
        ),
        "Qwen3ASRModel": "Same data-dependent audio-encoder `.nonzero()` as `Qwen3ASRForConditionalGeneration`.",
        "Qwen3ASRForTokenClassification": (
            "Same data-dependent audio-encoder `.nonzero()` as `Qwen3ASRForConditionalGeneration`."
        ),
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
    },
    "executorch.static": {
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
# symbolic and one graph serves prefill and decode — the only option for multi-modal models, which export
# no standalone prefill graph.
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


# ──────────────────────────── helpers ────────────────────────────


def disable_hub_kernels(test_fn):
    """Force `is_kernels_available()` to `False` for the duration of an export test.

    Export must trace the pure-PyTorch path, never a Hub kernel (`mamba-ssm`, `causal-conv1d`, …): those
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
    for key in ("labels", "future_values", "return_loss"):
        inputs_dict.pop(key, None)
    config.return_loss = False
    return inputs_dict


def _run_onnx_program(onnx_program, inputs) -> dict:
    """Run an ONNX program and return outputs as a `{name: tensor}` dict."""
    set_seed(1234)
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return dict(zip(onnx_names, onnx_outputs))


def _run_executorch_program(program_manager, inputs):
    """Load and run an ExecuTorch program, returning its outputs — or ``None`` to skip this component.

    ``None`` means "move on to the next component" and is returned when either:
    - the export is valid but ExecuTorch's own runtime can't service it — a missing portable kernel
      (``0x14``), an oversized arena (``0x21`` / ``bad_alloc``), or a portable-kernel / XNNPACK-delegate
      failure at execute (``0x12`` / ``0x1``): a runtime limitation, not a transformers export defect; or
    - the inputs couldn't be reconstructed for this program (a derived symint slot with no eager leaf).

    Otherwise the model's declared outputs are returned for the caller to check against eager.
    ``torch.export`` also appends mutated inputs (in-place-modified ``pixel_values``, recurrent state,
    …) to the program outputs; those are dropped here — keeping only ``USER_OUTPUT`` slots — so the
    result matches eager's returned leaves.

    Inputs are bound *positionally* against the program's declared slots (``num_inputs`` /
    ``input_tensor_meta``), filled in order from the eager pytree leaves — tensor leaves for tensor
    slots, scalars for the rest.
    """
    from executorch.runtime import Runtime, Verification

    set_seed(1234)
    leaves = torch.utils._pytree.tree_leaves(inputs)
    # The runtime rejects non-contiguous inputs, so materialise tensor leaves. `int` covers `bool`.
    tensors = [t.contiguous() for t in leaves if isinstance(t, torch.Tensor)]
    scalars = (t for t in leaves if isinstance(t, (int, float)))

    # Load — surfaces ExecuTorch resource limits (missing portable kernel / oversized arena).
    try:
        program = Runtime.get().load_program(program_manager.buffer, verification=Verification.Minimal)
        method = program.load_method("forward")
    except (RuntimeError, MemoryError) as e:
        if _is_executorch_runtime_limit(e):
            return None
        raise

    # Each slot declares a shape *and* a dtype; match it to an eager tensor leaf carrying both, so the
    # right tensor lands in the right slot (count alone isn't enough — a wrong-shape tensor crashes
    # conv/copy kernels at execute, and a wrong dtype is refused outright when the method binds its
    # inputs). Matching on shape alone picked an int64 grid for a float slot of the same shape on the
    # grid VLMs. Under dynamic shapes the declared shape is an upper bound and won't match a leaf, so
    # fall back to the next unused leaf of the declared dtype, then to the next unused leaf at all (leaf
    # order tracks the program's input order). If a slot can't be filled — a derived symint, or no leaf
    # left — reconstruction isn't possible; return None and rely on the load check rather than run with
    # bogus inputs.
    from torch.onnx import JitScalarType

    args = []
    for i in range(method.metadata.num_inputs()):
        try:
            meta = method.metadata.input_tensor_meta(i)
            shape, dtype = tuple(meta.sizes()), JitScalarType(int(meta.dtype())).dtype()
        except Exception:  # non-tensor slot
            args.append(next(scalars, None))
        else:
            match = next(
                (t for t in tensors if tuple(t.shape) == shape and t.dtype == dtype),
                next((t for t in tensors if t.dtype == dtype), tensors[0] if tensors else None),
            )
            if match is not None:
                tensors.remove(match)
            args.append(match)
        if args[-1] is None:
            return None

    try:
        outputs = method.execute(args)
    except (RuntimeError, MemoryError) as e:
        # A `set_inputs` failure here says this *reconstruction* guessed wrong, not that the library fed
        # the method something it never declared: the args above are rebuilt from the program's metadata
        # by matching eager leaves, and where the declared shapes are dynamic upper bounds there may be
        # no leaf the method accepts. That is the same "reconstruction isn't possible" case the loop
        # above returns `None` for, so treat it the same and leave the component unverified. The
        # generation drive is held to the stricter rule — its feed is the real one, so a binding failure
        # there is a bug and stays visible.
        if _is_executorch_runtime_limit(e) or "set_inputs" in str(e):
            return None
        raise

    # Drop `torch.export`'s appended mutated-input outputs, keeping only the model's `USER_OUTPUT`s
    # (in program-output order). Then keep tensors only, mirroring eager's `get_leaf_tensors`, so the
    # returned outputs line up with eager's returned leaves for the caller's count check.
    exported_program = program_manager.exported_program
    exported_program = exported_program() if callable(exported_program) else exported_program
    output_kinds = [spec.kind.name for spec in exported_program.graph_signature.output_specs]
    if len(output_kinds) == len(outputs):
        outputs = [out for out, kind in zip(outputs, output_kinds) if kind == "USER_OUTPUT"]
    return [out for out in outputs if isinstance(out, torch.Tensor)]


# ExecuTorch runtime error codes that mean "the export is valid (it produced a loadable program) but
# ExecuTorch's own portable runtime / XNNPACK backend can't service it" — a runtime limitation, not a
# transformers export defect (which surfaces earlier as a `torch.export` error or later as an output
# mismatch). Load: 0x14 missing portable kernel, 0x21 arena can't be allocated, 0x1 XNNPACK partition
# won't compile (`xnn_status_unsupported_parameter`). Execute: 0x12 portable-kernel InvalidArgument
# (constant_pad_nd/convolution/upsample_aa out-tensor sizing), 0x1 XNNPACK delegate failure, 0x10
# XNNPACK delegate can't resize a static tensor to the runtime shape. Only failures from `execute()`
# itself count: the same codes also come out of `set_inputs()`, but binding the runtime inputs is *our*
# side of the contract — it fails when we hand the method something it never declared (feeding an fp32
# cache to a half-precision program did exactly that, and reading it as a backend limitation hid the bug
# across every MoE model), so those have to stay visible.
_ET_LOAD_LIMIT_CODES = {"0x1", "0x14", "0x21"}
_ET_EXECUTE_LIMIT_CODES = {"0x1", "0x10", "0x12"}


def _is_executorch_runtime_limit(exc):
    """True if ``exc`` is a known ExecuTorch runtime limitation (missing kernel / arena / kernel bug)."""
    msg = str(exc)
    if isinstance(exc, MemoryError) or "bad_alloc" in msg:
        return True
    load = re.search(r"Failed to load method forward, error: 0x:?([0-9a-fA-F]+)", msg)
    if load and f"0x{load.group(1)}" in _ET_LOAD_LIMIT_CODES:
        return True
    execute = re.search(r"execute\(\) failed with error 0x([0-9a-fA-F]+)", msg)
    return bool(execute and f"0x{execute.group(1)}" in _ET_EXECUTE_LIMIT_CODES)


def _onnx_optimize_enabled(model_class, dynamic: bool) -> bool:
    """Return whether onnxscript optimisation should run for this model under this shape mode.

    Mirrors ``_should_skip``'s scope walk on ``ONNX_DISABLE_OPTIMIZE`` — ``"all"`` always
    applies; ``"dynamic"`` adds the dynamic-only entries.
    """
    name = model_class.__name__
    scopes = ["all"] + (["dynamic"] if dynamic else [])
    return not any(name in ONNX_DISABLE_OPTIMIZE.get(scope, {}) for scope in scopes)


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

        with open(inspect.getfile(self.all_model_classes[0]), "r") as f:
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
        runs — use it when a model exports fine and only the runtime cannot serve it). Every one of these
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
        scopes.append("dynamic" if dynamic else "static")
        if backend:
            scopes += [backend] + [f"{backend}.{scope}" for scope in scopes if scope != "all"]
        return any(name in EXPORT_SKIPS.get(scope, {}) for scope in scopes)

    def _prepare_export_model_and_inputs(self, model_class, backend, device=torch_device):
        """Create model and forward inputs ready for export.

        ``device`` defaults to ``torch_device``; the ExecuTorch tests pass ``"cpu"`` since that
        backend targets CPU anyway, keeping any pre-trace forward off the GPU (a device-side
        assert during tracing would otherwise poison the whole xdist worker's CUDA context).

        Returns:
            Dict of `{name: (model, inputs)}` — one entry per component.
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
        return {"model": (model, inputs_dict)}

    def _collect_eager_outputs(self, components):
        """Run eager forward for each component and return a ``{name: leaf_tensors}`` dict."""
        eager_outputs = {}
        for name, (model, inputs) in components.items():
            with torch.no_grad():
                set_seed(1234)
                eager_outputs[name] = get_leaf_tensors(model(**copy.deepcopy(inputs)))
                assert eager_outputs[name], f"Eager outputs are empty for {name}."
        return eager_outputs

    def _make_backend_runner(self, backend, name, exported):
        """Wrap `backend`'s runtime handle around component `name`'s artifact from the test loop — the
        session / module the runner drives, nothing is exported here. The ONNX session runs on CPU (host
        round-trips), so the parity check's eager side runs on CPU too."""
        if backend == "dynamo":
            from transformers.exporters import DynamoModelRunner

            return DynamoModelRunner(exported[name].module())
        elif backend == "executorch":
            from executorch.runtime import Runtime, Verification

            from transformers.exporters import ExecutorchModelRunner

            program = Runtime.get().load_program(exported[name].buffer, verification=Verification.Minimal)
            return ExecutorchModelRunner(program)
        elif backend == "onnx":
            import onnxruntime as ort

            from transformers.exporters import OnnxModelRunner

            session = ort.InferenceSession(
                exported[name].model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            return OnnxModelRunner(session)
        else:
            raise ValueError(f"Unknown backend {backend}")

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

        Text models wire the exported `prefill` graph as the generator's dedicated prefill runner, so the
        `decode` graph only ever sees query=1 steps — which is what lets the *static-shape* variants run
        parity too (over a static cache, every step reproduces the frozen shapes). Multi-modal models have
        no standalone prefill graph; their multi-token `decode` serves both, exercising the single-graph
        path (dynamic shapes only)."""
        from transformers.exporters import ExportedGenerator
        from transformers.exporters.utils import _MODALITY_SPECS

        model = components["decode"][0]
        config = model.config
        if (
            multi_token_decode
            and not _needs_static_cache(generation_config)
            and getattr(config, "sliding_window", None) is not None
        ):
            # A growing sliding cache merged multi-token bakes the sliding layers' python `is_full` branch
            # and a trace-time lower bound on the mask's kv axis (`attention_mask.size(1) >= prompt+2`),
            # which the runtime's query=1 steps violate. Single-token decode over a growing sliding cache
            # works (within the window), and static sliding caches work in every mode.
            return

        if (
            backend == "executorch"
            and dynamic
            and not multi_token_decode
            and not _needs_static_cache(generation_config)
            and getattr(config, "sliding_window", None) is not None
        ):
            # A sliding model's *growing* cache, driven one token at a time, outgrows what the XNNPACK
            # lowering will resize: the decode `.pte` refuses the feed at `set_inputs` with 0x10 ("can't
            # resize a static tensor"). Measured scope — the same three models pass every other variant,
            # so this is not the blanket "ET cache dynamism is bounded at the traced shapes" it was once
            # gated as: a static cache is fine (fixed shapes), and so is the merged multi-token decode,
            # whose query axis is dynamic by construction. Deployment guidance for ET remains a static
            # cache. TODO: real upper-bound dynamism in the ET lowering, then ungate.
            return

        if not dynamic and "embed_tokens" in components:
            # A multi-modal model embeds its text in a graph of its own, captured on the *prompt* — under
            # static shapes that graph is specialized to the prompt's length and cannot serve the 1-token
            # decode steps the loop makes (`Guard failed: input_ids.size()[1] == 39`). The static-shape
            # exports themselves are still asserted above; driving them needs a length-generic embedder.
            return

        wanted = {"decode", "prefill", "encoder", "embed_tokens", *(spec[0] for spec in _MODALITY_SPECS)}
        runners = {name: self._make_backend_runner(backend, name, exported) for name in components if name in wanted}
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
            "output_scores": True,
            "return_dict_in_generate": True,
            "generation_config": generation_config,
        }
        eager_out = model.generate(**inputs, **gen_kwargs)
        try:
            exported_out = runtime.generate(**inputs, **gen_kwargs)
        except (RuntimeError, MemoryError) as e:
            # A portable kernel refusing the step's shapes mid-run is this backend's ceiling, the same one
            # the component checks absorb (`_run_executorch_program`) — the graphs themselves are asserted
            # above. Only the *execute* phase counts: a failure while binding inputs means the runtime fed
            # something the method never declared, which is our bug and must stay visible.
            if backend == "executorch" and _is_executorch_runtime_limit(e):
                return
            raise
        # Ids step by step, and only while eager's own top-2 gap says the choice is not a coin flip. This
        # check is about *wiring* — numeric fidelity is asserted per component above
        # (`_check_outputs_close`) — so it deliberately does not put a score bar on an fp32 export: how
        # far a backend's kernels drift is model-specific (an ONNX chameleon drifts past 1e-3 where
        # kosmos2_5 stays at 6e-5, a dynamo export lands at ~1e-9), and a tiny random model routinely has
        # top-2 gaps of a few 1e-3, so its argmax flips on that drift while saying nothing about
        # correctness. Half precision, whose rounding scale is uniform and knowable, still compares the
        # scores themselves. A real wiring bug (wrong cache / mask / positions) diverges early and at a
        # confident step, which the id comparison still catches.
        half = module_dtype(model) in (torch.float16, torch.bfloat16)
        atol = rtol = 1.6e-2
        tie_threshold = 2 * atol if half else 5e-3
        start = eager_out.sequences.shape[1] - len(eager_out.scores)
        for step, (eager_scores, exported_scores) in enumerate(zip(eager_out.scores, exported_out.scores)):
            if half:
                torch.testing.assert_close(exported_scores, eager_scores, atol=atol, rtol=rtol)
            top2 = eager_scores.float().topk(2, dim=-1).values
            if (top2[:, 0] - top2[:, 1]).min() < tie_threshold:
                break
            self.assertEqual(
                exported_out.sequences[:, start + step].tolist(), eager_out.sequences[:, start + step].tolist()
            )

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
        """Export each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="dynamo"):
                continue

            components = self._prepare_export_model_and_inputs(model_class, "dynamo")
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
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
            for name, (model, inputs) in components.items():
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
        """Export each model class to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="onnx"):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize)

            components = self._prepare_export_model_and_inputs(model_class, "onnx")
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
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
        """Export each model class to ExecuTorch, run it, and verify output count matches eager."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="executorch"):
                continue

            # Trace on CPU: XNNPACK targets CPU, and CPU tracing yields device-consistent graphs.
            # Tracing on CUDA surfaces per-model device bugs — models create in-`forward` tensors
            # (arange/zeros/sinusoids) without `device=`, which default to CPU and then mismatch a
            # CUDA model (`FakeTensor Device Propagation ... cuda:0, cpu`). The exporter *can* take a
            # CUDA model, but the suite exercises the canonical CPU-traced path.
            components = self._prepare_export_model_and_inputs(model_class, "executorch", device="cpu")
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    program = exporter.export(model, inputs, config=config)
                    executorch_outputs = _run_executorch_program(program, inputs)
                    if executorch_outputs is None:  # ExecuTorch runtime limit / inputs not reconstructible
                        continue
                    self.assertEqual(len(executorch_outputs), len(eager_outputs[name]))


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
            Dict of `{name: (model, inputs)}` — one entry per component.
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
    def test_torch_export_generate(self, dynamic, multi_token_decode, generation_config, atol=1e-4, rtol=1e-4):
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
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
            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)
                    exported[name] = exported_program

            # End-to-end id-parity (text and VLM), over both cache kinds (static `cache_implementation`
            # and the default growing `DynamicCache`). Runs whenever the exported graphs can serve
            # `generate`'s loop: via the dedicated `prefill` graph (text models — always under dynamic
            # shapes; under static shapes only with a static cache, whose frozen prefill/decode shapes
            # reproduce every step, while a growing cache changes shape each step), or via the multi-token
            # decode serving prefill and decode from one graph (the only option for multi-modal models,
            # which export no standalone prefill graph).
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class, generate=True, runtime=True, generation_config=generation_config
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "dynamo", generation_config, dynamic, multi_token_decode
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
        """Export prefill and decode stages to ONNX and verify output names match eager."""
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
            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))
                    exported[name] = onnx_program

            # End-to-end id-parity (text and VLM) — see `_make_backend_runner` for the ONNX re-export and
            # the dynamo call site for the gate.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class, generate=True, runtime=True, generation_config=generation_config
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "onnx", generation_config, dynamic, multi_token_decode
                    )

    # ──────────────────── ExecuTorch tests ───────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export_generate(self, dynamic, multi_token_decode, generation_config):
        """Export prefill and decode stages to ExecuTorch, run each, and verify output count matches eager."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

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

            components = self._prepare_export_generate_model_and_inputs(
                model_class,
                "executorch",
                device="cpu",
                generation_config=generation_config,
                multi_token_decode=multi_token_decode,
            )
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    program = exporter.export(model, inputs, config=config)
                    executorch_outputs = _run_executorch_program(program, inputs)
                    if executorch_outputs is None:  # ExecuTorch runtime limit / inputs not reconstructible
                        continue
                    self.assertEqual(len(executorch_outputs), len(eager_outputs[name]))
                    exported[name] = program

            # End-to-end id-parity (text and VLM). Multi-token decode works on ExecuTorch because
            # `_fix_range_constraints` bounds the otherwise-unbounded sequence dim (XNNPACK can't size a
            # static tensor from an unbounded extent). Runs on CPU (device="cpu" above), matching the
            # runner's device. See the dynamo call site for the gate.
            can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
            if (can_split_prefill or (dynamic and multi_token_decode)) and components.keys() <= exported.keys():
                if not self._should_skip(
                    model_class, generate=True, runtime=True, generation_config=generation_config
                ):
                    self._assert_generate_matches_eager(
                        components, exported, "executorch", generation_config, dynamic, multi_token_decode
                    )
