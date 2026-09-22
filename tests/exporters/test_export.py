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
import importlib.util
import inspect
import itertools
import re

import pytest
import torch
from parameterized import parameterized

from transformers import GenerationConfig, set_seed
from transformers.exporters.exporter_dynamo import DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import (
    _OFF_GRAPH_CACHE_BACKENDS,
    ExecutorchConfig,
    ExecutorchExporter,
)
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.exporter_openvino import OpenVINOConfig, OpenVINOExporter
from transformers.exporters.utils import (
    cast_leaf_tensors,
    decompose_for_generation,
    decompose_multimodal,
    get_leaf_tensors,
    is_multimodal,
    module_device,
    module_dtype,
)
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch_greater_or_equal,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import is_executorch_available


# ──────────────────────────── skip lists ────────────────────────────
#
# A single mapping ``EXPORT_SKIPS[scope][model_class_name] = reason`` drives every skip.
# ``scope`` is a dotted path that narrows from broad (``"all"`` — every backend, every variant)
# to specific (``"onnx.generate"``, ``"onnx.dynamic"``, ``"openvino"``, …). At test time
# ``_should_skip`` walks the scopes that match the current ``(backend, generate, dynamic)``
# triple and returns ``True`` as soon as the model is found in any of them. Reasons live next
# to the model name so the "why" travels with the entry.
#
# A scope may also carry an ``.exactness`` suffix (``"exactness"``, ``"openvino.exactness"``, …).
# Those entries do not skip the test: the model is still exported and run, only the comparison
# against eager is dropped. Use them when the comparison itself is meaningless — a forward that
# draws its own randomness does not even agree with itself between two eager calls — so that a real
# export break still fails instead of hiding behind a full skip.
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
    # Exported and run as usual, but not compared against eager (see the ``.exactness`` note above).
    "exactness": {
        "VibeVoiceAsrModel": (
            "Its acoustic tokenizer is a VAE that samples inside the forward — `vae_std=0.625` of noise "
            "over latents whose mean magnitude is 1.7e-06 — so two eager runs on the same inputs differ "
            "by 0.037, more than the export gap itself. The exported graph returns the distribution's "
            "mean, since the RNG traces as zeros."
        ),
        "VibeVoiceAsrForConditionalGeneration": "Same VAE sampling as `VibeVoiceAsrModel`.",
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
        "VibeVoiceForConditionalGeneration": (
            "Generation uses two forward calls with different input shapes (prefill + noise scheduler); "
            "`decompose_prefill_decode` can't capture the full generate path reliably, causing flaky "
            "CUDAGraphs / export failures. TODO: handle in a follow-up PR."
        ),
    },
    # Every backend, dynamic-shape only.
    "dynamic": {
        "HieraForPreTraining": (
            "With `bool_masked_pos` set, `HieraEncoder.reroll` reshapes on the mask's unbacked token count, so "
            "dynamic shapes raise `GuardOnDataDependentSymNode` on `416*((u0//416)) < 2`. The other Hiera heads "
            "export fine under dynamic shapes, and every one of them exports under static shapes."
        ),
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
    # Generate path, dynamic-shape only — the multi-token decode (`_merge_decode_calls`) that keeps the
    # query axis symbolic. Backend-agnostic (it's in the shared decomposition). Static generate, which
    # captures a single-token decode with no merge, still runs.
    "generate.dynamic": {
        "MllamaForConditionalGeneration": (
            "Cross-attention decode indexes `cross_attention_mask[:, :, arange(seq) + past_seen_tokens]`; "
            "the multi-token merge grows the query axis but not the captured cross-attention mask, so the "
            "index runs past it (out of bounds → CUDA device-side assert). Single-token static generate is fine. "
            "TODO: grow `cross_attention_mask` in `_merge_decode_calls`."
        ),
        "ReformerModelWithLMHead": (
            "Chunked local attention assumes a chunk-aligned query length; the merged multi-token query "
            "(seq 2) mismatches the chunked key axis (`size 2 vs 6`). Single-token static generate is fine. "
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
    # Exported and run as usual, but not compared against eager (see the ``.exactness`` note above).
    "onnx.exactness": {
        "Wav2Vec2ForPreTraining": (
            "`codevector_perplexity` and `projected_quantized_states` come out of a Gumbel-softmax draw "
            "inside the forward, so the quantizer picks different codes each run and eager does not agree "
            "with itself either."
        ),
        "UniSpeechForPreTraining": "Same Gumbel-softmax quantizer draw as `Wav2Vec2ForPreTraining`.",
        "PatchTSTForPretraining": (
            "The tiny config sets `random_mask_ratio=0.0`, so nothing is masked and the pretraining loss is "
            "`0 / (0 + 1e-10)`: eager reports a clean `0.0` while ORT lands on `NaN`. The reconstruction it "
            "is computed from matches. TODO: revisit if the tiny config ever masks anything."
        ),
        "BitModel": (
            "ONNX Runtime's fp32 accumulation through this conv stack drifts ~0.0018 from torch, over the "
            "1e-3 tolerance but far from structural."
        ),
        "BitBackbone": "Same ORT accumulation as `BitModel` (~0.0048 across the feature maps).",
        "ClapModel": "Same ORT accumulation as `BitModel` (~0.0033 on the contrastive logits).",
        "CLIPSegForImageSegmentation": "Same ORT accumulation as `BitModel` (~0.0059 on the decoder logits).",
        "FlavaForPreTraining": "Same ORT accumulation as `BitModel` (~0.0039 on the contrastive logits).",
        "Tipsv2DptForDensePrediction": "Same ORT accumulation as `BitModel` (~0.0052 across the dense heads).",
        "Tipsv2DptForDepthEstimation": "Same ORT accumulation as `BitModel` (~0.0034).",
        "Tipsv2DptForNormalEstimation": "Same ORT accumulation as `BitModel` (~0.0048).",
        "Tipsv2DptForSemanticSegmentation": "Same ORT accumulation as `BitModel` (~0.0041).",
        "DepthProForDepthEstimation": (
            "`predicted_depth` differs by ~0.036 — the same ORT accumulation as `BitModel`, amplified by the "
            "multi-scale depth head's upsampling and fusion."
        ),
        "OneFormerModel": (
            "`task_token` differs by ~0.029. Everything else matches; the task MLP runs on a constant task "
            "input, so ORT's accumulation shows up undamped there."
        ),
        "OneFormerForUniversalSegmentation": "Same `task_token` divergence as `OneFormerModel`.",
        "TapasForQuestionAnswering": (
            "It selects one column with an `argmax` over `column_logits`, and in the tiny config those are "
            "tied: several rows have two columns at the maximum and one has all 32 (every column reads as "
            "padding, so they all sit at `CLOSE_ENOUGH_TO_LOG_ZERO`). ONNX Runtime breaks the tie "
            "differently from torch, so a different column is selected and the -10000 mask lands on "
            "different cells — the same arbitrary-but-valid choice as the detection models above."
        ),
        "FlaubertForQuestionAnswering": (
            "`end_top_index` is an index output chosen by `topk` over tied scores in the tiny test config, "
            "so ONNX Runtime breaks the tie differently — an equally valid choice, the same way OpenVINO "
            "does. `torch.export` still resolves it exactly as eager, so it stays compared there."
        ),
        "XLMForQuestionAnswering": "Same tied-`end_top_index` selection as `FlaubertForQuestionAnswering`.",
        "DFineModel": (
            "The tiny test config's classification head emits a constant, so the encoder's `topk` over "
            "`enc_outputs_class` picks among *tied* scores and ONNX Runtime breaks the tie differently: "
            "`enc_topk_bboxes` / `encoder_pred_boxes` hold the same boxes in another order."
        ),
        "DFineForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "Deimv2Model": "Same tied-`topk` selection as `DFineModel`.",
        "Deimv2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrModel": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrV2Model": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrV2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "PPDocLayoutV2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "PPDocLayoutV3ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "MMGroundingDinoModel": "Same tied-`topk` box selection as `DFineModel`.",
        "MMGroundingDinoForObjectDetection": "Same tied-`topk` box selection as `DFineModel`.",
        "LwDetrModel": (
            "The encoder's `topk` ranks proposals by scores that sit ~1e-4 apart in relative terms "
            "(~9.39e-07 absolute) in the tiny test config, so ONNX Runtime's slightly different arithmetic "
            "orders two of them the other way round: `enc_outputs_coord_logits` holds the same boxes, "
            "swapped, matching the other proposal's coordinates exactly."
        ),
        "LwDetrForObjectDetection": "Same near-tied `topk` ordering as `LwDetrModel`.",
    },
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
        "EfficientNetModel": (
            "ExecuTorch export exceeds the 1000s test timeout under both static and dynamic shapes "
            "(dynamic ~1400s); the depthwise-conv / SiLU stack lowers slowly."
        ),
        "EfficientNetForImageClassification": "Same `timeout` as `EfficientNetModel`.",
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
        "Swinv2Model": "Same `timeout` failure as `Mask2FormerModel`.",
        "Swinv2ForImageClassification": "Same `timeout` failure as `Mask2FormerModel`.",
        "Swinv2ForMaskedImageModeling": "Same `timeout` failure as `Mask2FormerModel`.",
        "Swinv2Backbone": "Same `timeout` failure as `Mask2FormerModel`.",
        "TimesformerModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "TimesformerForVideoClassification": "Same `timeout` failure as `Mask2FormerModel`.",
    },
    "executorch.static": {
        "Wav2Vec2BertModel": (
            "ExecuTorch *runtime* execution of the Conformer encoder exceeds the 15-min test timeout "
            "under static shapes (~1350s in the runtime, not lowering). Dynamic shapes stay under budget."
        ),
        "Wav2Vec2BertForCTC": "Same Conformer-encoder runtime `timeout` as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForSequenceClassification": "Same Conformer-encoder runtime `timeout` as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForAudioFrameClassification": "Same Conformer-encoder runtime `timeout` as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForXVector": "Same Conformer-encoder runtime `timeout` as `Wav2Vec2BertModel`.",
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
    "openvino.exactness": {
        "FlaubertForQuestionAnswering": (
            "`end_top_index` is an index output chosen by `topk` over tied scores in the tiny test config, so "
            "OpenVINO's tie-break picks different — equally valid — indices."
        ),
        "XLMForQuestionAnswering": "Same tied-`end_top_index` selection as `FlaubertForQuestionAnswering`.",
        "DFineModel": (
            "The tiny test config's classification head emits a constant, so the encoder's `topk` over "
            "`enc_outputs_class` picks 30 of 84 *tied* scores — an arbitrary choice that OpenVINO breaks "
            "differently from torch. The selected boxes are the same set in a different order, and "
            "`enc_topk_logits` matches exactly; a trained checkpoint has no such ties."
        ),
        "DFineForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "Deimv2Model": "Same tied-`topk` selection as `DFineModel`.",
        "Deimv2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrModel": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrV2Model": "Same tied-`topk` selection as `DFineModel`.",
        "RTDetrV2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "PPDocLayoutV2ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "PPDocLayoutV3ForObjectDetection": "Same tied-`topk` selection as `DFineModel`.",
        "MMGroundingDinoModel": "Same tied-`topk` box selection as `DFineModel`.",
        "MMGroundingDinoForObjectDetection": "Same tied-`topk` box selection as `DFineModel`.",
    },
    # OpenVINO, generate path only.
    "openvino.generate": {},
    # OpenVINO, dynamic-shape only.
    "openvino.dynamic": {
        "BigBirdModel": "OpenVINO conversion exceeds the 1000s test timeout under dynamic shapes.",
        "BigBirdForPreTraining": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForCausalLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerModel": "Shifted-window (Swin) backbone exceeds the 1000s test timeout under dynamic shapes.",
        "MaskFormerForInstanceSegmentation": "Same `timeout` as `MaskFormerModel`.",
        "Mask2FormerModel": "Deformable-attention pixel decoder exceeds the 1000s test timeout under dynamic shapes.",
        "Mask2FormerForUniversalSegmentation": "Same `timeout` as `Mask2FormerModel`.",
        "GroundingDinoModel": "Deformable-attention encoder exceeds the 1000s test timeout under dynamic shapes.",
        "GroundingDinoForObjectDetection": "Same `timeout` as `GroundingDinoModel`.",
        "MMGroundingDinoModel": "Same `timeout` as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `timeout` as `GroundingDinoModel`.",
        "Xcodec2Model": (
            "OpenVINO can't convert a rank-0 `aten.slice.Tensor` in the codec's dynamic-shape path "
            "(`input_rank.get_length() > 0` fails in slice shape inference). Static export converts fine."
        ),
        "HieraModel": (
            "OpenVINO export marks every input axis dynamic (`Dim.AUTO`), driving the Hiera mask-unit "
            "backbone's data-dependent unroll into `GuardOnDataDependentSymNode` (`416*(u0//416) < 2`). "
            "Pure `torch.export`/dynamo dynamic export passes (verified), so this is OpenVINO-specific. "
            "Static export works."
        ),
        "HieraBackbone": "Same OpenVINO all-dynamic-axes Hiera-backbone guard as `HieraModel`.",
        "HieraForImageClassification": "Same OpenVINO all-dynamic-axes Hiera-backbone guard as `HieraModel`.",
        "HieraForPreTraining": "Same OpenVINO all-dynamic-axes Hiera-backbone guard as `HieraModel`.",
    },
    # OpenVINO, static-cache generate variants only (a `generation_config` requesting a static cache).
    "openvino.static-cache": {
        "MiniMaxM3SparseForConditionalGeneration": (
            "Exports fine, but OpenVINO inference of the language-model component fails at runtime "
            "(`Eltwise 'add_31' shape mismatch`): the sparse-MoE static cache (`MiniMaxM3VLSparseStaticCacheLayer`, "
            "an `idx_keys` state of shape `[2,1,9,16]`) doesn't broadcast against the decode inputs. Its "
            "non-static-cache generate variants pass."
        ),
    },
}


# ──────────────────────────── ONNX optimization toggles ────────────────────────────
# Not "skips" — these select whether `onnxscript` optimisation runs for a given model.
# Same scope-keyed shape as ``EXPORT_SKIPS`` for symmetry.


ONNX_DISABLE_OPTIMIZE: dict[str, dict[str, str]] = {
    # Disable for every variant.
    "all": {
        "LayoutLMv2Model": (
            "Detectron2 FPN backbone — onnxscript optimizer drops initializers still referenced by nodes, "
            "producing an invalid graph for ORT."
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
        "Wav2Vec2Model": (
            "The optimizer mis-folds the symbolic conv-length chain that sizes the feature-vector "
            "attention mask: the `zeros` it builds comes out `{batch, -2}` and ORT fails the `Expand` with "
            "`right operand cannot broadcast on dim 1`. `optimize=False` exports and matches eager to 2e-4."
        ),
        "Wav2Vec2ForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "WavLMModel": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "WavLMForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "WavLMForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "WavLMForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "HubertModel": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "HubertForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "HubertForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Data2VecAudioModel": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Data2VecAudioForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Data2VecAudioForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Data2VecAudioForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatModel": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatForPreTraining": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerModel": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerForCTC": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerForPreTraining": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerForSequenceClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ConformerForXVector": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ForXVector": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "Wav2Vec2ForPreTraining": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "WavLMForXVector": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "HubertForAudioFrameClassification": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "HubertForXVector": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "UniSpeechSatForXVector": "Same mis-folded conv-length chain as `Wav2Vec2Model`.",
        "ProphetNetModel": (
            "Onnxscript's `SplitToSequence` constant-folding trips `'NoneType' object has no attribute 'ndim'` "
            "under dynamic shapes. Static works after the vectorized `ngram_attention_bias` rewrite."
        ),
        "ProphetNetForConditionalGeneration": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ProphetNetDecoder": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ProphetNetForCausalLM": "Same `SplitToSequence` issue as `ProphetNetModel`.",
        "ZoeDepthForDepthEstimation": "Same `SplitToSequence` issue as `ProphetNetModel`.",
    },
}


# Parameterization for export tests: runs once with dynamic=True and once with dynamic=False.
DYNAMIC_EXPORT_PARAMS = parameterized.expand(
    [(False,), (True,)],
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}",
)

# Generation export tests run the cartesian product of two axes: dynamic vs static shapes, and the
# generation config used for the capture. `generation_config=None` is the model's own config (growing
# `DynamicCache`); `cache_implementation="static"` exports against a `StaticCache`. Every cache runs
# under both shape modes — under dynamic shapes a static cache still keeps a symbolic (resizable) size,
# it just writes at fixed positions. A dynamic-shape export also captures the `decode` stage multi-token
# (a symbolic query axis — continuation-from-past, or a plain prefill on an empty cache) rather than the
# single-token step, since a single-token axis can't stay symbolic.
_EXPORT_SHAPE_MODES = [False, True]  # dynamic=False (static shapes) / dynamic=True
_EXPORT_GENERATION_CONFIGS = [None, GenerationConfig(cache_implementation="static")]

GENERATE_EXPORT_PARAMS = parameterized.expand(
    list(itertools.product(_EXPORT_SHAPE_MODES, _EXPORT_GENERATION_CONFIGS)),
    name_func=lambda f, _, p: (
        f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}"
        + (f"_{p.args[1].cache_implementation}_cache" if p.args[1] is not None else "")
    ),
)


_EXECUTORCH_BACKENDS = ("xnnpack",)
if is_executorch_available() and importlib.util.find_spec("executorch.backends.mlx") is not None:
    try:
        from executorch.runtime import Runtime

        if "MLXBackend" in Runtime.get().backend_registry.registered_backend_names:
            _EXECUTORCH_BACKENDS += ("mlx",)
    except ImportError:
        pass  # The Python backend can be installed without the native runtime.

EXECUTORCH_EXPORT_PARAMS = parameterized.expand(
    list(itertools.product(_EXECUTORCH_BACKENDS, _EXPORT_SHAPE_MODES)),
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[1] else 'static'}_{p.args[0]}",
)
EXECUTORCH_GENERATE_EXPORT_PARAMS = parameterized.expand(
    [
        (backend, dynamic, generation_config, cache_implementation)
        for backend, dynamic, generation_config, cache_implementation in itertools.product(
            _EXECUTORCH_BACKENDS,
            _EXPORT_SHAPE_MODES,
            _EXPORT_GENERATION_CONFIGS,
            (None, "executorch_off_graph_cache"),
        )
        if (cache_implementation is None or backend in _OFF_GRAPH_CACHE_BACKENDS)
        and not (
            cache_implementation is None
            and backend == "mlx"
            and generation_config is not None
            and generation_config.cache_implementation == "static"
        )
    ],
    name_func=lambda f, _, p: (
        f"{f.__name__}_{'dynamic' if p.args[1] else 'static'}"
        + (f"_{p.args[2].cache_implementation}_cache" if p.args[2] is not None else "")
        + f"_{p.args[0]}"
        + ("_off_graph_cache" if p.args[3] is not None else "")
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
    """Run an ONNX program and return outputs as a `{name: torch.Tensor}` dict.

    ONNX Runtime hands back numpy arrays on CPU whatever device eager ran on; they come back as
    torch tensors so callers compare outputs the same way across backends (with `check_device=False`).
    """
    set_seed(1234)
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return {name: torch.as_tensor(value) for name, value in zip(onnx_names, onnx_outputs)}


def _run_openvino_model(ov_model, inputs) -> dict:
    """Compile an OpenVINO model and run it, returning outputs as a `{name: torch.Tensor}` dict.

    Feeds the tensor leaves that survived as input ports (stateful folding removes cache
    inputs), seeds folded state variables from the sample cache leaves so outputs correspond
    to the same inputs eager saw, supplies the identity `beam_idx`, and passes scalar kwargs
    through under their FX placeholder names.
    """
    import numpy as np
    import openvino

    set_seed(1234)
    compiled = openvino.compile_model(ov_model, "AUTO")
    request = compiled.create_infer_request()
    leaves = {path: tensor.cpu() for path, tensor in get_leaf_tensors(inputs).items()}
    batch = next(iter(leaves.values())).shape[0] if leaves else 1

    feed = {}
    for port in compiled.inputs:
        # Passthrough tensors carry both an input and an output name — check every alias.
        for name in port.get_names():
            path = re.sub(r"^input\.", "", name)
            if path in leaves:
                feed[name] = leaves[path]
            elif name == "beam_idx":
                feed[name] = np.arange(batch, dtype=np.int32)
            elif name in inputs:
                feed[name] = np.array(inputs[name])
            else:
                continue
            break

    # Folded state variables read zeros on the first infer — seed them from the sample leaves
    # (cast to the variable's dtype: the exporter may retype state, e.g. i64 lengths to i32).
    # The variable id is ``input.<path>output.<path>``.
    def _state_path(state):
        return state.name[len("input.") : (len(state.name) - len("input.output.")) // 2 + len("input.")]

    for state in request.query_state():
        path = _state_path(state)
        if path in leaves:
            state.state = openvino.Tensor(leaves[path].numpy().astype(state.state.data.dtype, copy=False))

    results = request.infer(feed)
    outputs = {}
    for port in compiled.outputs:
        # Compilation may merge a named output tensor with an intermediate that kept its
        # numeric id — prefer the human-readable alias over ``get_any_name``'s sorted-first.
        names = sorted(port.get_names())
        name = next((n for n in names if not n.isdigit()), names[0])
        outputs[re.sub(r"^output\.", "", name)] = torch.as_tensor(results[port])

    # Folded state tensors are outputs too — read them back so the returned dict covers the
    # same leaves eager returns.
    for state in request.query_state():
        outputs[_state_path(state)] = torch.as_tensor(state.state.data.copy())

    return outputs


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

    # Each slot declares its shape; match it to an eager tensor leaf of that shape so the right tensor
    # lands in the right slot (count alone isn't enough — a wrong-shape tensor crashes conv/copy
    # kernels at execute). Under dynamic shapes the declared shape is an upper bound and won't match a
    # leaf, so fall back to the next unused leaf (leaf order tracks the program's input order). If a
    # slot can't be filled — a derived symint, or no leaf of the right shape — reconstruction isn't
    # possible; return None and rely on the load check rather than run with bogus inputs.
    args = []
    for i in range(method.metadata.num_inputs()):
        try:
            shape = tuple(method.metadata.input_tensor_meta(i).sizes())
        except Exception:  # non-tensor slot
            args.append(next(scalars, None))
        else:
            match = next((t for t in tensors if tuple(t.shape) == shape), tensors[0] if tensors else None)
            if match is not None:
                tensors.remove(match)
            args.append(match)
        if args[-1] is None:
            return None

    try:
        outputs = method.execute(args)
    except (RuntimeError, MemoryError) as e:
        if _is_executorch_runtime_limit(e):
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
# XNNPACK delegate can't resize a static tensor to the runtime shape. The execute-phase codes surface
# from either `execute()` or `set_inputs()` (binding the runtime inputs is part of `Method.execute`).
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
    execute = re.search(r"(?:execute\(\)|set_inputs\(\) for method '\w+') failed with error 0x([0-9a-fA-F]+)", msg)
    return bool(execute and f"0x{execute.group(1)}" in _ET_EXECUTE_LIMIT_CODES)


def _onnx_optimize_enabled(model_class, dynamic: bool) -> bool:
    """Return whether onnxscript optimisation should run for this model under this shape mode.

    Mirrors ``_should_skip``'s scope walk on ``ONNX_DISABLE_OPTIMIZE`` — ``"all"`` always
    applies; ``"dynamic"`` adds the dynamic-only entries.
    """
    name = model_class.__name__
    scopes = ["all"] + (["dynamic"] if dynamic else [])
    return not any(name in ONNX_DISABLE_OPTIMIZE.get(scope, {}) for scope in scopes)


# ──────────────────────────── mixins ────────────────────────────


def _zero_padded_positions(outputs, attention_mask):
    """Zero every output position that `attention_mask` masks out, for outputs shaped like the mask."""
    keep = attention_mask.bool()
    zeroed = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.dim() >= 2 and tuple(value.shape[:2]) == tuple(keep.shape):
            mask = keep.reshape(keep.shape + (1,) * (value.dim() - 2)).to(value.device)
            value = value * mask.to(value.dtype)
        zeroed[key] = value
    return zeroed


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

        with open(inspect.getfile(self.all_model_classes[0]), "r", encoding="utf-8") as f:
            source_code = f.read()
            # TODO: add use_experts_implementation support to remaining MoE models
            if "for expert" in source_code and "use_experts_implementation" not in source_code:
                self.skipTest(reason="Model architecture uses eager MoE implementation which is not torch exportable")

    def _export_scopes(
        self, generate=False, dynamic=False, backend=None, static_cache=False, off_graph_cache=False
    ) -> list[str]:
        """The ``EXPORT_SKIPS`` scopes matching the current test, broad to specific.

        ``"all"`` always applies, ``"generate"`` only for generate tests, ``"dynamic"`` / ``"static"``
        for that shape variant on every backend, ``"generate.dynamic"`` for the multi-token decode path,
        ``"static-cache"`` for a variant whose ``generation_config`` requests a static cache,
        ``"<backend>"`` for that backend, and ``"<backend>.<variant>"`` (including
        ``"<backend>.static-cache"``) for the more-specific intersections. Generation scopes also
        distinguish ``in_graph`` and ``off_graph`` cache variants.
        """
        scopes = ["all"]
        if generate:
            scopes.append("generate")
            if dynamic:
                scopes.append("generate.dynamic")
        scopes.append("dynamic" if dynamic else "static")
        if static_cache:
            scopes.append("static-cache")
        if backend:
            scopes.append(backend)
            if generate:
                scopes.append(f"{backend}.generate")
                scopes.append(f"{backend}.generate.{'off_graph' if off_graph_cache else 'in_graph'}")
            scopes.append(f"{backend}.dynamic" if dynamic else f"{backend}.static")
            if static_cache:
                scopes.append(f"{backend}.static-cache")
        return scopes

    def _should_skip(
        self, model_class, generate=False, dynamic=False, backend=None, generation_config=None, off_graph_cache=False
    ):
        """Return True if this model class should be skipped for export tests.

        Walks the scopes from :meth:`_export_scopes` and returns ``True`` as soon as the model is
        listed in any of them. Also skips static-cache variants on models that can't compile
        fullgraph — they don't support a static cache at all.
        """
        name = model_class.__name__
        static_cache = _needs_static_cache(generation_config)
        if static_cache and not model_class._can_compile_fullgraph:
            return True
        if off_graph_cache and (
            not model_class._supports_attention_backend or not model_class._supports_default_dynamic_cache()
        ):
            return True
        scopes = self._export_scopes(
            generate=generate,
            dynamic=dynamic,
            backend=backend,
            static_cache=static_cache,
            off_graph_cache=off_graph_cache,
        )
        return any(name in EXPORT_SKIPS.get(scope, {}) for scope in scopes)

    def _should_skip_exactness(self, model_class, generate=False, dynamic=False, backend=None, generation_config=None):
        """Return True if this model exports and runs but its outputs can't be compared to eager.

        Same scope walk as :meth:`_should_skip`, against the ``.exactness`` variant of each scope.
        """
        name = model_class.__name__
        static_cache = _needs_static_cache(generation_config)
        scopes = self._export_scopes(generate=generate, dynamic=dynamic, backend=backend, static_cache=static_cache)
        # ``"all"`` narrows to a bare ``"exactness"``, the way ``"generate"`` and ``"dynamic"`` are spelled
        scopes = ["exactness" if scope == "all" else f"{scope}.exactness" for scope in scopes]
        return any(name in EXPORT_SKIPS.get(scope, {}) for scope in scopes)

    def _prepare_export_model_and_inputs(self, model_class, device=torch_device):
        """Create model and forward inputs ready for export.

        ``device`` defaults to ``torch_device``; the ExecuTorch tests pass ``"cpu"`` since that
        backend targets CPU anyway, keeping any pre-trace forward off the GPU (a device-side
        assert during tracing would otherwise poison the whole xdist worker's CUDA context).

        Returns:
            `{name: (model, inputs)}`: one entry per component (a whole model, or decomposed submodels
            for a multimodal model), each paired with its forward inputs.
        """
        if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(device)
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

    def _check_outputs_close(self, actual, expected, atol, rtol, check_device=True, inputs=None):
        """Assert outputs are close, allowing up to 5% element-level mismatch.

        When `inputs` carries an `attention_mask`, the positions it masks out are zeroed on both sides
        first. A fully-masked row has no defined value -- attention over it is a softmax with nothing to
        attend to -- so each runtime fills it differently and no caller reads it; comparing those
        positions measures nothing.
        """
        # a model with per-layer masks passes a `dict` here, and a flex-attention one a `BlockMask`;
        # only a plain 2D `(batch, seq)` tensor maps onto output positions. NaViT-style vision models
        # (siglip2) mark their valid patches with `pixel_attention_mask` instead.
        inputs = inputs or {}
        attention_mask = inputs.get("attention_mask")
        if not torch.is_tensor(attention_mask):
            attention_mask = inputs.get("pixel_attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            actual = _zero_padded_positions(actual, attention_mask)
            expected = _zero_padded_positions(expected, attention_mask)
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

            components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, f"Exported outputs are empty for {name}.")

                    if not self._should_skip_exactness(model_class, dynamic=dynamic):
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
    def test_onnx_export(self, dynamic, atol=1e-3, rtol=1e-3):
        """Export each model class to ONNX, run it, and verify outputs match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="onnx"):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize)

            components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, f"ONNX outputs are empty for {name}.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))
                    if not self._should_skip_exactness(model_class, dynamic=dynamic, backend="onnx"):
                        self._check_outputs_close(
                            onnx_outputs, eager_outputs[name], atol=atol, rtol=rtol, check_device=False, inputs=inputs
                        )

    # ──────────────────── OpenVINO tests ─────────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @disable_hub_kernels
    def test_openvino_export(self, dynamic, atol=1e-3, rtol=1e-3):
        """Export each model class to OpenVINO IR, run it, and verify outputs match eager."""
        self._skip_if_not_exportable()
        exporter = OpenVINOExporter()
        config = OpenVINOConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="openvino"):
                continue

            components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    ov_model = exporter.export(model, inputs, config=config)
                    ov_outputs = _run_openvino_model(ov_model, inputs)
                    self.assertTrue(ov_outputs, f"OpenVINO outputs are empty for {name}.")
                    self.assertEqual(set(ov_outputs.keys()), set(eager_outputs[name].keys()))
                    if not self._should_skip_exactness(model_class, dynamic=dynamic, backend="openvino"):
                        self._check_outputs_close(
                            ov_outputs, eager_outputs[name], atol=atol, rtol=rtol, check_device=False, inputs=inputs
                        )

    # ──────────────────── ExecuTorch tests ───────────────────────

    @EXECUTORCH_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export(self, backend, dynamic):
        """Export each model class to ExecuTorch, run it, and verify output count matches eager."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(backend=backend, dynamic=dynamic)

        for model_class in self.all_model_classes:
            if any(
                self._should_skip(model_class, dynamic=dynamic, backend=scope) for scope in ("executorch", backend)
            ):
                continue
            # Trace on CPU: XNNPACK targets CPU, and CPU tracing yields device-consistent graphs.
            # Tracing on CUDA surfaces per-model device bugs — models create in-`forward` tensors
            # (arange/zeros/sinusoids) without `device=`, which default to CPU and then mismatch a
            # CUDA model (`FakeTensor Device Propagation ... cuda:0, cpu`). The exporter *can* take a
            # CUDA model, but the suite exercises the canonical CPU-traced path.
            components = self._prepare_export_model_and_inputs(model_class, device="cpu")
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

    def _prepare_export_generation_request(self, model_class, device=torch_device, off_graph_cache=False):
        """Prepare the original generation request before the exporter captures components."""
        kwargs = {"batch_size": 1} if off_graph_cache else {}
        config, inputs_dict = self.prepare_config_and_inputs_for_generate(**kwargs)
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)
        if off_graph_cache:
            if "input_ids" in inputs_dict:
                # Test a plain, unpadded token request without auxiliary generation inputs.
                inputs_dict = {
                    "input_ids": inputs_dict["input_ids"],
                    "attention_mask": torch.ones_like(inputs_dict["input_ids"]),
                }
            if config.model_type == "gemma4_text":
                # Exercise shared KV independently of the unsupported MoE lowering.
                config.enable_moe_block = False
        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(device)
        set_model_for_less_flaky_test(model)
        inputs_dict = cast_leaf_tensors(inputs_dict, dtype=module_dtype(model), device=module_device(model))
        return model, inputs_dict

    def _prepare_export_generate_model_and_inputs(
        self, model_class, device=torch_device, generation_config=None, multi_token_decode=False
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
            `{name: (model, inputs)}`: the components mapping (see `_prepare_export_model_and_inputs`).
        """
        model, inputs_dict = self._prepare_export_generation_request(model_class, device=device)
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
    def test_torch_export_generate(self, dynamic, generation_config, atol=1e-4, rtol=1e-4):
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class, generate=True, dynamic=dynamic, backend="dynamo", generation_config=generation_config
            ):
                continue
            components = self._prepare_export_generate_model_and_inputs(
                model_class, generation_config=generation_config, multi_token_decode=dynamic
            )
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    if not self._should_skip_exactness(
                        model_class,
                        generate=True,
                        dynamic=dynamic,
                        backend="dynamo",
                        generation_config=generation_config,
                    ):
                        self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_onnx_export_generate(self, dynamic, generation_config):
        """Export prefill and decode stages to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class, generate=True, dynamic=dynamic, backend="onnx", generation_config=generation_config
            ):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize)

            components = self._prepare_export_generate_model_and_inputs(
                model_class, generation_config=generation_config, multi_token_decode=dynamic
            )
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    set_seed(1234)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── OpenVINO tests ─────────────────────────

    @slow
    @GENERATE_EXPORT_PARAMS
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @disable_hub_kernels
    def test_openvino_export_generate(self, dynamic, generation_config):
        """Export prefill and decode stages to OpenVINO IR and verify output names match eager."""
        self._skip_if_not_exportable()
        exporter = OpenVINOExporter()
        config = OpenVINOConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(
                model_class, generate=True, dynamic=dynamic, backend="openvino", generation_config=generation_config
            ):
                continue

            components = self._prepare_export_generate_model_and_inputs(
                model_class, generation_config=generation_config, multi_token_decode=dynamic
            )
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    ov_model = exporter.export(model, inputs, config=config)
                    ov_outputs = _run_openvino_model(ov_model, inputs)
                    self.assertTrue(ov_outputs, "OpenVINO outputs are empty.")
                    self.assertEqual(set(ov_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @EXECUTORCH_GENERATE_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export_generate(self, backend, dynamic, generation_config, cache_implementation):
        """Export generation components; check runtime outputs or off-graph cache export contracts."""
        off_graph_cache = cache_implementation == "executorch_off_graph_cache"
        self._skip_if_not_exportable()
        if off_graph_cache:
            try:
                from executorch.extension.llm.cache.update_and_attend import update_and_attend  # noqa: F401
                from executorch.extension.llm.export.model_metadata import write_cache_geometry  # noqa: F401
            except ImportError:
                self.skipTest("Requires ExecuTorch's off-graph cache extension")
            from executorch.runtime import Runtime

            from transformers.cache_utils import get_layer_types_and_kwargs
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        exporter = ExecutorchExporter()
        config = ExecutorchConfig(backend=backend, dynamic=dynamic, cache_implementation=cache_implementation)
        tested_off_graph = False

        for model_class in self.all_generative_model_classes:
            if any(
                self._should_skip(
                    model_class,
                    generate=True,
                    dynamic=dynamic,
                    backend=scope,
                    generation_config=generation_config,
                    off_graph_cache=off_graph_cache,
                )
                for scope in ("executorch", backend)
            ):
                continue
            if off_graph_cache:
                model, inputs = self._prepare_export_generation_request(
                    model_class, device="cpu", off_graph_cache=True
                )
                if (
                    model.config.is_encoder_decoder
                    or is_multimodal(model)
                    or getattr(model.config, "attn_logit_softcapping", None) is not None
                    or inputs.get("head_mask") is not None
                ):
                    continue
                layer_types, _ = get_layer_types_and_kwargs(model.config)
                if not layer_types or not set(layer_types) <= {"full_attention", "sliding_attention"}:
                    continue
                tested_off_graph = True
                with self.subTest(model_class.__name__):
                    with torch.no_grad():
                        eager_outputs = get_leaf_tensors(model(**inputs, use_cache=False))
                    original_attention = model.config._attn_implementation
                    original_mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
                    try:
                        artifacts = exporter.export_for_generation(
                            model, inputs, config, generation_config, multi_token_decode=dynamic
                        )
                    finally:
                        self.assertEqual(model.config._attn_implementation, original_attention)
                        self.assertIs(ALL_ATTENTION_FUNCTIONS._global_mapping, original_mapping)
                    self.assertEqual(set(artifacts), {"prefill", "decode"})
                    for artifact in artifacts.values():
                        signature = artifact.exported_program().graph_signature
                        self.assertEqual(set(signature.user_inputs), {"input_ids", "position_ids"})
                        self.assertFalse(signature.buffers_to_mutate)
                        self.assertFalse(signature.user_inputs_to_mutate)
                        self.assertEqual(len(signature.user_outputs), len(eager_outputs))
                        # Serialize and load without executing MLX inference.
                        Runtime.get().load_program(artifact.buffer)
                continue
            components = self._prepare_export_generate_model_and_inputs(
                model_class,
                device="cpu",
                generation_config=generation_config,
                multi_token_decode=dynamic,
            )
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    program = exporter.export(model, inputs, config=config)
                    executorch_outputs = _run_executorch_program(program, inputs)
                    if executorch_outputs is None:  # ExecuTorch runtime limit / inputs not reconstructible
                        continue
                    self.assertEqual(len(executorch_outputs), len(eager_outputs[name]))

        if off_graph_cache and not tested_off_graph:
            self.skipTest("No model class supports the off-graph cache export configuration")
