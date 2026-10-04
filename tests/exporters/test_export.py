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
import warnings
from contextlib import contextmanager

import pytest
import torch
from parameterized import parameterized

from transformers import GenerationConfig, set_seed
from transformers.exporters.components import Component
from transformers.exporters.decompose import decompose_for_generation, decompose_multimodal, is_multimodal
from transformers.exporters.exporter_dynamo import _VARLEN_ATTENTION_PATHS, DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.exporter_openvino import OpenVINOConfig, OpenVINOExporter
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
# Adding a new skip: pick the most specific scope that applies and add a ``"Name": "reason"``
# entry. Add a new scope key if the existing ones don't fit.


EXPORT_SKIPS: dict[str, dict[str, str]] = {
    # Every backend, every variant.
    "all": {
        "VideoMAEForPreTraining": "Computes its loss even with `return_loss=False`, hitting a data-dependent guard in `mse_loss`.",
        "OpenAIPrivacyFilterModel": (
            "Defaults to eager experts, which loop over `expert_hit.nonzero()` (data-dependent); "
            "`set_experts_implementation('batched_mm')` exports."
        ),
        "OpenAIPrivacyFilterForTokenClassification": "Same as `OpenAIPrivacyFilterModel`.",
        "GlmImageModel": (
            "Vision attention splits by `lengths.tolist()` (data-dependent), and even with the varlen patch the "
            "export runs long."
        ),
        "GlmImageForConditionalGeneration": "Same as `GlmImageModel`.",
    },
    # Every backend, generate path only.
    "generate": {
        "Blip2ForConditionalGeneration": (
            "`generate()` calls the inner language model directly, so the top-level `forward` is never captured."
        ),
        "InstructBlipForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "InstructBlipVideoForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "Kosmos2ForConditionalGeneration": "Same `generate()`-delegation as Blip2.",
        "RecurrentGemmaForCausalLM": (
            "Keeps its recurrent state in module attributes rather than a `Cache`, so it can't be carried between "
            "graph calls."
        ),
        "MoshiForConditionalGeneration": (
            "Its per-step audio kwargs (`moshi_audio_codes`, `user_audio_codes`) reach no graph. TODO: carry "
            "model-specific per-step kwargs through the decomposition."
        ),
        "DiaForConditionalGeneration": (
            "Decodes several codebooks at once (3-D `decoder_input_ids`), so the runtime's 4-D causal decoder "
            "mask has the wrong rank."
        ),
        "Gemma3nForConditionalGeneration": (
            "Its text model takes `per_layer_inputs`, which the multi-modal decomposition does not carry "
            "(`TypeError` on `self.language_model(...)`)."
        ),
        "VibeVoiceForConditionalGeneration": (
            "Generation runs two forwards with different input shapes (prefill + noise scheduler), which the "
            "decomposition can't capture."
        ),
    },
    # Every backend, dynamic-shape only.
    "dynamic": {
        "Sam2Model": "Exporting the Hiera backbone with dynamic H/W exceeds the test timeout.",
        "Sam2VisionModel": (
            "torch's constraint solver raises `NotImplementedError` on Hiera's window-partition guard (`Eq(s/32 - "
            "(s/4)//8, 0)`)."
        ),
        "HieraForPreTraining": (
            "The MAE head keeps a data-dependent number of patches, which `reroll` then guards on. Static shapes work."
        ),
        "SeamlessM4TForSpeechToSpeech": (
            "`q_length > 1 and ... and is_causal` evaluates to a `SymBool`, which SDPA rejects as `is_causal`. "
            "Static shapes work. TODO: handle on the exporter side, see "
            "https://github.com/huggingface/transformers/pull/46196#discussion_r3717333141"
        ),
        "SeamlessM4TForSpeechToText": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
        "SeamlessM4Tv2ForSpeechToSpeech": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
        "SeamlessM4Tv2ForSpeechToText": "Same `SymBool` `is_causal` as `SeamlessM4TForSpeechToSpeech`.",
    },
    # Generate path, dynamic-shape only. Backend-agnostic (it's in the shared decomposition).
    "generate.dynamic": {
        "ReformerModelWithLMHead": (
            "Carries LSH state as `past_buckets_states` instead of a `Cache`, so the runtime can't feed the "
            "exported signature."
        ),
    },
    # Generate path, the *runtime* half only: these export fine, and the export assertions still run —
    # what fails is driving the exported graphs through `generate`.
    "generate.runtime": {
        "KyutaiSpeechToTextForConditionalGeneration": (
            "Encodes audio window by window inside `prepare_inputs_for_generation`, a model-specific loop the "
            "generic runtime doesn't reproduce."
        ),
        "MiniMaxForCausalLM": (
            "`MiniMaxCache` keeps its lightning-attention state in a separate `linear_cache` list that "
            "`materialize_cache_layers` can't fill. TODO: move it into real `LinearAttentionLayer`s."
        ),
        "xLSTMForCausalLM": "`xLSTMCache` lacks the `Cache` API `generate` relies on (`layers`, `is_compileable`).",
        "DeepseekV4ForCausalLM": (
            "Its compressor cache state is `None` until the compressor first fires, so a fresh cache has fewer "
            "pytree leaves than the traced one. The static-cache variants pass."
        ),
        "CsmForConditionalGeneration": (
            "Generates a frame of codebooks per step (3-D `input_ids`), which needs the model's own two-stage loop."
        ),
        "HiggsAudioV2ForConditionalGeneration": (
            "`prepare_inputs_for_generation` rewrites each step's inputs (audio-id masking, last codebook row "
            "only), which the generic runtime doesn't reproduce."
        ),
        "XLMWithLMHeadModel": (
            "`prepare_inputs_for_generation` appends a mask token and builds `langs` every step, inputs only the "
            "model can produce."
        ),
        "XLNetLMHeadModel": (
            "`prepare_inputs_for_generation` builds `perm_mask` / `target_mapping` and a three-token step over "
            "`mems` every step."
        ),
        "BltForCausalLM": "Expects an `EncoderDecoderCache` pair, while `generate` builds a plain `DynamicCache`.",
        "RwkvForCausalLM": (
            "Carries its state under its own `state` kwarg rather than a `Cache`, so the runtime's cache handling "
            "doesn't apply."
        ),
    },
    # The runtime drives these, but not from a *merged* decode — every other variant is served.
    "dynamo.generate.runtime.multi_token": {
        "VoxtralRealtimeForConditionalGeneration": (
            "The merged decode folds 4 audio rows into the feature axis, and the reshape bakes `frames // 4 != "
            "1`, which single-token steps violate. ONNX reshapes at run time and passes."
        ),
    },
    # The runtime drives these, but not from a *merged* decode — every other variant is served.
    "generate.runtime.multi_token": {
        "ClvpForCausalLM": (
            "The merged decode, captured on continuation steps, asserts `attention_mask.size()[1] >= 4`, which "
            "the 3-token prompt fails. TODO: check that the decode graph serves the prompt before dropping the "
            "prefill graph."
        ),
    },
    "generate.multi_token": {
        "ZayaForCausalLM": (
            "The router's `router_hidden_states[:, -seq_length:]` slice specializes the merged decode's query "
            "axis to 2."
        ),
        "ZambaForCausalLM": "Its mixer uses the sequential scan, which unrolls and bakes the query length.",
        "ProphetNetForCausalLM": (
            "The eager forward itself refuses multi-token steps with a cache (`use_cache` only for length-1 "
            "`decoder_input_ids`)."
        ),
    },
    # ONNX, every variant.
    "onnx": {
        "DFineModel": (
            "Every anchor scores the same on the tiny model, so which ones `topk` keeps is arbitrary, and ORT "
            "picks differently from torch."
        ),
        "DFineForObjectDetection": "Same as `DFineModel`.",
        "Deimv2Model": "Same as `DFineModel`.",
        "Deimv2ForObjectDetection": "Same as `DFineModel`.",
        "MMGroundingDinoModel": "Same as `DFineModel`.",
        "MMGroundingDinoForObjectDetection": "Same as `DFineModel`.",
        "PPDocLayoutV2ForObjectDetection": "Same as `DFineModel`.",
        "PPDocLayoutV3ForObjectDetection": "Same as `DFineModel`.",
        "RTDetrModel": "Same as `DFineModel`.",
        "RTDetrForObjectDetection": "Same as `DFineModel`.",
        "RTDetrV2Model": "Same as `DFineModel`.",
        "RTDetrV2ForObjectDetection": "Same as `DFineModel`.",
        "TapasForQuestionAnswering": (
            "The column `argmax` ties on the tiny model and ORT breaks the tie differently, setting every other "
            "cell to `-10000`."
        ),
        "CHMv2ForDepthEstimation": (
            "`run_decompositions` emits a `detach_(alias(...))` pair the functional-graph check rejects. TODO: "
            "file upstream."
        ),
        "PixioModel": "Lowering exceeds the test timeout.",
        "PixioBackbone": "Same `timeout` failure as `PixioModel`.",
    },
    # ONNX, generate path only.
    "onnx.generate": {
        "ReformerModelWithLMHead": "Chunked local attention bakes a constant index past the cached-keys axis under static decode.",
    },
    # ONNX, driving the exported graphs through `generate` only — they export and run standalone.
    "onnx.generate.runtime": {
        "ProphetNetForConditionalGeneration": (
            "The prefill session names its encoder input `encoder_last_hidden_state`, which the runtime feed "
            "doesn't provide. TODO: align the naming."
        ),
    },
    # ONNX, dynamic-shape only.
    "onnx.dynamic": {
        "GroundingDinoModel": (
            "Same `detach_(alias(...))` retrace failure as `CHMv2ForDepthEstimation`, under dynamic shapes only."
        ),
        "GroundingDinoForObjectDetection": "Same as `GroundingDinoModel`.",
        "BigBirdModel": "Lowering exceeds the test timeout under dynamic shapes.",
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
    # ExecuTorch, every variant.
    "executorch": {
        "Qwen3ASRForConditionalGeneration": (
            "Loading fails to allocate a tensor (`0x21`), and the Python runtime's `load_method` takes no allocator."
        ),
        "Siglip2VisionModel": "`_upsample_bilinear2d_aa.out` rejects its own output extent once the axis is dynamic.",
        "Siglip2ForImageClassification": "Same `_upsample_bilinear2d_aa` output-extent check as `Siglip2VisionModel`.",
        "JetMoeModel": (
            "Routes tokens with a data-dependent `split(expert_size.tolist())`, which EXIR's edge-dialect arg "
            "validator can't specialize."
        ),
        "JetMoeForCausalLM": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "JetMoeForSequenceClassification": "Same data-dependent MoE/MoA routing as `JetMoeModel`.",
        "Lfm2VlForConditionalGeneration": (
            "Its NaViT packer makes the vision extents unbacked, and torch's `slice` decomposition guards on them "
            "(`u88 < 0`). TODO: precompute the per-image geometry outside the graph."
        ),
        "Lfm2VlModel": "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`.",
        "MiniCPMV4_6ForConditionalGeneration": "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`.",
        "MiniCPMV4_6Model": "Same unbacked NaViT extents as `Lfm2VlForConditionalGeneration`.",
        "FlavaModel": "XNNPACK's partitioner forms a dependency cycle across the interleaved encoder streams.",
        "FlavaForPreTraining": "Same fused-partition dependency cycle as `FlavaModel` (wraps it).",
        "PPDocLayoutV3ForObjectDetection": (
            "Constant dedup duplicates the shared detection head, and `_unsafe_adjust_original_program` raises "
            "`KeyError` on the second copy."
        ),
        "Deimv2Model": (
            "Same `topk` tie as the ONNX `DFineModel` entry; ExecuTorch picks differently on the DINOv3 variant."
        ),
        "Deimv2ForObjectDetection": "Same as `Deimv2Model`.",
        "PPDocLayoutV2ForObjectDetection": "Same as `Deimv2Model`.",
    },
    "executorch.dynamic": {
        "Qwen3NextModel": "The dynamic lowering exceeds the test timeout. The static variant runs.",
        "Qwen3NextForQuestionAnswering": "Same timeout as `Qwen3NextModel`.",
        "Qwen3NextForSequenceClassification": "Same timeout as `Qwen3NextModel`.",
        "Qwen3NextForTokenClassification": "Same timeout as `Qwen3NextModel`.",
        "Qwen3_5Model": "Same timeout as `Qwen3NextModel`.",
        "Qwen3_5ForConditionalGeneration": "Same as `Qwen3_5Model`.",
        "Qwen3_5ForSequenceClassification": "Same as `Qwen3_5Model`.",
        "Qwen3_5ForTokenClassification": "Same as `Qwen3_5Model`.",
        "OneFormerModel": "The dynamic lowering exceeds the test timeout. The static variant runs.",
        "OneFormerForUniversalSegmentation": "Same as `OneFormerModel`.",
        "MaskFormerForInstanceSegmentation": "The dynamic lowering exceeds the test timeout in symbolic-shape reasoning.",
        "Qwen3_5ForCausalLM": "Same >1000s symbolic-shape lowering as `MaskFormerForInstanceSegmentation`.",
        "Qwen3NextForCausalLM": "Same >1000s symbolic-shape lowering as `MaskFormerForInstanceSegmentation`.",
        "MaskFormerSwinModel": "Windowed attention on symbolic H/W: lowering exceeds the test timeout.",
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
        "Mask2FormerModel": "Lowering exceeds the test timeout under dynamic shapes.",
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
        "Swin2SRModel": (
            "The ahead-of-time arena plan for the windowed attention's unbounded dims comes out at 466 GiB, and "
            "the worker is OOM-killed. Static shapes run."
        ),
        "Swin2SRForImageSuperResolution": "Same as `Swin2SRModel`.",
        "TimesformerModel": "Same `timeout` failure as `Mask2FormerModel`.",
        "TimesformerForVideoClassification": "Same `timeout` failure as `Mask2FormerModel`.",
    },
    "executorch.static": {
        "SplinterForPreTraining": "`aten::nonzero.out` can't resize its data-dependent output under a static export.",
        "MusicFlamingoForConditionalGeneration": "Same `nonzero` resize as `SplinterForPreTraining`.",
        "MusicFlamingoModel": "Same as `MusicFlamingoForConditionalGeneration`.",
        "PaddleOCRVLForConditionalGeneration": "Its image encoder's `view_copy.out` fails `check_view_copy_args` at run time.",
        "Wav2Vec2BertModel": (
            "The conv feature extractor's stacked floor-divisions produce a reshape ExecuTorch rejects (`shape "
            "... is invalid`)."
        ),
        "Wav2Vec2BertForCTC": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForSequenceClassification": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForAudioFrameClassification": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "Wav2Vec2BertForXVector": "Same conv-shape reshape failure as `Wav2Vec2BertModel`.",
        "GroundingDinoModel": "Same shared-head `KeyError` as `PPDocLayoutV3ForObjectDetection`.",
        "GroundingDinoForObjectDetection": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
        "MMGroundingDinoModel": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `bbox_embed` shared-head `KeyError` as `GroundingDinoModel`.",
    },
}


# XNNPACK partitioner configs to withhold, keyed by scope like `EXPORT_SKIPS`. XNNPACK can claim a subgraph its
# own compiler then refuses at method load; withholding that op's config leaves it to the portable kernels and
# keeps the rest delegated. An entry belongs here once the program runs without it: the test passing is not
# enough, the class's `ExecuTorch runtime limitation tolerated` warning must be gone too.
EXECUTORCH_PARTITION_EXCLUDE: dict[str, dict[str, tuple[str, ...]]] = {
    # Dynamic shapes only — the static variants lower and run fully delegated.
    "dynamic": {
        # XNNPACK can't propagate shapes through `unsqueeze_copy` on a vision encoder dynamic on every axis.
        "MuseGlimmerForConditionalGeneration": ("UnsqueezeCopyConfig",),
        "MuseGlimmerModel": ("UnsqueezeCopyConfig",),
    },
    # Both shape variants.
    "all": {
        # Rank-7 activations, past the 6 dimensions XNNPACK can define.
        "UnivNetModel": ("CloneDimOrderConfig", "PermuteConfig", "UnsqueezeCopyConfig", "ViewCopyConfig"),
    },
    # Static shapes only — the dynamic variants lower and run fully delegated.
    "static": {
        # XNNPACK claims `aten.view_copy` in the GatedDeltaNet backbone, then refuses it at method load (`0x1`).
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
        # The flow head's high-rank activations; no smaller set of configs runs.
        "PerceiverForOpticalFlow": ("CloneConfig", "PermuteConfig", "ViewCopyConfig"),
    },
}


# Classes exported without the partitioner (`ExecutorchConfig(partition=False)`), where no single withheld
# config suffices.
EXECUTORCH_DISABLE_PARTITION: dict[str, dict[str, str]] = {
    # Both shape variants.
    "all": {
        "PerceiverForMultimodalAutoencoding": (
            "Needs `aten::rand_like.out`, which ExecuTorch doesn't ship; undelegated, it fails with the tolerated "
            "`0x14`."
        ),
    },
}


# Classes exported without `onnxscript` optimization, keyed by scope like `EXPORT_SKIPS`.
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

_GENERATE_VARIANTS = [
    (dynamic, multi_token, config)
    for dynamic, multi_token, config in itertools.product(
        _EXPORT_SHAPE_MODES, _EXPORT_DECODE_MODES, _EXPORT_GENERATION_CONFIGS
    )
    # A merged multi-token decode under static shapes would freeze its query axis at 2 — a graph no
    # decode step could ever run.
    if not (multi_token and not dynamic)
]


def _generate_variant_name(dynamic, multi_token, config) -> str:
    return (
        ("dynamic" if dynamic else "static")
        + ("_multi_token" if multi_token else "")
        + (f"_{config.cache_implementation}_cache" if config.cache_implementation else "")
    )


GENERATE_EXPORT_PARAMS = parameterized.expand(
    _GENERATE_VARIANTS, name_func=lambda f, _, p: f"{f.__name__}_{_generate_variant_name(*p.args)}"
)


def _executorch_backends() -> tuple[str, ...]:
    """The ExecuTorch backends this install can run: XNNPACK always, MLX where its runtime is registered."""
    backends = ("xnnpack",)
    if is_executorch_available() and importlib.util.find_spec("executorch.backends.mlx") is not None:
        try:
            from executorch.runtime import Runtime
        except ImportError:  # The Python backend can be installed without the native runtime.
            return backends
        if "MLXBackend" in Runtime.get().backend_registry.registered_backend_names:
            backends += ("mlx",)
    return backends


_EXECUTORCH_BACKENDS = _executorch_backends()
EXECUTORCH_EXPORT_PARAMS = parameterized.expand(
    list(itertools.product(_EXECUTORCH_BACKENDS, _EXPORT_SHAPE_MODES)),
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[1] else 'static'}_{p.args[0]}",
)
EXECUTORCH_GENERATE_EXPORT_PARAMS = parameterized.expand(
    [
        (backend, *variant)
        for backend, variant in itertools.product(_EXECUTORCH_BACKENDS, _GENERATE_VARIANTS)
        # MLX has no static cache.
        if not (backend == "mlx" and variant[2].cache_implementation == "static")
    ],
    name_func=lambda f, _, p: f"{f.__name__}_{_generate_variant_name(*p.args[1:])}_{p.args[0]}",
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


@contextmanager
def _tolerating_executorch_limits(label: str):
    """Swallow only the failures where ExecuTorch itself refuses to run an otherwise valid program — a named
    code from the *load* or *execute* phase, or a `bad_alloc` (`_is_executorch_runtime_limit`). A transformers
    export defect surfaces earlier as a `torch.export` error, or later as an output mismatch.

    Anything about how *we* fed the method stays visible: `set_inputs` fails when we hand it something it
    never declared, and a missing input means the decomposition produced a component we cannot feed. A model
    that genuinely needs an exception belongs in `EXPORT_SKIPS`, argued, where it can be seen.
    """
    try:
        yield
    except (RuntimeError, MemoryError) as error:
        if not _is_executorch_runtime_limit(error):
            raise
        # A tolerated failure still reports the test as passed, so say so — otherwise a green run is
        # indistinguishable from one where the program actually ran.
        warnings.warn(
            f"{label}: ExecuTorch runtime limitation tolerated; this test passes without running the "
            f"program — add it to `EXECUTORCH_DISABLE_PARTITION` if lowering it undelegated runs "
            f"instead: {str(error).strip().splitlines()[0]}",
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


# The outputs that are what a `topk` picked: a detector's selected queries and a QA head's top positions. Where
# the scores it picked from tie, which element a kernel keeps is arbitrary — torch's own CPU and CUDA kernels
# disagree — so `_check_outputs_close` leaves these out whenever the model returns anything else to compare.
_SELECTED_BY_TOPK = frozenset(
    {
        "enc_topk_logits",
        "enc_topk_bboxes",
        "start_top_log_probs",
        "start_top_index",
        "end_top_log_probs",
        "end_top_index",
    }
)


def _assert_values_close(case, actual: dict, expected: dict, atol: float, rtol: float) -> None:
    """Compare the tensors a runtime produced against eager's, on eager's device and in eager's dtype.

    Checked on top of the names, since a graph can answer with the right names and the wrong numbers. A
    backend stores at the precision it computes in, which for a half model is `f16` where eager kept
    `bfloat16` — the same numbers to the tolerance, in a type the graph chose. Integers likewise: OpenVINO
    keeps an `int64` counter in `int32` state, since its CPU plugin holds no `i64` variables. What the
    precisions are is checked where the export declares them, not here. The device is eager's for the same
    reason: OpenVINO answers on the host wherever the model was built, and moving its outputs over is what
    lets the model stay on the GPU, where exporting from a GPU-resident model gets exercised too.
    """
    shared = {
        name: actual[name].to(expected[name].device, expected[name].dtype) for name in expected if name in actual
    }
    case.assertTrue(shared, "the runtime produced none of the tensors eager did.")
    case._check_outputs_close(shared, {name: expected[name] for name in shared}, atol=atol, rtol=rtol)


def _assert_openvino_outputs_close(case, runtime, actual: dict, expected: dict, atol: float, rtol: float) -> None:
    """Every leaf eager returns is accounted for — as an output, or as folded state — and holds eager's values.

    An OpenVINO export turns each round-tripped cache tensor into an internal variable the plugin keeps
    between calls, so the graph returns logits and the cache stays behind its `Assign` sinks. Reading it
    back off those sinks is what checks that the graph wrote the keys and values eager did, rather than
    excusing what is missing.
    """
    case.assertTrue(actual, "OpenVINO outputs are empty.")
    runner = getattr(runtime, "runner", None)
    folded = runner.state_tensors() if runner is not None and runner.owns_state else {}
    produced = {**actual, **folded}
    case.assertEqual(set(produced), set(expected))
    _assert_values_close(case, produced, expected, atol, rtol)


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
        return {"model": Component(model, inputs_dict)}

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

    def _maybe_assert_generate_matches_eager(
        self, model_class, components, exported, backend, generation_config, dynamic, multi_token_decode
    ):
        """End-to-end id-parity, wherever the exported graphs can serve `generate`'s loop.

        Checking each component against the call it was captured from leaves the hand-offs between them
        untested — a component is handed its cache, while the loop expects each graph to carry what the one
        before it wrote. The loop runs via the dedicated `prefill` graph (always under dynamic shapes, under
        static ones only with a static cache, whose frozen shapes reproduce every step), or via the
        multi-token decode serving prefill and decode from one graph — and only once every component exported.
        """
        can_split_prefill = "prefill" in exported and (dynamic or _needs_static_cache(generation_config))
        if not (can_split_prefill or (dynamic and multi_token_decode)) or not components.keys() <= exported.keys():
            return
        if self._should_skip(
            model_class,
            generate=True,
            dynamic=dynamic,
            backend=backend,
            multi_token=multi_token_decode,
            generation_config=generation_config,
            runtime=True,
        ):
            return
        self._assert_generate_matches_eager(
            components, exported, backend, generation_config, dynamic, multi_token_decode
        )

    def _assert_generate_matches_eager(
        self, components, exported, backend, generation_config, dynamic, multi_token_decode
    ):
        """Wrap the exported components in `backend`'s `ModelRunner`s, reassemble them into the
        `generate`-driving runtime via `ExportedGenerator` — the same artifacts-plus-configs
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
            *(spec.component for spec in _MODALITY_SPECS),
            *(spec.component for spec in _STREAMING_EMBEDDERS.values()),
        }
        # The runners themselves, not the per-graph runtimes: `ExportedGenerator` assembles the generation
        # loop out of `ModelRunner`s, and a single-graph runtime is an `ExportedModel` *wrapping* one.
        runners = {name: exported[name].runtime().runner for name in components if name in wanted}
        runtime = ExportedGenerator(runners, model.config, model.generation_config)
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
        # A runtime holding its own cache is asked directly how much it holds, because nothing in the
        # outputs would say: a graph attending to an empty cache still answers plausibly, and on a small
        # model those logits sit well inside any tolerance. Only the decode graph is asked — it is the one
        # carrying the whole sequence, prompt included, where a prefill graph holds just what it ran.
        read = exported_out.sequences.shape[1] - 1
        decode_runner = getattr(runtime, "_decode_runner", None)
        if getattr(decode_runner, "owns_state", False) and decode_runner.state_length:
            self.assertEqual(
                decode_runner.state_length,
                read,
                f"{backend} kept {decode_runner.state_length} of the {read} tokens it read: a graph that "
                "owns its cache was never handed what the graph before it wrote",
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
        selected = expected.keys() & _SELECTED_BY_TOPK
        if selected and len(selected) < len(expected):
            actual = {name: value for name, value in actual.items() if name not in selected}
            expected = {name: value for name, value in expected.items() if name not in selected}
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

    def _exporter_and_config(self, backend, model_class, dynamic, executorch_backend=None):
        """The exporter and export config a backend's tests use for `model_class`."""
        if backend == "dynamo":
            return DynamoExporter(), DynamoConfig(dynamic=dynamic)
        if backend == "onnx":
            optimize = _onnx_optimize_enabled(model_class, dynamic)
            return OnnxExporter(), OnnxConfig(dynamic=dynamic, optimize=optimize, external_data=False)
        if backend == "openvino":
            return OpenVINOExporter(), OpenVINOConfig(dynamic=dynamic)
        # Per class: a graph whose delegate refuses its own partitioner's claim lowers undelegated.
        return ExecutorchExporter(), ExecutorchConfig(
            backend=executorch_backend,
            dynamic=dynamic,
            partition=_executorch_partition_enabled(model_class, dynamic),
            partition_exclude=_executorch_partition_exclude(model_class, dynamic),
        )

    def _run_and_compare(self, backend, exported, inputs, expected, label, atol, rtol) -> bool:
        """Run one exported component and assert it matches eager; `False` when ExecuTorch could not run it."""
        if backend == "dynamo":
            with torch.no_grad():
                set_seed(1234)
                actual = exported.runtime()(**copy.deepcopy(inputs))
            self.assertTrue(actual, f"Exported outputs are empty for {label}.")
            self._check_outputs_close(actual, expected, atol=atol, rtol=rtol)
            return True
        if backend == "onnx":
            actual = exported.runtime()(**inputs)
            self.assertTrue(actual, f"ONNX outputs are empty for {label}.")
            self.assertEqual(set(actual.keys()), set(expected.keys()))
            _assert_values_close(self, actual, expected, atol, rtol)
            return True
        if backend == "openvino":
            runtime = exported.runtime()
            _assert_openvino_outputs_close(self, runtime, runtime(**inputs), expected, atol, rtol)
            return True
        # Building the runner stays inside the tolerance: loading the method is where ExecuTorch reports a
        # missing kernel or an oversized arena.
        with _tolerating_executorch_limits(label):
            outputs = exported.runtime()(**inputs)
            tensors = {name: tensor for name, tensor in outputs.items() if isinstance(tensor, torch.Tensor)}
            self.assertEqual(len(tensors), len(expected))
            _assert_values_close(self, tensors, expected, atol, rtol)
            return True
        return False

    def _export_and_compare(
        self,
        backend,
        *,
        dynamic,
        atol,
        rtol,
        executorch_backend=None,
        generate=False,
        multi_token_decode=False,
        generation_config=None,
    ):
        """Export every model class (or its generation components) to `backend` and check each against eager.

        For `generate`, the exported components are then driven through `generate` and checked against eager
        too (`_maybe_assert_generate_matches_eager`). ExecuTorch traces on CPU: XNNPACK targets CPU, and a CUDA
        trace surfaces models that build in-`forward` tensors without `device=`.
        """
        self._skip_if_not_exportable()
        skip_kwargs = {"multi_token": multi_token_decode, "generation_config": generation_config} if generate else {}
        scopes = (backend, executorch_backend) if backend == "executorch" else (backend,)
        device = "cpu" if backend == "executorch" else torch_device

        for model_class in self.all_generative_model_classes if generate else self.all_model_classes:
            if any(
                self._should_skip(model_class, generate=generate, dynamic=dynamic, backend=scope, **skip_kwargs)
                for scope in scopes
            ):
                continue
            exporter, config = self._exporter_and_config(backend, model_class, dynamic, executorch_backend)
            if generate:
                components = self._prepare_export_generate_model_and_inputs(
                    model_class,
                    backend,
                    device=device,
                    generation_config=generation_config,
                    multi_token_decode=multi_token_decode,
                    decoder_writes_cross_cache=exporter.decoder_writes_cross_cache,
                )
            else:
                components = self._prepare_export_model_and_inputs(model_class, backend, device=device)
            eager_outputs = self._collect_eager_outputs(components)

            exported = {}
            for name, component in components.items():
                label = f"{model_class.__name__}/{name}"
                with self.subTest(label):
                    output = exporter.export(component.module, component.inputs, config=config)
                    # Only a component that ran goes on to the generate check.
                    if self._run_and_compare(
                        backend, output, component.inputs, eager_outputs[name], label, atol, rtol
                    ):
                        exported[name] = output

            if generate:
                self._maybe_assert_generate_matches_eager(
                    model_class, components, exported, backend, generation_config, dynamic, multi_token_decode
                )

    # ──────────────────── torch.export tests ─────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_torch_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._export_and_compare("dynamo", dynamic=dynamic, atol=atol, rtol=rtol)

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
        """Export each model class to ONNX and verify output names match eager."""
        self._export_and_compare("onnx", dynamic=dynamic, atol=atol, rtol=rtol)

    # ──────────────────── ExecuTorch tests ───────────────────────

    @EXECUTORCH_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export(self, backend, dynamic, atol=1e-3, rtol=1e-3):
        """Export each model class to ExecuTorch, run it, and verify its outputs match eager."""
        self._export_and_compare("executorch", dynamic=dynamic, executorch_backend=backend, atol=atol, rtol=rtol)


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
        self,
        model_class,
        backend,
        device=torch_device,
        generation_config=None,
        multi_token_decode=False,
        decoder_writes_cross_cache=False,
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
            model,
            inputs_dict,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
            decoder_writes_cross_cache=decoder_writes_cross_cache,
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
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._export_and_compare(
            "dynamo",
            dynamic=dynamic,
            generate=True,
            multi_token_decode=multi_token_decode,
            generation_config=generation_config,
            atol=atol,
            rtol=rtol,
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
    def test_onnx_export_generate(self, dynamic, multi_token_decode, generation_config, atol=1e-3, rtol=1e-3):
        """Export prefill and decode stages to ONNX and verify output names match eager."""
        self._export_and_compare(
            "onnx",
            dynamic=dynamic,
            generate=True,
            multi_token_decode=multi_token_decode,
            generation_config=generation_config,
            atol=atol,
            rtol=rtol,
        )

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_openvino_export(self, dynamic, atol=1e-3, rtol=1e-3):
        """Export each model class to OpenVINO IR and verify output names match eager."""
        self._export_and_compare("openvino", dynamic=dynamic, atol=atol, rtol=rtol)

    @GENERATE_EXPORT_PARAMS
    @slow
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_openvino_export_generate(self, dynamic, multi_token_decode, generation_config, atol=1e-3, rtol=1e-3):
        """Export prefill and decode stages to OpenVINO IR and verify output names match eager."""
        self._export_and_compare(
            "openvino",
            dynamic=dynamic,
            generate=True,
            multi_token_decode=multi_token_decode,
            generation_config=generation_config,
            atol=atol,
            rtol=rtol,
        )

    # ──────────────────── ExecuTorch tests ───────────────────────

    @EXECUTORCH_GENERATE_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export_generate(
        self, backend, dynamic, multi_token_decode, generation_config, atol=1e-3, rtol=1e-3
    ):
        """Export prefill and decode stages to ExecuTorch, run each, and verify they match eager."""
        self._export_and_compare(
            "executorch",
            dynamic=dynamic,
            executorch_backend=backend,
            generate=True,
            multi_token_decode=multi_token_decode,
            generation_config=generation_config,
            atol=atol,
            rtol=rtol,
        )
