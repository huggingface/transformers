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
import inspect
import re

import pytest
import torch
from parameterized import parameterized

from transformers import set_seed
from transformers.exporters.exporter_dynamo import DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.utils import (
    decompose_for_generation,
    decompose_multimodal,
    get_leaf_tensors,
    is_multimodal,
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
    },
    # ONNX, every variant.
    "onnx": {
        "CHMv2ForDepthEstimation": (
            "`run_decompositions` retraces through aot_autograd which emits a `detach_(alias(...))` "
            "pair the functional-graph assertion rejects (independent of any source `.detach()` — "
            "verified). Torch export works. TODO: file upstream `torch.export` issue."
        ),
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
        "Sam2Model": "Same Hiera-backbone dynamic-shape budget overrun as `Sam2VisionModel`.",
    },
    # ExecuTorch — lowering failures grouped by root cause; see the first entry of each
    # `Same ... as` chain for the full description.
    "executorch": {
        "BarkFineModel": (
            "ExecuTorch memory planning miscomputes the tensor spec (`buffer of size N, expected nbytes of M`) — a dtype-size mismatch in the lowered program."
        ),
        "ClvpModelForConditionalGeneration": (
            "A pass-through output aliases an input (`Output node is already in the inputs`)."
        ),
        "ColQwen2ForRetrieval": (
            "ExecuTorch dim-order lowering requires a copying view (`Cannot view a tensor ... with shape/strides`)."
        ),
        "DabDetrModel": ("XNNPACK partitioner: `Attempting to convert non-NHWC compatible node to NHWC`."),
        "DabDetrForObjectDetection": "Same `nhwc` failure as `DabDetrModel`.",
        "Ernie4_5_VLMoeModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Ernie4_5_VLMoeForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "FlavaForPreTraining": ("XNNPACK partitioner: `Invalid partition, found dependency cycles`."),
        "GPT2Model": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2LMHeadModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2DoubleHeadsModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForQuestionAnswering": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForSequenceClassification": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForTokenClassification": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Gemma3nModel": "Same `spec` failure as `BarkFineModel`.",
        "Gemma3nForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Glm46VModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm46VForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vMoeModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vMoeForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmImageModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmImageForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmOcrModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmOcrForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GroundingDinoModel": ("Lowering exceeds the test timeout under dynamic shapes."),
        "GroundingDinoForObjectDetection": "Same `timeout` failure as `GroundingDinoModel`.",
        "InstructBlipModel": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipVideoForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipVideoModel": "Same `spec` failure as `BarkFineModel`.",
        "MMGroundingDinoModel": "Same `timeout` failure as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `timeout` failure as `GroundingDinoModel`.",
        "MiniMaxM3VLModel": ("Serialization rejects an i64 constant (`bad number for type int32`)."),
        "MiniMaxM3SparseForConditionalGeneration": "Same `int32` failure as `MiniMaxM3VLModel`.",
        "PPDocLayoutV3ForObjectDetection": ("Delegation drops a referenced weight (`KeyError` on a state-dict key)."),
        "PaddleOCRVLForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "PerceptionLMModel": "Same `passthrough` failure as `ClvpModelForConditionalGeneration`.",
        "PerceptionLMForConditionalGeneration": "Same `passthrough` failure as `ClvpModelForConditionalGeneration`.",
        "Qwen2VLModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2VLForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2_5OmniThinkerForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Qwen2_5_VLModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2_5_VLForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3OmniMoeThinkerForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Qwen3_5Model": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForSequenceClassification": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForTokenClassification": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5MoeModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5MoeForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
    },
    "executorch.generate": {
        "PPFormulaNetForConditionalGeneration": (
            "ExecuTorch memory planning miscomputes the tensor spec (`buffer of size N, expected nbytes of M`) — a dtype-size mismatch in the lowered program."
        ),
    },
    "executorch.dynamic": {
        "BigBirdModel": ("Lowering exceeds the test timeout under dynamic shapes."),
        "BigBirdForPreTraining": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForCausalLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `BigBirdModel`.",
        "DepthProModel": (
            "`_ViewSpec is incompatible with its base` — mixed shape dynamism between a view and its base."
        ),
        "DepthProForDepthEstimation": "Same `viewspec` failure as `DepthProModel`.",
        "DonutSwinModel": (
            "ExecuTorch memory planning overflows under unbounded dynamic shapes (`mem_offset does not fit in 64 bits`)."
        ),
        "DonutSwinForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "Mask2FormerModel": "Same `timeout` failure as `BigBirdModel`.",
        "Mask2FormerForUniversalSegmentation": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerModel": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerForInstanceSegmentation": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerSwinModel": "Same `overflow` failure as `DonutSwinModel`.",
        "MaskFormerSwinBackbone": "Same `overflow` failure as `DonutSwinModel`.",
        "MllamaModel": "Same `overflow` failure as `DonutSwinModel`.",
        "MllamaForConditionalGeneration": "Same `overflow` failure as `DonutSwinModel`.",
        "PvtModel": "Same `viewspec` failure as `DepthProModel`.",
        "PvtForImageClassification": "Same `viewspec` failure as `DepthProModel`.",
        "Sam2Model": ("Delegation drops a referenced weight (`KeyError` on a state-dict key)."),
        "Sam2VisionModel": "Same `timeout` failure as `BigBirdModel`.",
        "Swin2SRModel": "Same `overflow` failure as `DonutSwinModel`.",
        "Swin2SRForImageSuperResolution": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinModel": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinBackbone": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinForMaskedImageModeling": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2Model": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2ForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2ForMaskedImageModeling": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2Backbone": "Same `overflow` failure as `DonutSwinModel`.",
        "VitDetModel": "Same `viewspec` failure as `DepthProModel`.",
        "VitDetBackbone": "Same `viewspec` failure as `DepthProModel`.",
        "Wav2Vec2BertForCTC": ("`flatc` schema compilation fails when serializing the program."),
        "Wav2Vec2BertModel": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForSequenceClassification": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForAudioFrameClassification": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForXVector": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
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
    },
}


# Parameterization for export tests: runs once with dynamic=True and once with dynamic=False.
DYNAMIC_EXPORT_PARAMS = parameterized.expand(
    [(False,), (True,)],
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}",
)

# Maximum time (in seconds) for a single export test before it is killed.
EXPORT_TEST_TIMEOUT = 1000

# Minimum torch version the exporters target — older releases lack `torch.export` features the
# exporters rely on, so the export sweep is skipped (not failed) below this. Sourced from the
# exporter itself so the test and the runtime check can't drift apart.
MIN_EXPORT_TORCH_VERSION = DynamoExporter.min_versions["torch"]


# ──────────────────────────── helpers ────────────────────────────


def _cast_inputs(obj, device, dtype):
    """Recursively move tensors to `device`, casting floating-point tensors to `dtype`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype) if obj.is_floating_point() else obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _cast_inputs(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_inputs(v, device, dtype) for v in obj)
    return obj


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


def _onnx_optimize_enabled(model_class, dynamic: bool) -> bool:
    """Return whether onnxscript optimisation should run for this model under this shape mode.

    Mirrors ``_should_skip``'s scope walk on ``ONNX_DISABLE_OPTIMIZE`` — ``"all"`` always
    applies; ``"dynamic"`` adds the dynamic-only entries.
    """
    name = model_class.__name__
    scopes = ["all"] + (["dynamic"] if dynamic else [])
    return not any(name in ONNX_DISABLE_OPTIMIZE.get(scope, {}) for scope in scopes)


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

    def _should_skip(self, model_class, generate=False, dynamic=False, backend=None):
        """Return True if this model class should be skipped for export tests.

        Walks the scopes in ``EXPORT_SKIPS`` from broad to specific that match the current
        ``(backend, generate, dynamic)`` triple — ``"all"`` always applies, ``"generate"`` only
        for generate tests, ``"<backend>"`` for that backend, and ``"<backend>.<variant>"`` for
        the more-specific intersections.
        """
        name = model_class.__name__
        scopes = ["all"]
        if generate:
            scopes.append("generate")
        if backend:
            scopes.append(backend)
            if generate:
                scopes.append(f"{backend}.generate")
            if dynamic:
                scopes.append(f"{backend}.dynamic")
        return any(name in EXPORT_SKIPS.get(scope, {}) for scope in scopes)

    def _prepare_export_model_and_inputs(self, model_class):
        """Create model and forward inputs ready for export.

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
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        # Cast inputs to model device/dtype
        try:
            model_param = next(iter(model.parameters()))
            inputs_dict = _cast_inputs(inputs_dict, model_param.device, model_param.dtype)
        except StopIteration:
            pass

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

    def _check_outputs_close(self, actual, expected, atol, rtol, check_device=True):
        """Assert outputs are close, allowing up to 5% element-level mismatch."""
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
    def test_torch_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class):
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

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    def test_onnx_export(self, dynamic):
        """Export each model class to ONNX and verify output names match eager."""
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

    # ──────────────────── ExecuTorch tests ───────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    def test_executorch_export(self, dynamic):
        """Export each model class to ExecuTorch (xnnpack on CPU, cuda on GPU) and verify no errors."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="executorch"):
                continue

            components = self._prepare_export_model_and_inputs(model_class)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs, config=config)


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

    def _prepare_export_generate_model_and_inputs(self, model_class):
        """Decompose a generative model into exportable components.

        For multi-modal models: decomposes the prefill stage into individual submodules plus the decode stage.
        For decoder-only models: returns prefill and decode components.

        Returns:
            Dict of `{name: (model, inputs)}` — one entry per component.
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        return decompose_for_generation(model, inputs_dict)

    # ──────────────────── torch.export tests ─────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    def test_torch_export_generate(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

            components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    def test_onnx_export_generate(self, dynamic):
        """Export prefill and decode stages to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True, dynamic=dynamic, backend="onnx"):
                continue

            optimize = _onnx_optimize_enabled(model_class, dynamic)
            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic, optimize=optimize)

            components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    set_seed(1234)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    def test_executorch_export_generate(self, dynamic):
        """Export prefill and decode stages to ExecuTorch and verify no errors."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True, dynamic=dynamic, backend="executorch"):
                continue

            components = self._prepare_export_generate_model_and_inputs(model_class)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs, config=config)
