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
    decompose_encoder_decoder,
    decompose_prefill_decode,
    get_leaf_tensors,
    is_multicomponent,
)
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)


# ──────────────────────────── skip lists ────────────────────────────

# Model classes skipped for all export backends.
EXPORT_SKIP_MODEL_CLASSES = {
    # VideoMAE computes loss even when return_loss=False, hitting a data-dependent guard in mse_loss.
    # TODO: fix VideoMAE to skip loss computation when labels are not provided.
    "VideoMAEForPreTraining",
    # CHMv2 hits two torch.export bugs: detach_ nodes surviving into run_decompositions, and
    # constant tensors (lifted_tensor) missing from the constants dictionary (SpecViolationError).
    # TODO: fix upstream in PyTorch.
    "CHMv2ForDepthEstimation",
    # PerceiverModel has internal .encoder/.decoder attributes that trigger is_multicomponent(),
    # but they are unusual low-level components with non-standard interfaces that cannot be
    # independently exported. TODO: add proper decomposition support for Perceiver.
    "PerceiverModel",
    # T5Gemma2 hits a VR bound conflict (tgt_bound=VR[64, int_oo] vs src_bound=VR[64, int64_max-2])
    # from get_image_placeholder_mask creating a constraint u0 = 32*s99 during dynamic export.
    # TODO: fix by bounding image placeholder computation to avoid int_oo conflict.
    "T5Gemma2ForConditionalGeneration",
    # VoxtralRealtime passes t_cond=None to decoder layers when time_tensor is not in test inputs,
    # causing AdaRMSNorm to receive None and fail in linear(). TODO: fix model to skip ada_rms_norm
    # when t_cond is None, or ensure test inputs always provide a time_tensor.
    "VoxtralRealtimeForConditionalGeneration",
    # PPDocLayoutV3 passes a ModuleList as a forward argument (order_head), which is not supported
    # by get_auto_dynamic_shapes and torch.export. TODO: refactor to avoid passing modules as inputs.
    "PPDocLayoutV3ForObjectDetection",
}


# Model classes where ONNX optimization must be disabled due to upstream onnxscript bugs.
# SplitToSequence constant folding crashes with "'NoneType' object has no attribute 'ndim'".
# TODO: remove once onnxscript fixes FoldConstantsPass for SplitToSequence with None inputs.
ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES = {
    "ProphetNetModel",
    "ProphetNetForConditionalGeneration",
    "ProphetNetDecoder",
    "ProphetNetForCausalLM",
    "ZoeDepthForDepthEstimation",
}


# Model classes skipped for generate export tests only.
EXPORT_GENERATE_SKIP_MODEL_CLASSES = {
    # These VLMs override generate() and delegate to an inner language model without ever calling
    # the top-level forward(), so decompose_prefill_decode can't capture prefill/decode inputs.
    # TODO: refactor these models to call the top-level forward() instead of using internal submodules
    "Blip2ForConditionalGeneration",
    "InstructBlipForConditionalGeneration",
    "InstructBlipVideoForConditionalGeneration",
    "Kosmos2ForConditionalGeneration",
    # RecurrentGemma stores recurrent/conv state as module attributes instead of using a Cache object,
    # which is incompatible with torch.export (state captured at trace time can't flow between calls).
    # TODO: refactor RecurrentGemma to use a cache-based SSM pattern (like Mamba/Mamba2).
    "RecurrentGemmaForCausalLM",
    # Moshi creates blank_user_audio_codes inside generate() and passes it as a forward kwarg.
    # The resulting ONNX input has mismatched rank (scalar vs 3D) because the tensor is created
    # outside the traced forward graph. TODO: refactor to make blank_user_audio_codes part of the model state.
    "MoshiForConditionalGeneration",
    # UdopForConditionalGeneration's exported decoder output is missing 'attention_mask' vs eager,
    # due to a mismatch in how the encoder cross-attention mask flows through the generate decomposition.
    # TODO: investigate UDOP's encoder-decoder output structure in the context of decomposed export.
    "UdopForConditionalGeneration",
}


# Parameterization for export tests: runs once with dynamic=True and once with dynamic=False.
DYNAMIC_EXPORT_PARAMS = parameterized.expand(
    [(False,), (True,)],
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}",
)

# Maximum time (in seconds) for a single export test before it is killed.
EXPORT_TEST_TIMEOUT = 600


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
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return dict(zip(onnx_names, onnx_outputs))


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
    Multicomponent models (VLMs, encoder-decoders detected by `is_multicomponent`) are
    automatically decomposed and each submodule is tested independently.
    """

    def _skip_if_not_exportable(self):
        """Skip the test if the model architecture is not exportable."""
        if not self.test_torch_exportable:
            self.skipTest(reason="Model architecture is not Dynamo exportable/traceable")

        with open(inspect.getfile(self.all_model_classes[0]), "r") as f:
            source_code = f.read()
            # TODO: rewrite chunked attention loops as tensor ops or use torch._dynamo.allow_in_graph
            if "for q, k, v in zip(*splits)" in source_code:
                self.skipTest(reason="Model architecture uses chunked attention which is not torch exportable")
            # TODO: add use_experts_implementation support to remaining MoE models
            if "for expert" in source_code and "use_experts_implementation" not in source_code:
                self.skipTest(reason="Model architecture uses eager MoE implementation which is not torch exportable")
            # TODO: make get_rope_index export-safe (avoid data-dependent indexing)
            if "get_rope_index" in source_code:
                self.skipTest(reason="Model architecture uses get_rope_index which is not torch exportable")

    def _should_skip(self, model_class, generate=False):
        """Return True if this model class should be skipped for export tests."""
        if model_class.__name__ in EXPORT_SKIP_MODEL_CLASSES:
            return True
        if generate and model_class.__name__ in EXPORT_GENERATE_SKIP_MODEL_CLASSES:
            return True
        return False

    def _prepare_export_model_and_inputs(self, model_class):
        """Create model and forward inputs ready for export.

        Returns:
            List of `(name, model, inputs)` triplets — one per component.
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

        if is_multicomponent(model):
            return decompose_encoder_decoder(model, inputs_dict)
        return [("model", model, inputs_dict)]

    def _collect_eager_outputs(self, components):
        """Run eager forward for each component and return a ``{name: leaf_tensors}`` dict."""
        eager_outputs = {}
        for name, model, inputs in components:
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

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_torch_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter(export_config=DynamoConfig(dynamic=dynamic))

        for model_class in self.all_model_classes:
            if self._should_skip(model_class):
                continue

            components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, f"Exported outputs are empty for {name}.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_onnx_export(self, dynamic):
        """Export each model class to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class):
                continue

            optimize = model_class.__name__ not in ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES
            exporter = OnnxExporter(export_config=OnnxConfig(dynamic=dynamic, optimize=optimize))

            components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs)
                    set_seed(1234)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, f"ONNX outputs are empty for {name}.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_executorch_export(self):
        """Export each model class to ExecuTorch (xnnpack on CPU, cuda on GPU) and verify no errors."""

        self._skip_if_not_exportable()
        backend = "cuda" if torch_device == "cuda" else "xnnpack"
        exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend=backend))

        for model_class in self.all_model_classes:
            if self._should_skip(model_class):
                continue

            components = self._prepare_export_model_and_inputs(model_class)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs)


class ExportGenerateTesterMixin:
    """Mixin providing generation-aware export tests for torch.export, ONNX, and ExecuTorch backends.

    Mix into a model test class alongside ``ExportTesterMixin`` and ``GenerationTesterMixin``.

    Required attributes on the host class (in addition to those from ``ExportTesterMixin``):
    - ``all_generative_model_classes`` — iterable of generative model class objects to test.
    - ``prepare_config_and_inputs_for_generate()`` — returns ``(config, inputs_dict)`` suitable
      for ``model.generate()``.

    Each generative model is decomposed into prefill and decode components via
    :func:`decompose_prefill_decode`.  Multicomponent models (VLMs) additionally decompose
    the prefill stage into individual submodules via :func:`decompose_encoder_decoder`.
    """

    def _prepare_export_generate_model_and_inputs(self, model_class):
        """Decompose a generative model into exportable components.

        For multicomponent models (VLMs, encoder-decoders): decomposes the prefill stage
        into individual submodules plus the decode stage.
        For decoder-only models: returns prefill and decode components.

        Returns:
            List of `(name, model, inputs)` triplets — one per component.
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        # create prefill/decode copies of the model
        stages = decompose_prefill_decode(model, inputs_dict)

        if is_multicomponent(model):
            _, prefill_model, prefill_inputs = stages[0]
            components = decompose_encoder_decoder(prefill_model, prefill_inputs)
            return components + stages[1:]  # encoder-decoder components + decode stage

        return stages

    # ──────────────────── torch.export tests ─────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_torch_export_generate(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter(export_config=DynamoConfig(dynamic=dynamic))

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

            components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_onnx_export_generate(self, dynamic):
        """Export prefill and decode stages to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

            optimize = model_class.__name__ not in ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES
            exporter = OnnxExporter(export_config=OnnxConfig(dynamic=dynamic, optimize=optimize))

            components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs)
                    set_seed(1234)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_executorch_export_generate(self):
        """Export prefill and decode stages to ExecuTorch and verify no errors."""

        self._skip_if_not_exportable()
        backend = "cuda" if torch_device == "cuda" else "xnnpack"
        exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend=backend))

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

            components = self._prepare_export_generate_model_and_inputs(model_class)

            for name, model, inputs in components:
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs)
