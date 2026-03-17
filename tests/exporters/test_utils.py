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

# Model classes skipped for dynamic export tests only (static passes).
EXPORT_SKIP_DYNAMIC_MODEL_CLASSES = {
    # SEW and SEW-D use DisentangledSelfAttention where the attention mask gets baked as a constant
    # in the ONNX graph (size = sample export seq_len) due to fixed-size relative position embeddings.
    # At ORT inference with a different seq_len the mask and scores shapes are incompatible.
    # TODO: identify exact ONNX node where the fixed size originates and make it dynamic.
    "SEWDModel",
    "SEWModel",
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
EXPORT_TEST_TIMEOUT = 60


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


def _disable_loss(config, inputs_dict):
    """Remove labels and disable loss computation so eager and exported outputs match."""
    for key in ("labels", "future_values", "return_loss"):
        inputs_dict.pop(key, None)
    config.return_loss = False


def _run_onnx_program(onnx_program, inputs) -> dict:
    """Run an ONNX program and return outputs as a ``{name: tensor}`` dict."""
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return dict(zip(onnx_names, onnx_outputs))


# ──────────────────────────── mixins ────────────────────────────


class ExportTesterMixin:
    """Mixin providing non-generate export tests for torch.export, ONNX, and ExecuTorch backends.

    Inherited by ``ModelTesterMixin`` so every model test class gets export tests automatically.
    Expects the host class to provide: ``all_model_classes``,
    ``model_tester``, ``test_torch_exportable``, ``_prepare_for_class``.
    """

    def _skip_if_not_exportable(self):
        """Skip the test if the model architecture is not exportable."""
        if not self.test_torch_exportable:
            self.skipTest(reason="Model architecture is not Dynamo exportable/traceable")

        # TODO: these source-code greps silently hide unexportable models. Each pattern should be
        # fixed at the model level and the affected models moved to EXPORT_SKIP_MODEL_CLASSES.
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

    def _should_skip(self, model_class, generate=False, dynamic=False):
        """Return True if this model class should be skipped for export tests."""
        if model_class.__name__ in EXPORT_SKIP_MODEL_CLASSES:
            return True
        if generate and model_class.__name__ in EXPORT_GENERATE_SKIP_MODEL_CLASSES:
            return True
        if dynamic and model_class.__name__ in EXPORT_SKIP_DYNAMIC_MODEL_CLASSES:
            return True
        return False

    def _prepare_export_model_and_inputs(self, model_class):
        """Create model and forward inputs ready for export.

        Returns:
            List of ``(name, model, inputs)`` triplets — one per component.
        """
        if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}
        _disable_loss(config, inputs_dict)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        # Cast inputs to model device/dtype
        try:
            model_param = next(iter(model.parameters()))
            inputs_dict = _cast_inputs(inputs_dict, model_param.device, model_param.dtype)
        except StopIteration:
            pass

        # Sanity check: inputs must work with eager forward before we attempt export
        try:
            with torch.no_grad():
                model(**copy.deepcopy(inputs_dict))
        except Exception as e:
            raise RuntimeError(
                f"Eager forward failed for {model_class.__name__} before export. "
                f"The test is not preparing inputs correctly. Inputs: {list(inputs_dict.keys())}"
            ) from e

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
        """Test if model can be exported with torch.export.export()"""
        self._skip_if_not_exportable()

        exporter = DynamoExporter(export_config=DynamoConfig(dynamic=dynamic))

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic):
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
        """Test if model can be exported with torch.onnx.export()"""
        self._skip_if_not_exportable()

        exporter = OnnxExporter(export_config=OnnxConfig(dynamic=dynamic))

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic):
                continue

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
        """Test if model can be exported with ExecuTorchExporter."""

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
    """Mixin providing generation export tests for torch.export, ONNX, and ExecuTorch backends.

    Inherited by ``GenerationTesterMixin`` so every generative model test class gets
    generation export tests automatically. Expects the host class to also inherit
    ``ExportTesterMixin`` (for shared helpers) and ``GenerationTesterMixin`` (for
    ``prepare_config_and_inputs_for_generate`` and ``all_generative_model_classes``).
    """

    def _prepare_export_generate_model_and_inputs(self, model_class):
        """Decompose a generative model into exportable components.

        For multicomponent models (VLMs, encoder-decoders): decomposes the prefill stage
        into individual submodules plus the decode stage.
        For decoder-only models: returns prefill and decode components.

        Returns:
            List of ``(name, model, inputs)`` triplets — one per component.
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}
        _disable_loss(config, inputs_dict)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        prefill_decode = decompose_prefill_decode(model, inputs_dict)

        if is_multicomponent(model):
            _, prefill_model, prefill_inputs = prefill_decode[0]
            return decompose_encoder_decoder(prefill_model, prefill_inputs) + [prefill_decode[1]]
        return prefill_decode

    # ──────────────────── torch.export tests ─────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    def test_torch_export_generate(self, dynamic, atol=1e-4, rtol=1e-4):
        """Test if generative model can be exported for prefill and decode."""
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
        """Test if generative model can be ONNX-exported for prefill and decode."""
        self._skip_if_not_exportable()

        exporter = OnnxExporter(export_config=OnnxConfig(dynamic=dynamic))

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

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
        """Test if generative model can be ExecuTorch-exported for prefill and decode."""

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
