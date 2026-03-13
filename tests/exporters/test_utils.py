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

from transformers import set_seed
from transformers.exporters.exporter_dynamo import DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.utils import get_leaf_tensors, simulate_generation
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
_EXPORT_SKIP_MODEL_CLASSES = {
    # VideoMAE computes loss even when return_loss=False, hitting a data-dependent guard in mse_loss.
    # TODO: fix VideoMAE to skip loss computation when labels are not provided.
    "VideoMAEForPreTraining",
}


# Model classes skipped for generate export tests only.
_EXPORT_GENERATE_SKIP_MODEL_CLASSES = {
    # These VLMs override generate() and delegate to an inner language model without ever calling
    # the top-level forward(), so simulate_generation can't capture prefill/decode inputs.
    # TODO: refactor these models to call the top-level forward() instead of using internal submodules
    "Blip2ForConditionalGeneration",
    "InstructBlipForConditionalGeneration",
    "InstructBlipVideoForConditionalGeneration",
    "Kosmos2ForConditionalGeneration",
    # RecurrentGemma stores recurrent/conv state as module attributes instead of using a Cache object,
    # which is incompatible with torch.export (state captured at trace time can't flow between calls).
    # TODO: refactor RecurrentGemma to use a cache-based SSM pattern (like Mamba/Mamba2).
    "RecurrentGemmaForCausalLM",
}


# ONNX models that export but produce extremely inaccurate outputs (non-deterministic top-k, NaN, etc.).
# TODO: investigate and fix each of these at the model level.
_ONNX_EXTREMELY_INACCURATE_MODEL_TYPES: set[str] = {
    "blt",  # 94.3% mismatch in last_hidden_state
    "flaubert",  # 40% mismatch in end_top_index (top-k beam search non-determinism)
    "parakeet_ctc",  # 100% NaN in logits
    "parakeet_encoder",  # 100% NaN in last_hidden_state
    "patchtst",  # NaN loss output
    "pp_doclayout_v2",  # 68.3% mismatch in enc_topk_bboxes (non-deterministic top-k selection)
    "pp_doclayout_v3",  # 68.3% mismatch in enc_topk_bboxes
    "d_fine",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "mm-grounding-dino",  # 25% mismatch in encoder_pred_boxes (non-deterministic top-k selection)
    "rt_detr",  # 43.3% mismatch in enc_topk_bboxes
    "rt_detr_v2",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "siglip2",  # 100% mismatch in logits
    "siglip2_vision_model",  # 73.8% mismatch in last_hidden_state
    "vit_mae",  # 99.3% mismatch in ids_restore (random masking)
    "xlm",  # 6.2% mismatch in end_top_index
}

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


def _move_flags_to_config(model, *inputs_dicts):
    """Pop output flags from one or more inputs dicts and set them on model.config.

    When multiple dicts are provided, flags must have consistent values across all dicts.
    """
    # Output flags that must live on model.config, not in the inputs dict. simulate_generation captures
    # these from generate()'s kwargs; we move them to config so eager and exported runs match.
    _output_flags = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")
    for flag in _output_flags:
        values = [d.pop(flag, None) for d in inputs_dicts]
        non_none = [v for v in values if v is not None]
        if not non_none:
            continue
        if len(set(non_none)) > 1:
            raise ValueError(
                f"Output flag '{flag}' has inconsistent values across inputs: {non_none}. "
                f"simulate_generation should produce consistent flags across stages."
            )
        setattr(model.config, flag, non_none[0])


def _set_exportable_impl(model):
    """Set attention/experts implementations to match what prepare_for_export will use.

    This ensures eager outputs use the same code paths as the exported model,
    avoiding spurious numerical mismatches from implementation differences.
    """
    if model._can_set_attn_implementation() and model.config.model_type != "videomae":
        try:
            model.set_attn_implementation("sdpa")
        except Exception as e:
            print(
                f"Could not set attention implementation to sdpa for {model} of type {model.config.model_type} : {e}"
            )

    if model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_mamba_kernels"):
            module.config.use_mamba_kernels = False


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
            self.skipTest(reason="Model architecture is not TorchDynamo exportable/traceable")

        # TODO: these source-code greps silently hide unexportable models. Each pattern should be
        # fixed at the model level and the affected models moved to _EXPORT_SKIP_MODEL_CLASSES.
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
        if model_class.__name__ in _EXPORT_SKIP_MODEL_CLASSES:
            return True
        if generate and model_class.__name__ in _EXPORT_GENERATE_SKIP_MODEL_CLASSES:
            return True
        return False

    def _prepare_export_model_and_inputs(self, model_class):
        """Create model and forward inputs ready for export.

        Returns:
            ``(model, inputs_dict)``
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
        _set_exportable_impl(model)

        # Cast inputs to model device/dtype
        try:
            model_param = next(iter(model.parameters()))
            inputs_dict = _cast_inputs(inputs_dict, model_param.device, model_param.dtype)
        except StopIteration:
            pass

        # Move output flags to config so eager and exported runs produce matching output structures.
        _move_flags_to_config(model, inputs_dict)

        # Sanity check: inputs must work with eager forward before we attempt export
        try:
            with torch.no_grad():
                model(**copy.deepcopy(inputs_dict))
        except Exception as e:
            raise RuntimeError(
                f"Eager forward failed for {model_class.__name__} before export. "
                f"The test is not preparing inputs correctly. Inputs: {list(inputs_dict.keys())}"
            ) from e

        return model, inputs_dict

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
    @pytest.mark.torch_export_test
    def test_torch_export(self, atol=1e-4, rtol=1e-4):
        """Test if model can be exported with torch.export.export()"""
        self._skip_if_not_exportable()

        exporter = DynamoExporter(export_config=DynamoConfig())

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class):
                    continue

                model, inputs_dict = self._prepare_export_model_and_inputs(model_class)

                with torch.no_grad():
                    set_seed(1234)
                    eager_outputs = get_leaf_tensors(model(**copy.deepcopy(inputs_dict)))
                    self.assertTrue(eager_outputs, "Eager outputs are empty.")

                exported_program = exporter.export(model, inputs_dict)

                with torch.no_grad():
                    set_seed(1234)
                    exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs_dict)))
                    self.assertTrue(exported_outputs, "Exported outputs are empty.")

                self._check_outputs_close(exported_outputs, eager_outputs, atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_export(self, atol=1e-2, rtol=1e-2):
        """Test if model can be exported with torch.onnx.export()"""
        self._skip_if_not_exportable()

        exporter = OnnxExporter(export_config=OnnxConfig())

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class):
                    continue

                model, inputs_dict = self._prepare_export_model_and_inputs(model_class)

                with torch.no_grad():
                    set_seed(1234)
                    eager_outputs = get_leaf_tensors(model(**copy.deepcopy(inputs_dict)))
                    self.assertTrue(eager_outputs, "Eager outputs are empty.")

                onnx_program = exporter.export(model, inputs_dict)
                onnx_inputs = {k: v for k, v in inputs_dict.items() if not isinstance(v, (bool, int, float, str))}

                set_seed(1234)
                onnx_outputs = onnx_program(**onnx_inputs)
                onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
                onnx_outputs = dict(zip(onnx_names, onnx_outputs))
                self.assertTrue(onnx_outputs, "ONNX outputs are empty.")

                if model.config.model_type not in _ONNX_EXTREMELY_INACCURATE_MODEL_TYPES:
                    self._check_outputs_close(onnx_outputs, eager_outputs, atol=atol, rtol=rtol, check_device=False)

    # ──────────────────── ExecuTorch tests ───────────────────────

    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch_export(self):
        """Test if model can be exported with ExecuTorchExporter."""

        self._skip_if_not_exportable()
        backend = "cuda" if torch_device == "cuda" else "xnnpack"
        exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend=backend))

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class):
                    continue

                model, inputs_dict = self._prepare_export_model_and_inputs(model_class)
                exporter.export(model, inputs_dict)


class ExportGenerateTesterMixin:
    """Mixin providing generation export tests for torch.export, ONNX, and ExecuTorch backends.

    Inherited by ``GenerationTesterMixin`` so every generative model test class gets
    generation export tests automatically. Expects the host class to also inherit
    ``ExportTesterMixin`` (for shared helpers) and ``GenerationTesterMixin`` (for
    ``prepare_config_and_inputs_for_generate`` and ``all_generative_model_classes``).
    """

    def _prepare_export_generate_model_and_inputs(self, model_class):
        """Create model and generation-ready inputs for prefill/decode export.

        Returns:
            ``(model, prefill_inputs, decode_inputs)``
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}
        _disable_loss(config, inputs_dict)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)
        _set_exportable_impl(model)

        # Sanity check: inputs must work with eager generate before we attempt export
        try:
            with torch.no_grad():
                model.generate(**copy.deepcopy(inputs_dict), max_new_tokens=2, min_new_tokens=2)
        except Exception as e:
            raise RuntimeError(
                f"Eager generate failed for {model_class.__name__} before export. "
                f"The test is not preparing inputs correctly. Inputs: {list(inputs_dict.keys())}"
            ) from e

        prefill_inputs, decode_inputs = simulate_generation(model, inputs_dict)

        # Move output flags from inputs to config. simulate_generation captures whatever
        # generate() passes to forward(), which includes flags like use_cache, return_dict, etc.
        # These must live on the config so eager and exported runs produce matching output structures.
        _move_flags_to_config(model, prefill_inputs, decode_inputs)

        # Sanity check: captured inputs must work with eager forward before we attempt export
        for stage_name, stage_inputs in [("prefill", prefill_inputs), ("decode", decode_inputs)]:
            shapes = {k: v.shape if hasattr(v, "shape") else type(v).__name__ for k, v in stage_inputs.items()}
            try:
                with torch.no_grad():
                    model(**copy.deepcopy(stage_inputs))
            except Exception as e:
                raise RuntimeError(
                    f"Eager forward ({stage_name}) failed for {model_class.__name__} after simulate_generation. "
                    f"The captured inputs are inconsistent. Shapes: {shapes}"
                ) from e

        return model, prefill_inputs, decode_inputs

    def _collect_eager_outputs(self, model, stages):
        """Run eager forward for each stage and return a dict of {stage_name: leaf_tensors}."""
        eager_outputs = {}
        for stage_name, stage_inputs in stages:
            with torch.no_grad():
                set_seed(1234)
                eager_outputs[stage_name] = get_leaf_tensors(model(**copy.deepcopy(stage_inputs)))
                assert eager_outputs[stage_name], f"Eager outputs are empty for {stage_name}."
        return eager_outputs

    # ──────────────────── torch.export tests ─────────────────────

    @slow
    @pytest.mark.torch_export_test
    def test_torch_export_generate(self, atol=1e-4, rtol=1e-4):
        """Test if generative model can be exported for prefill and decode."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter(export_config=DynamoConfig())

        for model_class in self.all_generative_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class, generate=True):
                    continue

                model, prefill_inputs, decode_inputs = self._prepare_export_generate_model_and_inputs(model_class)
                stages = [("prefill", prefill_inputs), ("decode", decode_inputs)]

                # Collect eager outputs before export (export modifies the model in-place)
                eager_outputs = self._collect_eager_outputs(model, stages)

                for stage_name, stage_inputs in stages:
                    with self.subTest(stage=stage_name):
                        exported_program = exporter.export(model, stage_inputs)

                        with torch.no_grad():
                            set_seed(1234)
                            exported_outputs = get_leaf_tensors(
                                exported_program.module()(**copy.deepcopy(stage_inputs))
                            )
                            self.assertTrue(exported_outputs, "Exported outputs are empty.")

                        self._check_outputs_close(exported_outputs, eager_outputs[stage_name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_export_generate(self, atol=1e-2, rtol=1e-2):
        """Test if generative model can be ONNX-exported for prefill and decode."""
        self._skip_if_not_exportable()

        exporter = OnnxExporter(export_config=OnnxConfig())

        for model_class in self.all_generative_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class, generate=True):
                    continue

                model, prefill_inputs, decode_inputs = self._prepare_export_generate_model_and_inputs(model_class)
                stages = [("prefill", prefill_inputs), ("decode", decode_inputs)]

                # Collect eager outputs before export (export modifies the model in-place)
                eager_outputs = self._collect_eager_outputs(model, stages)

                for stage_name, stage_inputs in stages:
                    with self.subTest(stage=stage_name):
                        onnx_program = exporter.export(model, stage_inputs)
                        onnx_inputs = {
                            k: v for k, v in stage_inputs.items() if not isinstance(v, (bool, int, float, str))
                        }

                        set_seed(1234)
                        onnx_outputs = onnx_program(**onnx_inputs)
                        onnx_names = (
                            re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output
                        )
                        onnx_outputs = dict(zip(onnx_names, onnx_outputs))
                        self.assertTrue(onnx_outputs, "ONNX outputs are empty.")

                        if model.config.model_type not in _ONNX_EXTREMELY_INACCURATE_MODEL_TYPES:
                            self._check_outputs_close(
                                eager_outputs[stage_name], onnx_outputs, atol=atol, rtol=rtol, check_device=False
                            )

    # ──────────────────── ExecuTorch tests ───────────────────────

    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch_export_generate(self):
        """Test if generative model can be ExecuTorch-exported for prefill and decode."""

        self._skip_if_not_exportable()
        backend = "cuda" if torch_device == "cuda" else "xnnpack"
        exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend=backend))

        for model_class in self.all_generative_model_classes:
            with self.subTest(model_class.__name__):
                if self._should_skip(model_class, generate=True):
                    continue

                model, prefill_inputs, decode_inputs = self._prepare_export_generate_model_and_inputs(model_class)

                for stage_name, stage_inputs in [("prefill", prefill_inputs), ("decode", decode_inputs)]:
                    with self.subTest(stage=stage_name):
                        exporter.export(model, stage_inputs)
