# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""QNN (Qualcomm HTP / NPU) hooks for the ExecuTorch exporter.

"Shift-left" of QNN export from optimum-executorch into transformers: three hooks that the
``ExecutorchExporter`` dispatches when ``ExecutorchConfig(backend="qnn")`` is used.

- ``prepare_for_qnn``  — pre-export prep: static-KV-cache wrap for decoder-only LMs + HTP compiler spec.
- ``quantize_for_qnn`` — PT2E quantization (8a8w/16a4w/…) *after* torch.export (``_POST_EXPORT_TRANSFORM``).
- ``_lower_for_qnn``   — backend-specific lowering: QNN pre-export + edge passes inside a QnnManagerContext.

All ExecuTorch/QNN imports are lazy (inside the functions) so importing this module never requires the
``executorch`` package to be installed — only the ``qnn`` backend path pulls it in.
"""

from __future__ import annotations

from typing import Any

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .utils import module_device


if is_torch_available():
    import torch
    from torch.export import ExportedProgram

    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


def prepare_for_qnn(model: "PreTrainedModel", sample_inputs: dict[str, Any], config: Any = None):
    """Qualcomm HTP (NPU) inference via the ExecuTorch QNN backend.

    1. **Static-KV-cache wrap.** For decoder-only LMs, wrap the model in
       ``TorchExportableModuleForDecoderOnlyLM`` so the exported graph is decode-capable (KV cache as
       I/O) and the causal mask/positions are real graph inputs — without this a fixed-seq-len export
       folds the mask to constants and QNN can't serialize ("No graph inputs present"). ``sample_inputs``
       is rewritten to ``{input_ids, cache_position}`` to match the wrapper's forward.
    2. **Compiler spec.** ``use_fp16`` is off when ``config.quantize`` is set (PT2E runs later in
       ``quantize_for_qnn``); the SoC (context binaries are SoC-locked) comes from ``config.soc_model``.
    3. The partitioner is necessary but **not sufficient** — QNN lowering is finished by ``_lower_for_qnn``.
    """
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )

    from ..integrations.executorch import TorchExportableModuleForDecoderOnlyLM

    soc_model = getattr(config, "soc_model", "SM8750")
    quantize = getattr(config, "quantize", None)
    static_cache = getattr(config, "static_cache", True)
    max_cache_len = getattr(config, "max_cache_len", 128)

    model.requires_grad_(False)
    device = module_device(model)
    if device is not None and device.type != "cpu":
        model = model.to(device="cpu")

    # Decode-capable graph for decoder-only LMs via a static KV cache.
    if static_cache and isinstance(model, PreTrainedModel) and model.can_generate():
        model.config.use_cache = True
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
            model.generation_config.cache_implementation = "static"
        text_config = model.config.get_text_config()
        if getattr(text_config, "max_cache_len", None) is None:
            text_config.max_cache_len = max_cache_len
        wrapped = TorchExportableModuleForDecoderOnlyLM(model, batch_size=1, max_cache_len=max_cache_len)
        input_ids = sample_inputs["input_ids"]
        sample_inputs = {
            "input_ids": input_ids,
            "cache_position": torch.arange(input_ids.shape[-1], dtype=torch.long),
        }
        model = wrapped
    elif isinstance(model, PreTrainedModel):
        # Prefill-only graph: disable the KV cache so the model returns logits only. With use_cache=True
        # it emits a DynamicCache, which breaks PT2E calibration (NaN observers) and pytree flattening.
        model.config.use_cache = False

    backend_options = generate_htp_compiler_spec(use_fp16=quantize is None)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc_model),
        backend_options=backend_options,
    )
    partitioner = [QnnPartitioner(compiler_specs)]
    return model, sample_inputs, partitioner


def quantize_for_qnn(exported_program: "ExportedProgram", sample_inputs, config) -> "ExportedProgram":
    """QNN PT2E quantization (8a8w / 16a4w / …) on the exported graph.

    QNN quantizes *after* export: ``prepare_pt2e`` annotates the (core-ATen) graph with a
    ``QnnQuantizer``, we calibrate, then ``convert_pt2e`` inserts Q/DQ. Because ``convert_pt2e`` returns
    an nn.Module while lowering consumes an ``ExportedProgram``, we re-export the converted module.
    ``run_decompositions()`` first inlines transformers HOPs (``WrapWithSetGradEnabled``) that PT2E /
    QNN passes can't introspect. No-op when ``config.quantize`` is None (the FP16 path).
    """
    quant_dtype = getattr(config, "quantize", None)
    if quant_dtype is None:
        return exported_program

    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset, QnnExecuTorchBackendType
    from torchao.quantization.pt2e import MovingAverageMinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=getattr(QcomChipset, config.soc_model),
    )
    # Per-channel weights + MovingAverageMinMax activations = the native Qualcomm LLM recipe;
    # the default observer is NaN-sensitive on transformer activations.
    quantizer.set_default_quant_config(
        quant_dtype=getattr(QuantDtype, f"use_{quant_dtype}"),
        is_conv_per_channel=True,
        is_linear_per_channel=True,
        act_observer=MovingAverageMinMaxObserver,
    )

    try:
        module = exported_program.run_decompositions().module()
    except Exception as exc:
        logger.warning("QNN PT2E: run_decompositions() failed (%s); using the module as-is.", type(exc).__name__)
        module = exported_program.module()
    prepared = prepare_pt2e(module, quantizer)

    calibration_inputs = getattr(config, "calibration_inputs", None)
    if not calibration_inputs:
        logger.warning(
            "QNN PT2E: no `calibration_inputs` provided; calibrating with the single `sample_inputs` "
            "batch. This is functional but yields poor %s accuracy — pass representative batches via "
            "ExecutorchConfig(calibration_inputs=...) for production.",
            quant_dtype,
        )
        calibration_inputs = [sample_inputs]
    with torch.no_grad():
        for batch in calibration_inputs:
            prepared(**batch)

    converted = convert_pt2e(prepared)
    args, kwargs = exported_program.example_inputs
    return torch.export.export(converted, args, kwargs=kwargs, strict=False)


def _lower_for_qnn(exported_program, partitioner, compile_config):
    """QNN lowering: run QNN's pre-export pipeline + edge passes inside a ``QnnManagerContext``.

    The exporter's generic ``to_edge_transform_and_lower`` skips ``transform_for_export_pipeline``
    (decompose SDPA, lift constant scalars, canonicalize conv, remove alias_copy) and
    ``get_to_edge_transform_passes`` (FoldQDQ, LayoutTransform, TagQuantIO …) and never enters a
    ``QnnManagerContext`` — so a bare ``QnnPartitioner`` crashes (``KeyError aten.alias_copy.default``,
    rank-3 FullyConnected, "No QnnManager active"). This mirrors ``to_edge_transform_and_lower_to_qnn``
    but operates on the ExportedProgram the exporter already produced (single export, QNN edge config).
    """
    from executorch.backends.qualcomm._passes.qnn_pass_manager import get_qnn_pass_manager_cls
    from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
    from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import QnnManagerContext
    from executorch.backends.qualcomm.utils.utils import qnn_edge_config
    from executorch.exir.program import to_edge_transform_and_lower

    compiler_specs = partitioner[0].compiler_specs_snapshot
    # Decompose to core ATen first: transformers' export can leave HOPs (WrapWithSetGradEnabled,
    # vmap/flat_apply) that QNN's LiftConstantScalarOperands can't introspect. Skip for an already
    # quantized graph (quantize_for_qnn already decomposed; re-decomposing would disturb the Q/DQ).
    already_quantized = any(
        "quantize" in str(getattr(n, "target", "")) for n in exported_program.graph_module.graph.nodes
    )
    if not already_quantized:
        try:
            exported_program = exported_program.run_decompositions()
        except Exception as exc:
            # Static-cache graphs (in-place KV buffer copy_) can trip aot_export inside
            # run_decompositions. Proceed without it; QNN passes handle non-HOP graphs fine.
            logger.warning("QNN: run_decompositions() failed (%s); lowering the graph as-is.", type(exc).__name__)

    pass_manager = get_qnn_pass_manager_cls(QnnExecuTorchBackendType.kHtpBackend)()
    exported_program = pass_manager.transform_for_export_pipeline(exported_program)
    transform_passes = pass_manager.get_to_edge_transform_passes(exported_program, compiler_specs=compiler_specs)
    with QnnManagerContext({"forward": compiler_specs}):
        return to_edge_transform_and_lower(
            exported_program,
            transform_passes=transform_passes,
            partitioner=partitioner,
            compile_config=qnn_edge_config(),
        )
