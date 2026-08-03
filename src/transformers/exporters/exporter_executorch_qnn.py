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
"""QNN (Qualcomm AI Engine Direct / HTP) backend for the ExecuTorch exporter.

A QNN decoder-LLM export is a multi-graph, cross-graph-quantized pipeline (static_llama surgery +
fixed-shape CALIBRATE/PREFILL/DECODE graphs + 16a4w PT2E + encoding-override + per-KV requant). That
"wet clay" work is owned by ExecuTorch's Qualcomm backend and reused here verbatim, via its
return-not-write seam: ``export_llama(args, build_only=True)`` runs the surgery/quant/encoding-override
and returns the converted deploy graphs as a ``QnnLLMGraphSet`` (before any lowering/serialize).

``prepare_for_qnn`` turns that graph-set into a ``BackendExportPlan``; the generic exporter then
traces each graph with plain ``torch.export`` (no HF fx-fixes — q/dq already present) and ``qnn_lower``
runs the QNN transform + the generic ``to_edge_transform_and_lower`` (inside ``QnnManagerContext``).
So the ``.pte`` is emitted by the HF exporter and is structurally identical to native ``export_llama``.

Imported lazily so the exporter still works without the Qualcomm SDK/package.
"""

from __future__ import annotations

from collections import OrderedDict

from executorch.backends.qualcomm._passes.qnn_pass_manager import get_qnn_pass_manager_cls
from executorch.backends.qualcomm.llm_export import build_qnn_llm_graphset
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils.utils import (
    QnnManagerContext,
    flatbuffer_to_option,
    generate_qnn_executorch_option,
    qnn_edge_config,
)
from executorch.exir import to_edge_transform_and_lower

from ..utils import logging
from .exporter_executorch import BackendExportPlan, GraphSpec


logger = logging.get_logger(__name__)


def prepare_for_qnn(model, sample_inputs, config):
    """Build the QNN ``BackendExportPlan`` from ExecuTorch's static_llama seam.

    ① (surgery + 16a4w PT2E + encoding-override) is reused verbatim through the stable
    ``executorch.backends.qualcomm.llm_export.build_qnn_llm_graphset`` API — no dependency on
    ``executorch.examples``. It is model-driven (the HF model supplies the architecture) and returns
    the converted deploy graphs; ②③④ (trace + QNN transform + generic to_edge) run in the HF
    exporter / ``qnn_lower``.
    """
    options = config.backend_options or {}
    logger.info("QNN prepare: building deploy graph-set via backends.qualcomm.build_qnn_llm_graphset")
    gs = build_qnn_llm_graphset(
        model,
        decoder_model=options.get("decoder_model"),
        soc_model=options.get("soc_model"),
        model_mode=options.get("model_mode", "kv"),
        prompt=options.get("prompt"),
        calib_samples=options.get("calib_samples"),
        artifact_dir=options.get("artifact_dir"),
    )  # QnnLLMGraphSet — surgery+quant+encoding-override done

    method_set: "OrderedDict[str, GraphSpec]" = OrderedDict()
    for g in gs.modules:
        method_set[g] = GraphSpec(
            module=gs.modules[g], sample_inputs=gs.inputs[g], dynamic_shapes=None, role="deploy", mode="plain"
        )

    def qnn_lower(programs, cfg):
        # ③④ — per-graph transform_for_export_pipeline + get_to_edge_transform_passes, then one
        # generic to_edge_transform_and_lower inside QnnManagerContext (native lines 456-484).
        aten_programs = {}
        transform_passes = {}
        qnn_partitioners = {
            g: [QnnPartitioner(gs.compiler_specs[g], skip_node_op_set=gs.skip_node_op_set)]
            for g in programs
        }
        for g, ep in programs.items():
            option = generate_qnn_executorch_option(gs.compiler_specs[g])
            backend_type = flatbuffer_to_option(option).backend_options.backend_type
            pass_manager = get_qnn_pass_manager_cls(backend_type)()
            aten_programs[g] = pass_manager.transform_for_export_pipeline(ep)
            transform_passes[g] = pass_manager.get_to_edge_transform_passes(
                ep, passes_job=gs.passes_job[g], dep_table=gs.dep_table[g],
                compiler_specs=gs.compiler_specs[g], skip_node_op_set=gs.skip_node_op_set,
            )
        with QnnManagerContext(gs.compiler_specs):
            return to_edge_transform_and_lower(
                aten_programs, transform_passes=transform_passes, partitioner=qnn_partitioners,
                constant_methods=gs.constant_methods or None, compile_config=qnn_edge_config(),
            )

    return BackendExportPlan(
        method_set=method_set, partitioner=None, constant_methods=gs.constant_methods,
        lower=qnn_lower, backend_config=gs.backend_config, lowering_context=None,
    )
