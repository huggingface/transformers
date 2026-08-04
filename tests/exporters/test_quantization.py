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
"""Post-training quantization export tests.

PT2E quantization is a backend-agnostic recipe living in the shared Dynamo layer: pass a `quantizer`
(any PT2E `Quantizer` — `XNNPACKQuantizer`, `X86InductorQuantizer`, a vendor `QnnQuantizer`, …) on the
export config and the graph is quantized (`prepare_pt2e` → calibrate → `convert_pt2e`) before it's
returned/lowered — no hardcoded schemes. These cases exercise it across backends:

- **Dynamo** — the quantized `ExportedProgram` carries `quantize`/`dequantize` ops (inductor int8);
- **ONNX** — the ops translate to QDQ (`QuantizeLinear`/`DequantizeLinear`), incl. per-channel, and load in ORT;
- **ExecuTorch** — the `.pte` shrinks vs fp32; `export_for_generation` calibrates each component on its own
  captured set (one generate-level `calibration_dataset` fanned out via the decomposition capture);
- **calibration** — a missing `calibration_dataset` falls back to a single pass on the sample inputs, with a warning.

Quantization runs on the decomposed generation components (whose attention mask is a precomputed input),
avoiding the in-graph mask construction that trips PT2E on a full-model forward.
"""

import copy
import sys
import unittest
from unittest.mock import patch

import pytest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.utils import capture_calibration_inputs, decompose_for_generation
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_torch,
    require_torchao,
    run_command,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


MAX_CACHE_LEN = 16


def _qnn_available() -> bool:
    """The QNN backend needs the Qualcomm AI Engine Direct SDK. Probe in a subprocess: importing
    `executorch.backends.qualcomm` runs an auto-installer that mutates `LD_LIBRARY_PATH`, which would
    corrupt the pytest process for the other tests."""
    try:
        run_command([sys.executable, "-c", "import executorch.backends.qualcomm"])
        return True
    except Exception:
        return False


@slow
@require_torch
@require_torchao
class QuantizationExportTest(unittest.TestCase):
    def _tiny_model(self):
        torch.manual_seed(0)
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
        )
        return LlamaForCausalLM(config).eval()

    def _generation_config(self):
        return GenerationConfig(cache_implementation="static", max_cache_len=MAX_CACHE_LEN, do_sample=False)

    def _decode_component(self):
        """Multi-token `decode` component against a fixed-size `StaticCache`. The mask is a precomputed
        input, so PT2E has no in-graph mask construction to trip on."""
        model = self._tiny_model()
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        return decompose_for_generation(
            model, inputs, generation_config=self._generation_config(), multi_token_decode=True
        )["decode"]

    def _x86_quantizer(self):
        """torchao-native per-channel quantizer (no ExecuTorch dependency)."""
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )

        return X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())

    def _xnnpack_quantizer(self):
        """Per-tensor quantizer for the XNNPACK ExecuTorch backend."""
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        return XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    # ──────────────────────────────── Dynamo ────────────────────────────────

    @pytest.mark.torch_export_test
    def test_dynamo_quantized_program(self):
        """`DynamoExporter` with a `quantizer` returns a quantized `ExportedProgram` (quantize/dequantize
        ops folded in) — the backend-agnostic base every other exporter builds on."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        decode_model, decode_inputs = self._decode_component()
        exported = DynamoExporter().export(
            decode_model,
            copy.deepcopy(decode_inputs),
            DynamoConfig(
                dynamic=True, quantizer=self._x86_quantizer(), calibration_dataset=[copy.deepcopy(decode_inputs)]
            ),
        )
        quant_nodes = [
            node for node in exported.graph.nodes if node.op == "call_function" and "quantize" in str(node.target)
        ]
        self.assertGreater(len(quant_nodes), 0)

    @pytest.mark.torch_export_test
    def test_calibration_defaults_to_sample_inputs_with_warning(self):
        """With no `calibration_dataset`, calibration falls back to a single pass on the sample inputs and
        warns (one sample can hurt accuracy) — quantization still applies."""
        from transformers.exporters import DynamoConfig, DynamoExporter, exporter_dynamo

        decode_model, decode_inputs = self._decode_component()
        with patch.object(exporter_dynamo.logger, "warning_once") as warning_once:
            exported = DynamoExporter().export(
                decode_model, copy.deepcopy(decode_inputs), DynamoConfig(dynamic=True, quantizer=self._x86_quantizer())
            )
        warning_once.assert_called()
        self.assertTrue(
            any(node.op == "call_function" and "quantize" in str(node.target) for node in exported.graph.nodes)
        )

    # ───────────────────────────────── ONNX ─────────────────────────────────

    @require_onnxruntime
    @require_onnxscript
    @pytest.mark.onnx_export_test
    def test_onnx_quantized_qdq(self):
        """`OnnxExporter` translates the PT2E quantize/dequantize ops to ONNX QDQ
        (`QuantizeLinear`/`DequantizeLinear`) — including the per-channel dequant translation onnxscript
        lacks — and the model loads in ONNX Runtime."""
        from collections import Counter

        import onnxruntime as ort

        from transformers.exporters import OnnxConfig, OnnxExporter

        decode_model, decode_inputs = self._decode_component()
        program = OnnxExporter().export(
            decode_model,
            copy.deepcopy(decode_inputs),
            OnnxConfig(
                dynamic=True,
                quantizer=self._x86_quantizer(),
                calibration_dataset=[copy.deepcopy(decode_inputs)],
                external_data=False,
            ),
        )
        op_types = Counter(node.op_type for node in program.model_proto.graph.node)
        self.assertGreater(op_types["QuantizeLinear"], 0)
        self.assertGreater(op_types["DequantizeLinear"], 0)
        ort.InferenceSession(program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"])

    # ────────────────────────────── ExecuTorch ──────────────────────────────

    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch_quantized_pte_smaller(self):
        """`ExecutorchExporter` with a `quantizer` lowers to a materially smaller int8 `.pte` than fp32."""
        import os
        import tempfile

        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        decode_model, decode_inputs = self._decode_component()
        fp32 = ExecutorchExporter().export(
            decode_model, copy.deepcopy(decode_inputs), ExecutorchConfig(backend="xnnpack", dynamic=True)
        )
        int8 = ExecutorchExporter().export(
            decode_model,
            copy.deepcopy(decode_inputs),
            ExecutorchConfig(
                backend="xnnpack",
                dynamic=True,
                quantizer=self._xnnpack_quantizer(),
                calibration_dataset=[copy.deepcopy(decode_inputs)],
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp32.save(os.path.join(tmp, "fp32.pte"))
            int8.save(os.path.join(tmp, "int8.pte"))
            self.assertLess(
                os.path.getsize(os.path.join(tmp, "int8.pte")), os.path.getsize(os.path.join(tmp, "fp32.pte"))
            )

    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch_export_for_generation_per_component_calibration(self):
        """`export_for_generation` with a `quantizer`: the single config's generate-level
        `calibration_dataset` is fanned out (via the decomposition capture) into a per-component
        calibration set, and every component quantizes smaller than fp32."""
        import os
        import tempfile

        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        model = self._tiny_model()
        gen_config = self._generation_config()
        # generate-style calibration set with varied prompt lengths (needs a dynamic export)
        calibration = [
            {"input_ids": torch.randint(0, 64, (1, n)), "attention_mask": torch.ones(1, n, dtype=torch.long)}
            for n in (3, 4, 5)
        ]

        # the capture fans one generate-level dataset out to a per-component calibration set
        captured = capture_calibration_inputs(
            model, copy.deepcopy(calibration), generation_config=gen_config, multi_token_decode=True
        )
        self.assertEqual(set(captured), {"prefill", "decode"})
        self.assertTrue(all(len(inputs) == len(calibration) for inputs in captured.values()))

        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        quantized = ExecutorchExporter().export_for_generation(
            model,
            copy.deepcopy(inputs),
            ExecutorchConfig(
                backend="xnnpack",
                dynamic=True,
                quantizer=self._xnnpack_quantizer(),
                calibration_dataset=copy.deepcopy(calibration),
            ),
            generation_config=gen_config,
            multi_token_decode=True,
        )
        fp32 = ExecutorchExporter().export_for_generation(
            model,
            copy.deepcopy(inputs),
            ExecutorchConfig(backend="xnnpack", dynamic=True),
            generation_config=gen_config,
            multi_token_decode=True,
        )
        self.assertEqual(set(quantized), {"prefill", "decode"})
        for name in quantized:
            with tempfile.TemporaryDirectory() as tmp:
                quantized[name].save(os.path.join(tmp, "q.pte"))
                fp32[name].save(os.path.join(tmp, "f.pte"))
                self.assertLess(
                    os.path.getsize(os.path.join(tmp, "q.pte")), os.path.getsize(os.path.join(tmp, "f.pte"))
                )

    # ──────────────────────────── QNN (Qualcomm HTP) ────────────────────────────

    @unittest.skipUnless(_qnn_available(), "requires the Qualcomm AI Engine Direct (QNN) SDK")
    @require_executorch
    @pytest.mark.executorch_export_test
    def test_qnn_export(self):
        """QNN (HTP) int8 export of the decode component with a `QnnQuantizer`, via the generic
        `config.quantizer` recipe. Skipped unless the Qualcomm QNN SDK is installed."""
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        decode_model, decode_inputs = self._decode_component()
        program = ExecutorchExporter().export(
            decode_model,
            copy.deepcopy(decode_inputs),
            ExecutorchConfig(
                backend="qnn",
                dynamic=False,
                soc_model="SM8650",
                quantizer=QnnQuantizer(),
                calibration_dataset=[copy.deepcopy(decode_inputs)],
            ),
        )
        self.assertIsNotNone(program)
