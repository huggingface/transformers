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

Each backend quantizes every architecture family (dense / MoE / SSM) with the quantizers it supports: a
`pt2e_quantizer` on the traced graph (x86 for dynamo/ONNX/OpenVINO, XNNPACK for ExecuTorch), and the converted
model's own toolchain on ONNX (`onnxruntime_quantizer`) and OpenVINO (`nncf_quantizer`). Calibration and
per-component recipes go through `export_for_generation`, whose components take the attention mask as an input
(PT2E's retrace trips on in-graph mask construction).
"""

import copy
import unittest
from unittest.mock import patch

import pytest
from parameterized import parameterized

from tests.exporters.test_export import _run_onnx_program, _run_openvino_model, disable_hub_kernels
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.utils import capture_calibration_inputs, decompose_for_generation
from transformers.testing_utils import (
    require_executorch,
    require_nncf,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch,
    require_torchao,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


MAX_CACHE_LEN = 16


def _has_quantize_ops(exported) -> bool:
    """The exported FX graph carries PT2E quantize/dequantize nodes."""
    return any(n.op == "call_function" and "quantize" in str(n.target) for n in exported.graph.nodes)


def _has_dynamic_quant_ops(exported) -> bool:
    """Activations are quantized dynamically (runtime `choose_qparams`), not with static calibrated scales."""
    return any(n.op == "call_function" and "choose_qparams" in str(n.target) for n in exported.graph.nodes)


def _openvino_op_counts(ov_model) -> tuple[int, int]:
    """`(FakeQuantize nodes, low-precision weight constants)` in an OpenVINO model."""
    import openvino

    low_precision = (openvino.Type.i8, openvino.Type.u8, openvino.Type.i4, openvino.Type.u4)
    ops = list(ov_model.get_ops())
    fake_quantize = sum(op.get_type_name() == "FakeQuantize" for op in ops)
    low = sum(op.get_type_name() == "Constant" and op.get_output_element_type(0) in low_precision for op in ops)
    return fake_quantize, low


def _has_onnx_quantize_ops(program) -> bool:
    """The exported ONNX graph carries QDQ (`QuantizeLinear`) nodes."""
    return any(node.op_type == "QuantizeLinear" for node in program.model_proto.graph.node)


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

    def _moe_model(self):
        """Tiny MoE — exercises expert-routing / expert-linear quantization."""
        from transformers import MixtralConfig, MixtralForCausalLM

        torch.manual_seed(0)
        config = MixtralConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
            num_local_experts=2,
            num_experts_per_tok=2,
        )
        return MixtralForCausalLM(config).eval()

    def _ssm_model(self):
        """Tiny SSM — exercises conv1d / SSM-projection quantization (no attention)."""
        from transformers import Mamba2Config, Mamba2ForCausalLM

        torch.manual_seed(0)
        config = Mamba2Config(
            hidden_size=32,
            num_hidden_layers=2,
            vocab_size=64,
            num_heads=8,
            head_dim=8,
            state_size=8,
            n_groups=1,
            chunk_size=8,
            conv_kernel=4,
            expand=2,
        )
        return Mamba2ForCausalLM(config).eval()

    def _vlm_model(self):
        """Tiny VLM (CLIP vision + Llama text) and its image+text inputs."""
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

        torch.manual_seed(0)
        vision_config = CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
            num_channels=3,
        )
        text_config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
        )
        model = LlavaForConditionalGeneration(
            LlavaConfig(vision_config=vision_config, text_config=text_config, image_token_index=1)
        ).eval()
        # 4 image tokens = (image_size / patch_size) ** 2, then 2 text tokens
        inputs = {
            "input_ids": torch.tensor([[1, 1, 1, 1, 5, 6]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 32, 32),
        }
        return model, inputs

    def _generation_config(self):
        return GenerationConfig(cache_implementation="static", max_cache_len=MAX_CACHE_LEN, do_sample=False)

    def _decode_component(self, model=None):
        """Multi-token `decode` component against a fixed-size `StaticCache`. The mask is a precomputed
        input, so PT2E has no in-graph mask construction to trip on."""
        model = model if model is not None else self._tiny_model()
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        return decompose_for_generation(
            model, inputs, generation_config=self._generation_config(), multi_token_decode=True
        )["decode"]

    def _quantizer(self, name, dynamic=False):
        """The PT2E quantizer named `name`: `x86` (torchao, static per-channel int8, or dynamic int8 with `dynamic`)
        or `xnnpack` (static per-tensor, for ExecuTorch)."""
        if name == "x86":
            from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
                X86InductorQuantizer,
                get_default_x86_inductor_quantization_config,
            )

            return X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_dynamic=dynamic))
        if name == "xnnpack":
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )

            return XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        raise ValueError(f"unknown quantizer {name}")

    def _quantization(self, quantizer, inputs):
        """Config kwargs for the PT2E quantizer named `quantizer`, calibrated on `inputs`."""
        return {"pt2e_quantizer": self._quantizer(quantizer), "calibration_dataset": [copy.deepcopy(inputs)]}

    def _quantization_target(self, family):
        """The `(model, inputs)` to quantize for `family`, picked so the traced forward builds no attention
        mask in-graph — PT2E's `make_fx` retrace trips on in-graph mask construction:

        - dense / MoE: the multi-token `decode` component, whose attention mask is a precomputed input;
        - SSM: the plain forward — a state-space model has no attention mask to build.
        """
        if family == "dense":
            return self._decode_component(self._tiny_model())
        if family == "moe":
            return self._decode_component(self._moe_model())
        if family == "ssm":
            return self._ssm_model(), {"input_ids": torch.randint(0, 64, (1, 8))}
        raise ValueError(f"unknown family {family}")

    # ──────────────────────────────── Dynamo ────────────────────────────────

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_calibration_defaults_to_sample_inputs_with_warning(self):
        """With no `calibration_dataset`, calibration falls back to a single pass on the sample inputs and
        warns (one sample can hurt accuracy) — quantization still applies."""
        from transformers.exporters import DynamoConfig, DynamoExporter, exporter_dynamo

        decode_model, decode_inputs = self._decode_component()
        with patch.object(exporter_dynamo.logger, "warning_once") as warning_once:
            exported = DynamoExporter().export(
                decode_model,
                copy.deepcopy(decode_inputs),
                DynamoConfig(dynamic=False, pt2e_quantizer=self._quantizer("x86")),
            )
        messages = [call.args[0] for call in warning_once.call_args_list]
        self.assertTrue(any("calibration_dataset" in message for message in messages), messages)
        self.assertTrue(_has_quantize_ops(exported))

    def test_config_takes_one_quantizer(self):
        """A PT2E quantizer and the backend's own quantize at different stages; a config takes one of them."""
        from transformers.exporters import OnnxConfig, OpenVINOConfig

        def quantizer(model, dataset):
            return model

        with self.assertRaisesRegex(ValueError, "at most one"):
            OnnxConfig(pt2e_quantizer=object(), onnxruntime_quantizer=quantizer)
        with self.assertRaisesRegex(ValueError, "at most one"):
            OpenVINOConfig(pt2e_quantizer=object(), nncf_quantizer=quantizer)

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_calibration_dataset_captured_per_component(self):
        """A generate-level `calibration_dataset` becomes one calibration set per component; a multi-token decode
        graph also serves prefill, so its set carries each sample's prefill inputs too."""
        model = self._tiny_model()
        calibration = [
            {"input_ids": torch.randint(0, 64, (1, n)), "attention_mask": torch.ones(1, n, dtype=torch.long)}
            for n in (3, 4, 5)
        ]
        captured = capture_calibration_inputs(
            model, copy.deepcopy(calibration), generation_config=self._generation_config(), multi_token_decode=True
        )
        self.assertEqual(set(captured), {"prefill", "decode"})
        self.assertEqual(len(captured["prefill"]), len(calibration))
        self.assertEqual(len(captured["decode"]), 2 * len(calibration))

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_export_for_generation_fans_out_calibration(self):
        """`export_for_generation` hands each component's export its own captured calibration set in place of
        the single config's generate-level `calibration_dataset`."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        calibration = [
            {"input_ids": torch.randint(0, 64, (1, n)), "attention_mask": torch.ones(1, n, dtype=torch.long)}
            for n in (3, 4, 5)
        ]
        config = DynamoConfig(pt2e_quantizer=self._quantizer("x86"), calibration_dataset=calibration)
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        with patch.object(DynamoExporter, "export", autospec=True) as export:
            DynamoExporter().export_for_generation(
                self._tiny_model(),
                inputs,
                config,
                generation_config=self._generation_config(),
                multi_token_decode=True,
            )
        received = sorted(len(call.kwargs["config"].calibration_dataset) for call in export.call_args_list)
        self.assertEqual(received, [len(calibration), 2 * len(calibration)])

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_quantized_dynamo(self, family):
        """Every family's graph gains PT2E quantize ops with the x86 quantizer."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._quantization_target(family)
        exported = DynamoExporter().export(
            model, copy.deepcopy(inputs), DynamoConfig(dynamic=False, **self._quantization("x86", inputs))
        )
        self.assertTrue(_has_quantize_ops(exported))

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_vlm_per_component_quantization(self):
        """A VLM is quantized component by component, each with its own recipe, via a `{component: config}`
        dict on `export_for_generation` (multi-token decode): static int8 on the prompt's `language_model`,
        the lighter dynamic int8 on `decode`, and `lm_head` left in fp32."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._vlm_model()
        config = {
            "language_model": DynamoConfig(dynamic=True, pt2e_quantizer=self._quantizer("x86", dynamic=False)),
            "decode": DynamoConfig(dynamic=True, pt2e_quantizer=self._quantizer("x86", dynamic=True)),
            "lm_head": DynamoConfig(dynamic=True),
        }
        components = DynamoExporter().export_for_generation(model, inputs, config, multi_token_decode=True)

        self.assertTrue(_has_quantize_ops(components["language_model"]), "language_model should be quantized")
        self.assertTrue(_has_quantize_ops(components["decode"]), "decode should be quantized")
        self.assertFalse(_has_quantize_ops(components["lm_head"]), "lm_head should stay fp32")
        # the recipes really differ: decode is dynamically quantized, language_model statically
        self.assertTrue(_has_dynamic_quant_ops(components["decode"]), "decode should be dynamically quantized")
        self.assertFalse(_has_dynamic_quant_ops(components["language_model"]), "language_model should be static")

    # ──────────────────────────────── ONNX ──────────────────────────────────

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_quantized_onnx(self, family):
        """The x86 quantizer lowered to ONNX: every family gets a QDQ graph that runs in ONNX Runtime."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        model, inputs = self._quantization_target(family)
        program = OnnxExporter().export(
            model,
            copy.deepcopy(inputs),
            OnnxConfig(dynamic=False, external_data=False, **self._quantization("x86", inputs)),
        )
        self.assertTrue(_has_onnx_quantize_ops(program))
        # quantization error rules out an eager-parity check, so check it runs to finite outputs
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))

    @parameterized.expand([(family, dynamic) for family in ("dense", "moe", "ssm") for dynamic in (False, True)])
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_onnxruntime_quantized_onnx(self, family, dynamic):
        """ONNX Runtime quantizes the converted model itself (an `onnxruntime_quantizer`),
        statically calibrated on the model's inputs or dynamically: every family gains quantize nodes and still runs
        in ONNX Runtime."""
        from transformers.exporters import OnnxConfig, OnnxExporter, OnnxRuntimeQuantizer

        model, inputs = self._quantization_target(family)
        program = OnnxExporter().export(
            model,
            copy.deepcopy(inputs),
            OnnxConfig(
                dynamic=False,
                external_data=False,
                onnxruntime_quantizer=OnnxRuntimeQuantizer(dynamic=dynamic),
                calibration_dataset=[copy.deepcopy(inputs)],
            ),
        )
        quantize_op = "DynamicQuantizeLinear" if dynamic else "QuantizeLinear"
        self.assertTrue(any(node.op_type == quantize_op for node in program.model_proto.graph.node))
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))

    # ─────────────────────────────── OpenVINO ───────────────────────────────

    def _assert_openvino_quantized_model_runs(self, ov_model, model, inputs):
        """The quantized OpenVINO model runs to finite logits, close to the fp32 model's.

        With quantized activations (`FakeQuantize` nodes) the matmuls run in int8, whose accuracy needs a CPU with
        VNNI: without it (AVX2-only Intel, AMD Zen 2 and older) their int16 accumulators saturate, so only finiteness
        is checked.
        """
        outputs = _run_openvino_model(ov_model, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))
        int8_activations = _openvino_op_counts(ov_model)[0] > 0
        if int8_activations and not torch.cpu._is_vnni_supported():
            return
        with torch.no_grad():
            expected = model(**copy.deepcopy(inputs)).logits
        # Relative to the logits' scale: weight-only int8 stays within ~2% on these models, int8 activations ~10%.
        tolerance = (0.1 if int8_activations else 0.05) * expected.abs().max().item()
        torch.testing.assert_close(outputs["logits"].to(expected.dtype), expected, atol=tolerance, rtol=0)

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_openvino
    @require_nncf
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_quantized_openvino(self, family):
        """NNCF quantizes the converted IR itself (an `nncf_quantizer`): every family gains `FakeQuantize` nodes
        and int8 weights. Static export. Mamba2's depthwise conv (`GroupConvolution`) is left out: NNCF quantizes it
        along a channel axis the CPU plugin's `FakeQuantize` rejects."""
        import nncf

        from transformers.exporters import NNCFQuantizer, OpenVINOConfig, OpenVINOExporter

        model, inputs = self._quantization_target(family)
        quantizer = NNCFQuantizer(
            subset_size=1, ignored_scope=nncf.IgnoredScope(types=["GroupConvolution"], validate=False)
        )
        ov_model = OpenVINOExporter().export(
            model,
            copy.deepcopy(inputs),
            OpenVINOConfig(dynamic=False, nncf_quantizer=quantizer, calibration_dataset=[copy.deepcopy(inputs)]),
        )
        fake_quantize, low_precision = _openvino_op_counts(ov_model)
        self.assertGreater(fake_quantize, 0)
        self.assertGreater(low_precision, 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_openvino
    @require_nncf
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_openvino_weight_compression(self, family):
        """NNCF compresses the converted IR's weights to int8 (`NNCFQuantizer(weights_only=True)`), leaving activations
        alone, so the matmuls stay in floating point and accuracy is checked on any CPU."""
        import nncf

        from transformers.exporters import NNCFQuantizer, OpenVINOConfig, OpenVINOExporter

        model, inputs = self._quantization_target(family)
        quantizer = NNCFQuantizer(weights_only=True, mode=nncf.CompressWeightsMode.INT8_SYM)
        ov_model = OpenVINOExporter().export(
            model,
            copy.deepcopy(inputs),
            OpenVINOConfig(dynamic=False, nncf_quantizer=quantizer),
        )
        fake_quantize, low_precision = _openvino_op_counts(ov_model)
        self.assertEqual(fake_quantize, 0)
        self.assertGreater(low_precision, 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_openvino
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_pt2e_quantizer_on_openvino(self, family):
        """A `pt2e_quantizer` works on OpenVINO too: its weights stay unfolded behind quantize/dequantize pairs, which
        OpenVINO converts to `FakeQuantize` and compresses itself."""
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        model, inputs = self._quantization_target(family)
        ov_model = OpenVINOExporter().export(
            model,
            copy.deepcopy(inputs),
            OpenVINOConfig(dynamic=False, **self._quantization("x86", inputs)),
        )
        self.assertGreater(_openvino_op_counts(ov_model)[0], 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)

    # ────────────────────────────── ExecuTorch ──────────────────────────────

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_executorch
    @pytest.mark.executorch_export_test
    @disable_hub_kernels
    def test_quantized_executorch(self, family):
        """The same `pt2e_quantizer` recipe, lowered to an ExecuTorch `.pte`: every family's graph is quantized
        and lowers to a program. The x86 quantizer is absent — its per-channel q/dq ops have no out variant, so
        they stay undelegated and fail `to_executorch`; XNNPACK wants its per-tensor quantizer instead."""
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter
        from transformers.exporters.exporter_dynamo import DynamoExporter

        # lowering hides the quantize ops inside the XNNPACK delegate, so check the graph `_quantize` returns
        quantized = []
        quantize = DynamoExporter._quantize

        def record_quantized(exporter, *args, **kwargs):
            quantized.append(quantize(exporter, *args, **kwargs))
            return quantized[-1]

        model, inputs = self._quantization_target(family)
        with patch.object(DynamoExporter, "_quantize", record_quantized):
            program = ExecutorchExporter().export(
                model,
                copy.deepcopy(inputs),
                ExecutorchConfig(backend="xnnpack", dynamic=False, **self._quantization("xnnpack", inputs)),
            )
        self.assertIsNotNone(program)
        self.assertTrue(_has_quantize_ops(quantized[0]))
        # Not executed: running a quantized `.pte` in-process SIGABRTs under the pytest-rerunfailures plugin thread.
