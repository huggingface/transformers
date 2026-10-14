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

Each backend quantizes every architecture family (dense / MoE / SSM) with the quantizers it supports:
`PT2EQuantizer` and `NNCFTorchFXQuantizer` on the FX graph (x86 for dynamo/ONNX/OpenVINO, XNNPACK for ExecuTorch),
and the converted model's own toolchain on ONNX (`OnnxRuntimeQuantizer`, `NNCFOnnxQuantizer`) and OpenVINO
(`NNCFOpenVINOQuantizer`). Calibration and per-component recipes go through `export_for_generation`, whose components
take the attention mask as an input (PT2E's retrace trips on in-graph mask construction).
"""

import copy
import unittest
from unittest.mock import patch

import pytest
from parameterized import parameterized

from tests.exporters.test_export import (
    _run_executorch_program,
    _run_onnx_program,
    _run_openvino_model,
    disable_hub_kernels,
)
from tests.test_memory_cleanup_mixin import MemoryCleanupMixin
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.quantizers.utils import capture_calibration_inputs
from transformers.exporters.utils import decompose_for_generation
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


class QuantizationTesterMixin(MemoryCleanupMixin):
    """Model builders and quantizer helpers shared by the per-backend quantization tests."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _dense_model(self, **config_overrides):
        config = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "max_position_embeddings": 128,
        }
        return LlamaForCausalLM(LlamaConfig(**{**config, **config_overrides})).eval()

    def _moe_model(self):
        """Tiny MoE — exercises expert-routing / expert-linear quantization."""
        from transformers import MixtralConfig, MixtralForCausalLM

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
        model = model if model is not None else self._dense_model()
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        return decompose_for_generation(
            model, inputs, generation_config=self._generation_config(), multi_token_decode=True
        )["decode"]

    def _pt2e_quantizer(self, name, calibration_dataset=None, dynamic=False):
        """A `PT2EQuantizer` over the PT2E quantizer named `name`: `x86` (torchao, static per-channel int8, or dynamic
        int8 with `dynamic`) or `xnnpack` (ExecuTorch, static per-tensor)."""
        from transformers.exporters import PT2EQuantizer

        if name == "x86":
            from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
                X86InductorQuantizer,
                get_default_x86_inductor_quantization_config,
            )

            quantizer = X86InductorQuantizer().set_global(
                get_default_x86_inductor_quantization_config(is_dynamic=dynamic)
            )
        elif name == "xnnpack":
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )

            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        else:
            raise ValueError(f"unknown PT2E quantizer {name}")
        return PT2EQuantizer(quantizer, calibration_dataset=calibration_dataset)

    def _nncf_quantizer(self, inputs, weights_only, quantizer_class=None):
        """An NNCF quantizer (`NNCFTorchFXQuantizer` by default), int8 weight-only or fully quantized and calibrated on
        `inputs`."""
        import nncf

        from transformers.exporters import NNCFQuantizationConfig, NNCFTorchFXQuantizer, NNCFWeightCompressionConfig

        quantizer_class = quantizer_class or NNCFTorchFXQuantizer
        if weights_only:
            return quantizer_class(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT8_SYM))
        return quantizer_class(NNCFQuantizationConfig(subset_size=1), calibration_dataset=[copy.deepcopy(inputs)])

    def _quantization_target(self, family):
        """The `(model, inputs)` to quantize for `family`, picked so the traced forward builds no attention
        mask in-graph — PT2E's `make_fx` retrace trips on in-graph mask construction:

        - dense / MoE: the multi-token `decode` component, whose attention mask is a precomputed input;
        - SSM: the plain forward — a state-space model has no attention mask to build.
        """
        if family == "dense":
            return self._decode_component(self._dense_model())
        if family == "moe":
            return self._decode_component(self._moe_model())
        if family == "ssm":
            return self._ssm_model(), {"input_ids": torch.randint(0, 64, (1, 8))}
        raise ValueError(f"unknown family {family}")


@slow
@require_torch
@require_torchao
class DynamoQuantizationTest(QuantizationTesterMixin, unittest.TestCase):
    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_calibration_defaults_to_sample_inputs_with_warning(self):
        """With no `calibration_dataset`, calibration falls back to a single pass on the sample inputs and
        warns (one sample can hurt accuracy) — quantization still applies."""
        from transformers.exporters import DynamoConfig, DynamoExporter
        from transformers.exporters.quantizers import base

        decode_model, decode_inputs = self._decode_component()
        with patch.object(base.logger, "warning_once") as warning_once:
            exported = DynamoExporter().export(
                decode_model,
                copy.deepcopy(decode_inputs),
                DynamoConfig(dynamic=False, quantizer=self._pt2e_quantizer("x86")),
            )
        messages = [call.args[0] for call in warning_once.call_args_list]
        self.assertTrue(any("calibration_dataset" in message for message in messages), messages)
        self.assertTrue(_has_quantize_ops(exported))

    def test_quantizer_checks_export_format(self):
        """A quantizer refuses an export format it doesn't support before anything is traced."""
        from transformers.exporters import DynamoConfig, DynamoExporter, NNCFOpenVINOQuantizer, OnnxRuntimeQuantizer

        decode_model, decode_inputs = self._decode_component()
        for quantizer in (NNCFOpenVINOQuantizer(), OnnxRuntimeQuantizer()):
            with self.assertRaisesRegex(ValueError, "not dynamo"):
                DynamoExporter().export(decode_model, decode_inputs, DynamoConfig(quantizer=quantizer))

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_calibration_sample_missing_traced_inputs(self):
        """A calibration sample without the exported forward's inputs (here generate kwargs handed to a single
        component's quantizer) fails with a message naming the missing inputs."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        decode_model, decode_inputs = self._decode_component()
        generate_kwargs = {"input_ids": decode_inputs["input_ids"], "attention_mask": decode_inputs["attention_mask"]}
        quantizer = self._pt2e_quantizer("x86", [generate_kwargs])
        with self.assertRaisesRegex(ValueError, "lacks the traced inputs .*past_key_values"):
            DynamoExporter().export(decode_model, copy.deepcopy(decode_inputs), DynamoConfig(quantizer=quantizer))

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_fx_quantizer_on_dynamo(self):
        """An FX-stage quantizer receives the Dynamo export's FX graph."""
        from transformers.exporters import DynamoConfig, DynamoExporter, ExportFormat
        from transformers.exporters.quantizers import ExportQuantizer

        class RecordingQuantizer(ExportQuantizer):
            supported_formats = (ExportFormat.DYNAMO,)
            quantizes_fx_graph = True
            models = []

            def quantize(self, model, sample_inputs, export_format):
                self.models.append(model)
                return model

        decode_model, decode_inputs = self._decode_component()
        DynamoExporter().export(
            decode_model, copy.deepcopy(decode_inputs), DynamoConfig(quantizer=RecordingQuantizer())
        )
        self.assertEqual(len(RecordingQuantizer.models), 1)
        self.assertIsInstance(RecordingQuantizer.models[0], torch.fx.GraphModule)

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_calibration_dataset_captured_per_component(self):
        """A generate-level `calibration_dataset` becomes one calibration set per component; a multi-token decode
        graph also serves prefill, so its set carries each sample's prefill inputs too."""
        model = self._dense_model()
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
        """`export_for_generation` hands each component's export a copy of the quantizer holding that component's
        captured calibration set in place of the generate-level `calibration_dataset`."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        calibration = [
            {"input_ids": torch.randint(0, 64, (1, n)), "attention_mask": torch.ones(1, n, dtype=torch.long)}
            for n in (3, 4, 5)
        ]
        config = DynamoConfig(quantizer=self._pt2e_quantizer("x86", calibration))
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        with patch.object(DynamoExporter, "export", autospec=True) as export:
            DynamoExporter().export_for_generation(
                self._dense_model(),
                inputs,
                config,
                generation_config=self._generation_config(),
                multi_token_decode=True,
            )
        received = sorted(len(call.kwargs["config"].quantizer.calibration_dataset) for call in export.call_args_list)
        self.assertEqual(received, [len(calibration), 2 * len(calibration)])

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_pt2e_dynamo(self, family):
        """Every family's graph gains PT2E quantize ops with the x86 quantizer."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._quantization_target(family)
        exported = DynamoExporter().export(
            model,
            copy.deepcopy(inputs),
            DynamoConfig(dynamic=False, quantizer=self._pt2e_quantizer("x86", [copy.deepcopy(inputs)])),
        )
        self.assertTrue(_has_quantize_ops(exported))

    @parameterized.expand(
        [(family, weights_only) for family in ("dense", "moe", "ssm") for weights_only in (False, True)]
    )
    @require_nncf
    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_nncf_torch_fx_dynamo(self, family, weights_only):
        """NNCF quantizes the FX graph through its TorchFX backend (`NNCFTorchFXQuantizer`), fully (quantize ops,
        calibrated on the model's inputs) or weights only (int8 weights): every family's program still runs."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        if family == "moe" and not weights_only:
            self.skipTest("NNCF's TorchFX graph has no shape for the outputs of the MoE routing's `sort`")
        model, inputs = self._quantization_target(family)
        quantizer = self._nncf_quantizer(inputs, weights_only)
        exported = DynamoExporter().export(model, copy.deepcopy(inputs), DynamoConfig(quantizer=quantizer))
        if weights_only:
            tensors = [*exported.state_dict.values(), *exported.constants.values()]
            self.assertTrue(any(tensor.dtype == torch.int8 for tensor in tensors if isinstance(tensor, torch.Tensor)))
        else:
            self.assertTrue(_has_quantize_ops(exported))
        traced_inputs = {name: inputs[name] for name in exported.call_spec.in_spec.children_specs[1].context}
        with torch.no_grad():
            outputs = exported.module()(**copy.deepcopy(traced_inputs))
        self.assertTrue(outputs.logits.isfinite().all())

    @pytest.mark.torch_export_test
    @disable_hub_kernels
    def test_vlm_per_component_quantization(self):
        """A VLM is quantized component by component, each with its own recipe, via a `{component: config}`
        dict on `export_for_generation` (multi-token decode): static int8 on the prompt's `language_model`,
        the lighter dynamic int8 on `decode`, and `lm_head` left in fp32."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._vlm_model()
        static, dynamic = self._pt2e_quantizer("x86"), self._pt2e_quantizer("x86", dynamic=True)
        config = {
            "language_model": DynamoConfig(dynamic=True, quantizer=static),
            "decode": DynamoConfig(dynamic=True, quantizer=dynamic),
            "lm_head": DynamoConfig(dynamic=True),
        }
        components = DynamoExporter().export_for_generation(model, inputs, config, multi_token_decode=True)

        self.assertTrue(_has_quantize_ops(components["language_model"]), "language_model should be quantized")
        self.assertTrue(_has_quantize_ops(components["decode"]), "decode should be quantized")
        self.assertFalse(_has_quantize_ops(components["lm_head"]), "lm_head should stay fp32")
        # the recipes really differ: decode is dynamically quantized, language_model statically
        self.assertTrue(_has_dynamic_quant_ops(components["decode"]), "decode should be dynamically quantized")
        self.assertFalse(_has_dynamic_quant_ops(components["language_model"]), "language_model should be static")


@slow
@require_torch
@require_torchao
class OnnxQuantizationTest(QuantizationTesterMixin, unittest.TestCase):
    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_pt2e_onnx(self, family):
        """The x86 quantizer lowered to ONNX: every family gets a QDQ graph that runs in ONNX Runtime."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        model, inputs = self._quantization_target(family)
        program = OnnxExporter().export(
            model,
            copy.deepcopy(inputs),
            OnnxConfig(
                dynamic=False, external_data=False, quantizer=self._pt2e_quantizer("x86", [copy.deepcopy(inputs)])
            ),
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
    def test_onnxruntime_onnx(self, family, dynamic):
        """ONNX Runtime quantizes the converted model itself (`OnnxRuntimeQuantizer`), statically calibrated on the
        model's inputs or dynamically: every family gains quantize nodes and still runs in ONNX Runtime. Dynamic mode
        is kept to `MatMul`/`Gemm`: ONNX Runtime has no CPU kernel for the `ConvInteger` it makes of Mamba2's
        depthwise conv."""
        from transformers.exporters import OnnxConfig, OnnxExporter, OnnxRuntimeDynamicConfig, OnnxRuntimeQuantizer

        model, inputs = self._quantization_target(family)
        program = OnnxExporter().export(
            model,
            copy.deepcopy(inputs),
            OnnxConfig(
                dynamic=False,
                external_data=False,
                quantizer=OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig(op_types_to_quantize=["MatMul", "Gemm"]))
                if dynamic
                else OnnxRuntimeQuantizer(calibration_dataset=[copy.deepcopy(inputs)]),
            ),
        )
        quantize_op = "DynamicQuantizeLinear" if dynamic else "QuantizeLinear"
        self.assertTrue(any(node.op_type == quantize_op for node in program.model_proto.graph.node))
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))

    @parameterized.expand(
        [(family, weights_only) for family in ("dense", "moe", "ssm") for weights_only in (False, True)]
    )
    @require_onnxscript
    @require_onnxruntime
    @require_nncf
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_nncf_onnx_onnx(self, family, weights_only):
        """NNCF quantizes the converted ONNX model through its ONNX backend (`NNCFOnnxQuantizer`), fully (int8 QDQ,
        calibrated on the model's inputs) or weights only (int8): every family gains quantization nodes and still runs
        in ONNX Runtime. Weight compression leaves `Conv` out: NNCF gives Mamba2's depthwise conv weight a 2-D per-axis
        scale, which ONNX Runtime rejects (per-axis scales must be 1-D)."""
        import nncf

        from transformers.exporters import (
            NNCFOnnxQuantizer,
            NNCFQuantizationConfig,
            NNCFWeightCompressionConfig,
            OnnxConfig,
            OnnxExporter,
        )

        model, inputs = self._quantization_target(family)
        if weights_only:
            quantizer = NNCFOnnxQuantizer(
                NNCFWeightCompressionConfig(
                    mode=nncf.CompressWeightsMode.INT8_SYM,
                    ignored_scope=nncf.IgnoredScope(types=["Conv"], validate=False),
                )
            )
        else:
            quantizer = NNCFOnnxQuantizer(
                NNCFQuantizationConfig(subset_size=1), calibration_dataset=[copy.deepcopy(inputs)]
            )
        program = OnnxExporter().export(
            model, copy.deepcopy(inputs), OnnxConfig(dynamic=False, external_data=False, quantizer=quantizer)
        )
        quantize_op = "DequantizeLinear" if weights_only else "QuantizeLinear"
        self.assertTrue(any(node.op_type == quantize_op for node in program.model_proto.graph.node))
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))

    @parameterized.expand([(False,), (True,)])
    @require_onnxscript
    @require_onnxruntime
    @require_nncf
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_nncf_onnx_tied_embeddings(self, weights_only):
        """A tied `lm_head` reaches its `MatMul` through a `Transpose` of the embedding, which the ONNX optimizer
        doesn't fold past 8192 elements: `NNCFOnnxQuantizer` folds it so NNCF finds the weight and quantizes it."""
        from transformers.exporters import NNCFOnnxQuantizer, OnnxConfig, OnnxExporter

        model = self._dense_model(hidden_size=64, num_hidden_layers=1, vocab_size=1000, tie_word_embeddings=True)
        inputs = {"input_ids": torch.randint(0, 1000, (1, 8)), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        quantizer = self._nncf_quantizer(inputs, weights_only, NNCFOnnxQuantizer)
        program = OnnxExporter().export(
            model, copy.deepcopy(inputs), OnnxConfig(dynamic=False, external_data=False, quantizer=quantizer)
        )
        lm_head = [node for node in program.model.graph if node.op_type == "MatMul"][-1]
        self.assertEqual(lm_head.inputs[1].producer().op_type, "DequantizeLinear")
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))

    @parameterized.expand(
        [
            ("onnxruntime_static", "QuantizeLinear"),
            ("onnxruntime_dynamic", "DynamicQuantizeLinear"),
            ("nncf_quantization", "QuantizeLinear"),
            ("nncf_weight_compression", "DequantizeLinear"),
        ]
    )
    @require_onnxscript
    @require_onnxruntime
    @require_nncf
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_onnx_quantizers_past_2gb(self, recipe, quantize_op):
        """The ONNX quantizers take models past protobuf's 2GB limit: they quantize through files with external data,
        never a serialized `ModelProto`. A one-layer Llama with a 150k vocabulary holds 2.3GB of untied embeddings."""
        from transformers.exporters import (
            NNCFOnnxQuantizer,
            OnnxConfig,
            OnnxExporter,
            OnnxRuntimeDynamicConfig,
            OnnxRuntimeQuantizer,
        )

        model = self._dense_model(
            hidden_size=2048, num_attention_heads=16, num_hidden_layers=1, num_key_value_heads=4, vocab_size=150_000
        )
        self.assertGreater(sum(p.numel() * p.element_size() for p in model.parameters()), 2**31)
        inputs = {"input_ids": torch.randint(0, 64, (1, 8)), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        if recipe == "onnxruntime_static":
            quantizer = OnnxRuntimeQuantizer(calibration_dataset=[copy.deepcopy(inputs)])
        elif recipe == "onnxruntime_dynamic":
            quantizer = OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig(op_types_to_quantize=["MatMul", "Gemm"]))
        else:
            quantizer = self._nncf_quantizer(inputs, recipe == "nncf_weight_compression", NNCFOnnxQuantizer)
        program = OnnxExporter().export(model, copy.deepcopy(inputs), OnnxConfig(dynamic=False, quantizer=quantizer))
        # the IR model: `model_proto` would serialize past the limit
        self.assertTrue(any(node.op_type == quantize_op for node in program.model.graph))
        logits = program(**inputs)[0]
        self.assertTrue(torch.isfinite(torch.as_tensor(logits)).all())

    @parameterized.expand(
        [(family, weights_only) for family in ("dense", "moe", "ssm") for weights_only in (False, True)]
    )
    @require_onnxscript
    @require_onnxruntime
    @require_nncf
    @pytest.mark.onnx_export_test
    @disable_hub_kernels
    def test_nncf_torch_fx_onnx(self, family, weights_only):
        """`NNCFTorchFXQuantizer` quantizes the FX graph for ONNX too: full quantization becomes QDQ nodes, weight
        compression int8 initializers; both run in ONNX Runtime."""
        import onnx

        from transformers.exporters import OnnxConfig, OnnxExporter

        if family == "moe" and not weights_only:
            self.skipTest("NNCF's TorchFX graph has no shape for the outputs of the MoE routing's `sort`")
        model, inputs = self._quantization_target(family)
        quantizer = self._nncf_quantizer(inputs, weights_only)
        # The optimizer would constant-fold the dequantization of these tiny weights (under its 8192-element input
        # limit) back into float initializers; real models' weights are far above it.
        config = OnnxConfig(dynamic=False, external_data=False, optimize=not weights_only, quantizer=quantizer)
        program = OnnxExporter().export(model, copy.deepcopy(inputs), config)
        if weights_only:
            initializers = program.model_proto.graph.initializer
            self.assertTrue(any(tensor.data_type == onnx.TensorProto.INT8 for tensor in initializers))
        else:
            self.assertTrue(_has_onnx_quantize_ops(program))
        outputs = _run_onnx_program(program, copy.deepcopy(inputs))
        self.assertTrue(outputs)
        self.assertTrue(all(o.isfinite().all() for o in outputs.values() if o.is_floating_point()))


@slow
@require_torch
@require_torchao
class OpenVINOQuantizationTest(QuantizationTesterMixin, unittest.TestCase):
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

    @parameterized.expand(
        [(family, weights_only) for family in ("dense", "moe", "ssm") for weights_only in (False, True)]
    )
    @require_openvino
    @require_nncf
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_nncf_openvino_openvino(self, family, weights_only):
        """NNCF quantizes the converted OpenVINO model itself (`NNCFOpenVINOQuantizer`): fully (`FakeQuantize` nodes
        and int8 weights; Mamba2's depthwise conv, a `GroupConvolution`, is left out because NNCF quantizes it along a
        channel axis the CPU plugin's `FakeQuantize` rejects) or weights only (int8 weights, float matmuls)."""
        import nncf

        from transformers.exporters import (
            NNCFOpenVINOQuantizer,
            NNCFQuantizationConfig,
            NNCFWeightCompressionConfig,
            OpenVINOConfig,
            OpenVINOExporter,
        )

        model, inputs = self._quantization_target(family)
        if weights_only:
            quantizer = NNCFOpenVINOQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT8_SYM))
        else:
            quantizer = NNCFOpenVINOQuantizer(
                NNCFQuantizationConfig(
                    subset_size=1, ignored_scope=nncf.IgnoredScope(types=["GroupConvolution"], validate=False)
                ),
                calibration_dataset=[copy.deepcopy(inputs)],
            )
        ov_model = OpenVINOExporter().export(
            model, copy.deepcopy(inputs), OpenVINOConfig(dynamic=False, quantizer=quantizer)
        )
        fake_quantize, low_precision = _openvino_op_counts(ov_model)
        self.assertEqual(fake_quantize > 0, not weights_only)
        self.assertGreater(low_precision, 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_openvino
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_pt2e_openvino(self, family):
        """`PT2EQuantizer` works on OpenVINO too: its weights stay unfolded behind quantize/dequantize pairs, which
        OpenVINO converts to `FakeQuantize` and compresses itself."""
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        model, inputs = self._quantization_target(family)
        ov_model = OpenVINOExporter().export(
            model,
            copy.deepcopy(inputs),
            OpenVINOConfig(dynamic=False, quantizer=self._pt2e_quantizer("x86", [copy.deepcopy(inputs)])),
        )
        self.assertGreater(_openvino_op_counts(ov_model)[0], 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)

    @parameterized.expand(
        [(family, weights_only) for family in ("dense", "moe", "ssm") for weights_only in (False, True)]
    )
    @require_openvino
    @require_nncf
    @pytest.mark.openvino_export_test
    @disable_hub_kernels
    def test_nncf_torch_fx_openvino(self, family, weights_only):
        """`NNCFTorchFXQuantizer` quantizes the FX graph for OpenVINO too: full quantization becomes `FakeQuantize`
        nodes, weight compression int8 constants."""
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        if family == "moe" and not weights_only:
            self.skipTest("NNCF's TorchFX graph has no shape for the outputs of the MoE routing's `sort`")
        model, inputs = self._quantization_target(family)
        quantizer = self._nncf_quantizer(inputs, weights_only)
        ov_model = OpenVINOExporter().export(model, copy.deepcopy(inputs), OpenVINOConfig(quantizer=quantizer))
        fake_quantize, low_precision = _openvino_op_counts(ov_model)
        self.assertEqual(fake_quantize > 0, not weights_only)
        self.assertGreater(low_precision, 0)
        self._assert_openvino_quantized_model_runs(ov_model, model, inputs)


@slow
@require_torch
@require_torchao
class ExecutorchQuantizationTest(QuantizationTesterMixin, unittest.TestCase):
    def _assert_executorch_program_runs(self, program, inputs):
        """The `.pte` runs to finite outputs, unless ExecuTorch's runtime can't service it (`None`: an XNNPACK
        delegate it refuses to compile, a missing portable kernel), which is a runtime limitation rather than an export
        defect."""
        outputs = _run_executorch_program(program, copy.deepcopy(inputs))
        if outputs is not None:
            self.assertTrue(all(o.isfinite().all() for o in outputs if o.is_floating_point()))

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_executorch
    @pytest.mark.executorch_export_test
    @disable_hub_kernels
    def test_pt2e_executorch(self, family):
        """The same `PT2EQuantizer` recipe, lowered to an ExecuTorch `.pte`: every family's graph is quantized
        and lowers to a program. The x86 quantizer is absent — its per-channel q/dq ops have no out variant, so
        they stay undelegated and fail `to_executorch`; XNNPACK wants its per-tensor quantizer instead."""
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter, PT2EQuantizer

        # lowering hides the quantize ops inside the XNNPACK delegate, so check the graph the quantizer returns
        quantized = []
        quantize = PT2EQuantizer.quantize

        def record_quantized(quantizer, *args, **kwargs):
            quantized.append(quantize(quantizer, *args, **kwargs))
            return quantized[-1]

        model, inputs = self._quantization_target(family)
        with patch.object(PT2EQuantizer, "quantize", record_quantized):
            program = ExecutorchExporter().export(
                model,
                copy.deepcopy(inputs),
                ExecutorchConfig(
                    backend="xnnpack",
                    dynamic=False,
                    quantizer=self._pt2e_quantizer("xnnpack", [copy.deepcopy(inputs)]),
                ),
            )
        self.assertTrue(_has_quantize_ops(quantized[0]))
        self._assert_executorch_program_runs(program, inputs)

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @require_executorch
    @require_nncf
    @pytest.mark.executorch_export_test
    @disable_hub_kernels
    def test_nncf_torch_fx_executorch(self, family):
        """`NNCFTorchFXQuantizer`'s weight compression lowers to an ExecuTorch `.pte` (its full quantization doesn't:
        XNNPACK doesn't delegate those quantize ops)."""
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        model, inputs = self._quantization_target(family)
        quantizer = self._nncf_quantizer(inputs, weights_only=True)
        program = ExecutorchExporter().export(
            model, copy.deepcopy(inputs), ExecutorchConfig(backend="xnnpack", dynamic=False, quantizer=quantizer)
        )
        self._assert_executorch_program_runs(program, inputs)
