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
(any PT2E `Quantizer` — `X86InductorQuantizer`, `XNNPACKQuantizer`, a vendor `QnnQuantizer`, …) on the
export config and the graph is quantized (`prepare_pt2e` → calibrate → `convert_pt2e`) before it's
returned/lowered — no hardcoded schemes. The tests cover:

- **`test_quantized_{dynamo,onnx,executorch}`** — one per exporter backend (so each carries the right CI
  marker), each over the architecture families (dense / MoE / SSM) with the quantizer(s) natural to that
  backend: dynamo/onnx use the torchao-native `x86` quantizer (graph-level QDQ, no executorch dep); the
  executorch backend uses the per-tensor `xnnpack` and vendor `qnn` quantizers it can delegate (per-channel
  x86 has no delegated out variant). One structural check per cell — the artifact exports and carries the
  quant ops (dynamo `quantize`/`dequantize`, ONNX QDQ that loads in ORT, int8 `.pte`) — driven entirely by
  `config.quantizer`, no per-case code. QNN HTP's own gaps on MoE routing / SSM conv1d, and SDK/dep gaps,
  skip with a reason.
- **`test_vlm_per_component_quantization`** — a VLM quantized component-by-component, each with its OWN
  recipe (vision encoder static int8, decoder dynamic int8, `lm_head` fp32) via a per-component config dict.
- **calibration** — the generate-level `calibration_dataset` fanned out per component, and the
  single-sample fallback (with a warning) when it's omitted.

Quantization runs on the decomposed generation components (whose attention mask is a precomputed input),
avoiding the in-graph mask construction that trips PT2E on a full-model forward.
"""

import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest
from parameterized import parameterized

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.utils import capture_calibration_inputs, decompose_for_generation
from transformers.testing_utils import (
    require_executorch,
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
    corrupt the pytest process for the other tests. Use a script file rather than `python -c`: on an
    old glibc the installer re-execs Python under a staged loader and only the file path survives."""
    with tempfile.NamedTemporaryFile("w", suffix=".py") as probe:
        probe.write("import executorch.backends.qualcomm\n")
        probe.flush()
        try:
            run_command([sys.executable, probe.name])
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

    def _x86_quantizer(self, dynamic=False):
        """torchao-native quantizer (no ExecuTorch dependency): static per-channel int8, or — with
        `dynamic=True` — dynamic int8 (runtime-quantized activations, int8 weights), the lighter recipe
        typical for decoders (a true weight-only PT2E quantizer isn't available in torchao/executorch)."""
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )

        return X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_dynamic=dynamic))

    def _xnnpack_quantizer(self):
        """Per-tensor quantizer for the XNNPACK ExecuTorch backend."""
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        return XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    def _qnn_quantizer(self):
        """Vendor Qualcomm HTP quantizer (only lowers via the QNN ExecuTorch backend)."""
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

        return QnnQuantizer()

    def _quantizer(self, name):
        return {"x86": self._x86_quantizer, "xnnpack": self._xnnpack_quantizer, "qnn": self._qnn_quantizer}[name]()

    def _family_unit(self, family):
        """A single quantizable `(model, inputs)` unit per family — the module carrying that family's
        distinct ops: dense/MoE `decode` component, SSM full forward."""
        if family == "dense":
            return self._decode_component(self._tiny_model())
        if family == "moe":
            return self._decode_component(self._moe_model())
        if family == "ssm":
            # no static KV cache → quantize the full forward (no in-graph attention mask to trip PT2E)
            return self._ssm_model(), {"input_ids": torch.randint(0, 64, (1, 8))}
        raise ValueError(f"unknown family {family}")

    # ──────────────────────────────── Dynamo ────────────────────────────────

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

    @pytest.mark.torch_export_test
    def test_calibration_dataset_fans_out_per_component(self):
        """A single generate-level `calibration_dataset` is captured into a per-component set (one entry
        per generation component, each the same length as the input dataset) so every decomposed component
        calibrates on realistic activations rather than the sample inputs."""
        model = self._tiny_model()
        calibration = [
            {"input_ids": torch.randint(0, 64, (1, n)), "attention_mask": torch.ones(1, n, dtype=torch.long)}
            for n in (3, 4, 5)
        ]
        captured = capture_calibration_inputs(
            model, copy.deepcopy(calibration), generation_config=self._generation_config(), multi_token_decode=True
        )
        self.assertEqual(set(captured), {"prefill", "decode"})
        self.assertTrue(all(len(inputs) == len(calibration) for inputs in captured.values()))

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @pytest.mark.torch_export_test
    def test_quantized_dynamo(self, family):
        """Every architecture family quantizes to a quantized FX graph on the Dynamo backend, via the
        torchao-native x86 quantizer (the natural graph-level PT2E quantizer, no executorch dependency) —
        the same `config.quantizer` mechanism, no per-case code. The family unit is the module carrying
        that family's distinct ops (dense/MoE `decode` component, SSM full forward). Static export."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._family_unit(family)
        exported = DynamoExporter().export(
            model,
            copy.deepcopy(inputs),
            DynamoConfig(dynamic=False, quantizer=self._x86_quantizer(), calibration_dataset=[copy.deepcopy(inputs)]),
        )
        self.assertTrue(any(n.op == "call_function" and "quantize" in str(n.target) for n in exported.graph.nodes))

    @pytest.mark.torch_export_test
    def test_vlm_per_component_quantization(self):
        """A VLM is quantized component-by-component, each with its OWN recipe — the realistic pattern —
        via a `{component: config}` dict on `export_for_generation` (multi-token decode). The vision
        encoder and projector get static int8; the language model and the multi-token decode get the
        lighter dynamic recipe; `lm_head` stays fp32. Recipes are chosen per component, not globally."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        model, inputs = self._vlm_model()
        # False = static int8 (vision side), True = dynamic int8 (language/decoder side)
        recipes = {
            "image_encoder": False,
            "multi_modal_projector": False,
            "language_model": True,
            "decode": True,
        }
        config = {
            name: DynamoConfig(dynamic=True, quantizer=self._x86_quantizer(dynamic=dyn))
            for name, dyn in recipes.items()
        }
        config["lm_head"] = DynamoConfig(dynamic=True)  # per-component choice: keep the output head in fp32
        components = DynamoExporter().export_for_generation(model, inputs, config, multi_token_decode=True)

        def is_quantized(exported):
            return any(n.op == "call_function" and "quantize" in str(n.target) for n in exported.graph.nodes)

        def is_dynamic(exported):
            return any(n.op == "call_function" and "choose_qparams" in str(n.target) for n in exported.graph.nodes)

        for name in recipes:
            self.assertTrue(is_quantized(components[name]), f"{name} should be quantized")
        self.assertFalse(is_quantized(components["lm_head"]), "lm_head should stay fp32")
        # the recipes really differ: the decoder side is dynamically quantized, the vision side is static
        self.assertTrue(is_dynamic(components["decode"]), "decode should be dynamically quantized")
        self.assertFalse(is_dynamic(components["image_encoder"]), "image_encoder should be static int8")

    # ──────────────────────────────── ONNX ──────────────────────────────────

    @parameterized.expand([("dense",), ("moe",), ("ssm",)])
    @pytest.mark.onnx_export_test
    def test_quantized_onnx(self, family):
        """The same x86-quantizer recipe, lowered to ONNX: every family produces a QDQ graph
        (QuantizeLinear nodes) that ONNX Runtime can load. Static export throughout."""
        from transformers.utils import is_onnxruntime_available, is_onnxscript_available

        if not (is_onnxruntime_available() and is_onnxscript_available()):
            self.skipTest("requires onnxruntime + onnxscript")

        from collections import Counter

        import onnxruntime as ort

        from transformers.exporters import OnnxConfig, OnnxExporter

        model, inputs = self._family_unit(family)
        program = OnnxExporter().export(
            model,
            copy.deepcopy(inputs),
            OnnxConfig(
                dynamic=False,
                quantizer=self._x86_quantizer(),
                calibration_dataset=[copy.deepcopy(inputs)],
                external_data=False,
            ),
        )
        op_types = Counter(node.op_type for node in program.model_proto.graph.node)
        self.assertGreater(op_types["QuantizeLinear"], 0)
        # the QDQ graph must be a valid ONNX model ORT can load
        ort.InferenceSession(program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"])

    # ────────────────────────────── ExecuTorch ──────────────────────────────

    @parameterized.expand(
        [(family, quantizer) for family in ("dense", "moe", "ssm") for quantizer in ("xnnpack", "qnn")]
    )
    @require_executorch
    @pytest.mark.executorch_export_test
    def test_quantized_executorch(self, family, quantizer):
        """The same `config.quantizer` recipe, lowered to an ExecuTorch `.pte`: every family × delegatable
        quantizer produces a program. The x86 quantizer is absent — its per-channel q/dq ops have no out
        variant, so they stay undelegated and fail `to_executorch`; the ExecuTorch backends want the
        per-tensor `xnnpack`/`qnn` quantizers instead."""
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        if quantizer == "qnn":
            if not _qnn_available():
                self.skipTest("requires the Qualcomm QNN SDK")
            if family == "moe":
                # QNN's quantizer annotates the int64 routing `arange` for per-tensor quant, which its
                # `quantize_per_tensor` meta kernel rejects (float-only). A QNN HTP limitation, not ours.
                self.skipTest("QNN HTP quantizer can't annotate MoE integer routing tensors")
            if family == "ssm":
                # QNN's `CanonicalizeConv` pass unconditionally `unsqueeze`s a conv bias, which Mamba2's
                # bias-less grouped `conv1d` doesn't have. A QNN HTP limitation, not ours.
                self.skipTest("QNN HTP `CanonicalizeConv` pass can't lower Mamba's bias-less conv1d")

        et_backend = "qnn" if quantizer == "qnn" else "xnnpack"
        model, inputs = self._family_unit(family)
        program = ExecutorchExporter().export(
            model,
            copy.deepcopy(inputs),
            ExecutorchConfig(
                backend=et_backend,
                dynamic=False,
                quantizer=self._quantizer(quantizer),
                calibration_dataset=[copy.deepcopy(inputs)],
            ),
        )
        self.assertIsNotNone(program)
