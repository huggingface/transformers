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
"""Fusion tests, one class per backend.

OpenVINO: its CPU plugin only fuses the patterns it recognizes, and a graph that misses them still runs, just decomposed
and slower. These tests check that the decode graph `export_for_generation` produces keeps the fused attention with its
KV cache, RoPE and compressed-weight matmuls, on the tiny models of each decoder family's own tests.
"""

import collections
import unittest

from parameterized import parameterized

from tests.models.gemma3.test_modeling_gemma3 import Gemma3TextModelTester
from tests.models.granite.test_modeling_granite import GraniteModelTester
from tests.models.llama.test_modeling_llama import LlamaModelTester
from tests.models.mistral.test_modeling_mistral import MistralModelTester
from tests.models.mixtral.test_modeling_mixtral import MixtralModelTester
from tests.models.olmo2.test_modeling_olmo2 import Olmo2ModelTester
from tests.models.phi3.test_modeling_phi3 import Phi3ModelTester
from tests.models.qwen2.test_modeling_qwen2 import Qwen2ModelTester
from tests.models.qwen3.test_modeling_qwen3 import Qwen3ModelTester
from tests.models.qwen3_moe.test_modeling_qwen3_moe import Qwen3MoeModelTester
from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
from transformers.exporters import OpenVINOConfig, OpenVINOExporter
from transformers.testing_utils import require_nncf, require_openvino, require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


# The fused kernels the CPU plugin must compile each layer of a decode graph into. `MatMul` counts the matmuls left
# decomposed (an unfused attention or expert), so it must stay at 0.
DENSE_FUSIONS = {"ScaledDotProductAttention": 1, "RoPE": 2, "MatMul": 0}  # RoPE for the queries and the keys
MOE_FUSIONS = {**DENSE_FUSIONS, "GatherMatmul": 3}  # gate, up and down, computing only the routed experts

# model type: (the tester of its tiny model, fused kernels per layer)
MODELS = {
    "llama": (LlamaModelTester, DENSE_FUSIONS),
    "qwen2": (Qwen2ModelTester, DENSE_FUSIONS),
    "qwen3": (Qwen3ModelTester, DENSE_FUSIONS),
    "phi3": (Phi3ModelTester, DENSE_FUSIONS),
    "olmo2": (Olmo2ModelTester, DENSE_FUSIONS),
    "granite": (GraniteModelTester, DENSE_FUSIONS),
    "mistral": (MistralModelTester, DENSE_FUSIONS),  # sliding-window attention
    "gemma3": (Gemma3TextModelTester, DENSE_FUSIONS),  # sliding-window and full attention layers
    "mixtral": (MixtralModelTester, MOE_FUSIONS),
    "qwen3_moe": (Qwen3MoeModelTester, MOE_FUSIONS),
}


def _model(model_type, dtype=None):
    """The tiny causal LM the model's own tests use."""
    set_seed(0)
    config = MODELS[model_type][0](parent=None).get_config()
    return AutoModelForCausalLM.from_config(config).to(dtype or torch.float32).eval()


def _decode(model, multi_token_decode=True, quantizer=None):
    inputs = {"input_ids": torch.tensor([[1, 5, 7, 9, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    exported = OpenVINOExporter().export_for_generation(
        model,
        inputs,
        OpenVINOConfig(dynamic=True, quantizer=quantizer),
        generation_config=GenerationConfig(max_new_tokens=3),
        multi_token_decode=multi_token_decode,
    )
    return exported["decode"]


def _compiled_ops(ov_model):
    """`(layer type, runtime precision)` of every op the CPU plugin compiled `ov_model` into."""
    import openvino

    # f32 inference: the fusions must not depend on the CPU having bf16 (AMX or AVX-512 BF16)
    compiled = openvino.Core().compile_model(ov_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    return [
        (op.get_rt_info()["layerType"].astype(str), op.get_rt_info()["runtimePrecision"].astype(str))
        for op in compiled.get_runtime_model().get_ordered_ops()
    ]


@slow
@require_torch
@require_openvino
class OpenVINOFusionTest(unittest.TestCase):
    def assert_decode_fused(self, model, decode, fusions):
        num_layers = model.config.num_hidden_layers
        # The cache update feeds its state as is: anything between them keeps the plugin from fusing the cache into
        # its attention.
        assign_sources = [
            op.input_value(0).get_node().get_type_name()
            for op in decode.get_ordered_ops()
            if op.get_type_name() == "Assign"
        ]
        self.assertEqual(assign_sources, ["Concat"] * 2 * num_layers)
        layers = collections.Counter(layer for layer, _ in _compiled_ops(decode))
        self.assertEqual({op: layers[op] for op in fusions}, {op: n * num_layers for op, n in fusions.items()})

    # bf16, as checkpoints ship: MoE experts only fuse with 16-bit weights
    @parameterized.expand(list(MODELS))
    def test_decode_fuses(self, model_type):
        model = _model(model_type, torch.bfloat16)
        self.assert_decode_fused(model, _decode(model), MODELS[model_type][1])

    def test_float32_decode_fuses(self):
        model = _model("llama")
        self.assert_decode_fused(model, _decode(model), DENSE_FUSIONS)

    def test_single_token_decode_fuses(self):
        model = _model("llama", torch.bfloat16)
        self.assert_decode_fused(model, _decode(model, multi_token_decode=False), DENSE_FUSIONS)

    def test_16bit_model_exports_float32_graph_with_16bit_weights(self):
        import openvino

        model = _model("llama", torch.bfloat16)
        decode = _decode(model)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()), "the model keeps its dtype")
        self.assertTrue(all(output.get_element_type() == openvino.Type.f32 for output in decode.outputs))
        weights = [
            op
            for op in decode.get_ordered_ops()
            if op.get_type_name() == "Constant"
            and len(op.get_output_shape(0)) == 2
            and op.get_output_shape(0)[1] == model.config.hidden_size
        ]
        self.assertTrue(weights)
        self.assertTrue(all(op.get_element_type() == openvino.Type.bf16 for op in weights))
        self.assertIn("FullyConnected", {layer for layer, _ in _compiled_ops(decode)})

    @require_nncf
    def test_nncf_int8_weights_fuse_compressed_matmuls(self):
        import nncf

        from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig

        quantizer = NNCFOpenVINOQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT8_ASYM))
        model = _model("llama", torch.bfloat16)
        decode = _decode(model, quantizer=quantizer)
        precisions = collections.Counter(
            precision for layer, precision in _compiled_ops(decode) if layer == "FullyConnected"
        )
        # q, k, v, o, gate, up and down per layer read u8 weights in the fused matmul instead of decompressing them
        self.assertGreaterEqual(precisions["u8"], 7 * model.config.num_hidden_layers)
