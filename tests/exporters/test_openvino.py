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
"""OpenVINO fusion tests.

OpenVINO's CPU plugin only fuses the patterns it recognizes, and a graph that misses them still runs, just decomposed
and slower. These tests check that the decode graph `export_for_generation` produces keeps the fused attention with its
KV cache, RoPE and compressed-weight matmuls, on tiny models of each decoder family.
"""

import collections
import unittest

from parameterized import parameterized

from transformers import (
    GenerationConfig,
    GraniteConfig,
    GraniteForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Olmo2Config,
    Olmo2ForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    set_seed,
)
from transformers.exporters import OpenVINOConfig, OpenVINOExporter
from transformers.testing_utils import require_nncf, require_openvino, require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


NUM_LAYERS = 2

MODELS = {
    "llama": (LlamaConfig, LlamaForCausalLM, {}),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, {}),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 16}),
    "phi3": (Phi3Config, Phi3ForCausalLM, {"pad_token_id": 0}),
    "olmo2": (Olmo2Config, Olmo2ForCausalLM, {}),
    "granite": (GraniteConfig, GraniteForCausalLM, {}),
    "mixtral": (MixtralConfig, MixtralForCausalLM, {"num_local_experts": 4, "num_experts_per_tok": 2}),
    "qwen3_moe": (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
        {"head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64},
    ),
}


def _model(model_type, dtype=None):
    """A tiny GQA decoder of `model_type`."""
    config_class, model_class, overrides = MODELS[model_type]
    set_seed(0)
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        **overrides,
    )
    return model_class(config).to(dtype or torch.float32).eval()


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
    def assert_decode_fused(self, decode):
        # The cache update feeds its state as is: anything between them keeps the plugin from fusing the cache into
        # its attention.
        assign_sources = [
            op.input_value(0).get_node().get_type_name()
            for op in decode.get_ordered_ops()
            if op.get_type_name() == "Assign"
        ]
        self.assertEqual(assign_sources, ["Concat"] * 2 * NUM_LAYERS)
        layers = collections.Counter(layer for layer, _ in _compiled_ops(decode))
        self.assertEqual(layers["ScaledDotProductAttention"], NUM_LAYERS)
        self.assertEqual(layers["RoPE"], 2 * NUM_LAYERS)  # queries and keys
        self.assertEqual(layers["MatMul"], 0, "attention left decomposed")

    @parameterized.expand(list(MODELS))
    def test_decode_fuses_attention_rope_and_kv_cache(self, model_type):
        self.assert_decode_fused(_decode(_model(model_type)))

    @parameterized.expand(["mixtral", "qwen3_moe"])
    def test_moe_experts_fuse_into_gather_matmuls(self, model_type):
        decode = _decode(_model(model_type, torch.bfloat16))
        layers = collections.Counter(layer for layer, _ in _compiled_ops(decode))
        # gate, up and down per layer only compute the routed experts
        self.assertEqual(layers["GatherMatmul"], 3 * NUM_LAYERS)
        self.assertEqual(layers["MatMul"], 0)

    def test_single_token_decode_fuses_attention_rope_and_kv_cache(self):
        self.assert_decode_fused(_decode(_model("llama"), multi_token_decode=False))

    def test_16bit_model_fuses_attention_rope_and_kv_cache(self):
        self.assert_decode_fused(_decode(_model("llama", torch.bfloat16)))

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
            and op.get_output_shape(0)[1] == 64
        ]
        self.assertTrue(weights)
        self.assertTrue(all(op.get_element_type() == openvino.Type.bf16 for op in weights))
        self.assertIn("FullyConnected", {layer for layer, _ in _compiled_ops(decode)})

    @require_nncf
    def test_nncf_int8_weights_fuse_compressed_matmuls(self):
        import nncf

        from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig

        quantizer = NNCFOpenVINOQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT8_ASYM))
        decode = _decode(_model("llama", torch.bfloat16), quantizer=quantizer)
        precisions = collections.Counter(
            precision for layer, precision in _compiled_ops(decode) if layer == "FullyConnected"
        )
        # q, k, v, o, gate, up and down per layer read u8 weights in the fused matmul instead of decompressing them
        self.assertGreaterEqual(precisions["u8"], 7 * NUM_LAYERS)
