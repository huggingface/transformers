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

A backend only runs its fused kernels on the patterns it recognizes, and a graph that misses them still runs, decomposed
and slower. Each class exports the tiny model of each decoder family's own tests and counts, per layer, the fused ops
the backend runs, and that none of the ops they replace are left.
"""

import collections
import unittest

import numpy as np
import torch
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
from tests.test_memory_cleanup_mixin import MemoryCleanupMixin
from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
from transformers.exporters import OnnxConfig, OnnxExporter, OpenVINOConfig, OpenVINOExporter
from transformers.testing_utils import (
    require_nncf,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch,
    slow,
)


# model type: the tester of its tiny model
TESTERS = {
    "llama": LlamaModelTester,
    "qwen2": Qwen2ModelTester,
    "qwen3": Qwen3ModelTester,  # per-head query and key norms
    "phi3": Phi3ModelTester,
    "olmo2": Olmo2ModelTester,
    "granite": GraniteModelTester,
    "mistral": MistralModelTester,  # sliding-window attention
    "gemma3": Gemma3TextModelTester,  # sliding-window and full attention layers
    "mixtral": MixtralModelTester,  # mixture of experts
    "qwen3_moe": Qwen3MoeModelTester,  # mixture of experts
}
MOE_MODELS = {"mixtral", "qwen3_moe"}


def _model(model_type, dtype=None, **config):
    """The tiny causal LM the model's own tests use, with `config` overriding its config."""
    set_seed(0)
    model_config = TESTERS[model_type](parent=None).get_config()
    for name, value in config.items():
        setattr(model_config, name, value)
    return AutoModelForCausalLM.from_config(model_config, dtype=dtype or torch.float32).eval()


# The ONNX Runtime ops each layer of a prefill or decode graph must fuse into, per provider. `Neg`, `Softmax` and
# `ReduceMean` count the rotary embeddings, attentions and RMS norms left decomposed, so they must stay at 0.
ORT_UNFUSED = {"Neg": 0, "Softmax": 0, "ReduceMean": 0}
ORT_FUSIONS = {
    "cpu": {"GroupQueryAttention": 1, "MoE": 0, **ORT_UNFUSED},
    "cuda": {"GroupQueryAttention": 1, **ORT_UNFUSED},
    # WebGPU's `GroupQueryAttention` rotates at the positions after the past, not at `position_ids`: rotary stays its
    # own op, for the queries and the keys. It has no `FusedMatMul`, which would run on the CPU.
    "webgpu": {"GroupQueryAttention": 1, "MoE": 0, "RotaryEmbedding": 2, "FusedMatMul": 0, **ORT_UNFUSED},
    # DirectML's `GroupQueryAttention` reads neither rotary, the mask, nor packed q/k/v: `MultiHeadAttention` stays.
    "dml": {"MultiHeadAttention": 1, "GroupQueryAttention": 0, "MoE": 0, "RotaryEmbedding": 2, **ORT_UNFUSED},
}
# Only CUDA has `MoE`: elsewhere the router keeps its softmax (microsoft/onnxruntime#33866).
ORT_MOE_FUSIONS = {
    "cpu": {**ORT_FUSIONS["cpu"], "Softmax": 1},
    "cuda": {**ORT_FUSIONS["cuda"], "MoE": 1},
    "webgpu": {**ORT_FUSIONS["webgpu"], "Softmax": 1},
    "dml": {**ORT_FUSIONS["dml"], "Softmax": 1},
}


def _ort_fusions(model_type, ort_fusions, component):
    """The ONNX Runtime ops each layer of `model_type`'s `component` graph must fuse into for `ort_fusions`."""
    fusions = dict((ORT_MOE_FUSIONS if model_type in MOE_MODELS else ORT_FUSIONS)[ort_fusions])
    if ort_fusions == "cuda" and component == "prefill":
        # CUDA's `GroupQueryAttention` with a mask is slower than `MultiHeadAttention` for a prefill
        # (microsoft/onnxruntime#33869).
        fusions["MultiHeadAttention"] = fusions.pop("GroupQueryAttention")
    if ort_fusions == "cuda" and model_type == "olmo2":
        # A float16 OLMo2 rotates in float32: fused, it would rotate in float16.
        fusions["Neg"] = 2
    return fusions


def _ort_model(model_type="llama", ort_fusions="cpu"):
    """`_model`, in float16 for CUDA, which the per-head query and key norms need to fold into `GroupQueryAttention`.
    The experts are SiLU-gated, as checkpoints are: ONNX Runtime's `MoE` miscomputes gated GELU experts
    (microsoft/onnxruntime#33863). The head size is a multiple of 16, the only ones `GroupQueryAttention` rotates."""
    dtype = torch.float16 if ort_fusions == "cuda" else None
    config = {"hidden_act": "silu"} if model_type in MOE_MODELS else {}
    return _model(model_type, dtype, head_dim=16, **config)


def _padded_inputs(model):
    """Two rows, the first left-padded."""
    return {
        "input_ids": torch.randint(0, model.config.vocab_size, (2, 7)),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1] * 7]),
    }


def _export_for_ort(model, ort_fusions, inputs=None, dynamic=True):
    """The prefill and decode programs of `model`, fused for `ort_fusions`."""
    if inputs is None:
        inputs = {"input_ids": torch.tensor([[1, 5, 7, 9, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    exported = OnnxExporter().export_for_generation(
        model,
        dict(inputs),
        OnnxConfig(dynamic=dynamic, ort_fusions=ort_fusions),
        generation_config=GenerationConfig(max_new_tokens=3),
    )
    return {name: component.program for name, component in exported.components.items()}


def _decode_feeds(program, vocab_size):
    """Inputs for one new token after a random past cache."""
    generator = np.random.default_rng(0)
    inputs = {value.name: [int(dim) for dim in value.shape] for value in program.graph.inputs}
    past_length = next(shape[2] for name, shape in inputs.items() if "cache" in name)
    feeds = {}
    for name, shape in inputs.items():
        if "cache" in name:
            feeds[name] = generator.standard_normal(shape).astype(np.float32)
        elif "mask" in name:
            feeds[name] = np.ones(shape, dtype=np.int64)
        elif "position" in name:
            feeds[name] = np.full(shape, past_length, dtype=np.int64)
        else:
            feeds[name] = generator.integers(0, vocab_size, shape).astype(np.int64)
    return feeds


def _run(program, feeds):
    import onnx_ir as ir
    import onnxruntime as ort

    session = ort.InferenceSession(ir.to_proto(program).SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {item.name: feeds[item.name] for item in session.get_inputs()})


def _group_query_attentions(program):
    return [node for node in program.graph.all_nodes() if node.op_type == "GroupQueryAttention"]


@slow
@require_torch
@require_onnxscript
@require_onnxruntime
class OnnxRuntimeFusionTest(MemoryCleanupMixin, unittest.TestCase):
    """The ONNX Runtime ops `OnnxConfig(ort_fusions=...)` fuses the graph into, for each provider (building a `"cuda"`
    or `"webgpu"` graph needs no GPU), and on CPU the fused graph's outputs against the plain export's."""

    def assert_fused(self, model, program, fusions, component=None):
        num_layers = model.config.num_hidden_layers
        ops = collections.Counter(node.op_type for node in program.graph.all_nodes())
        self.assertEqual({op: ops[op] for op in fusions}, {op: n * num_layers for op, n in fusions.items()}, component)

    @parameterized.expand([(model_type, ort_fusions) for model_type in TESTERS for ort_fusions in ORT_FUSIONS])
    def test_generation_fuses(self, model_type, ort_fusions):
        model = _ort_model(model_type, ort_fusions)
        for component, program in _export_for_ort(model, ort_fusions).items():
            self.assert_fused(model, program, _ort_fusions(model_type, ort_fusions, component), component)

    def test_forward_matches_plain_export(self):
        model = _ort_model()
        inputs = _padded_inputs(model)
        fused = OnnxExporter().export(model, dict(inputs), OnnxConfig(dynamic=False, ort_fusions="cpu")).program
        plain = OnnxExporter().export(model, dict(inputs), OnnxConfig(dynamic=False)).program
        self.assert_fused(model, fused, ORT_FUSIONS["cpu"])
        ops = {node.op_type for node in fused.graph.all_nodes()}
        self.assertLessEqual({"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}, ops)
        self.assertFalse(any(node.domain == "com.microsoft" for node in plain.graph.all_nodes()))
        feeds = {name: value.numpy() for name, value in inputs.items()}
        np.testing.assert_allclose(_run(fused, feeds)[0], _run(plain, feeds)[0], atol=1e-4, rtol=1e-4)

    def test_decode_matches_plain_export(self):
        model = _ort_model()
        inputs = _padded_inputs(model)
        fused, plain = (_export_for_ort(model, fusions, inputs, dynamic=False)["decode"] for fusions in ("cpu", None))
        self.assert_fused(model, fused, ORT_FUSIONS["cpu"])
        feeds = _decode_feeds(plain, model.config.vocab_size)
        for fused_output, plain_output in zip(_run(fused, feeds), _run(plain, feeds)):
            np.testing.assert_allclose(fused_output, plain_output, atol=1e-4, rtol=1e-4)

    def test_decode_group_query_attention_rotates_and_reads_packed_projection(self):
        model = _ort_model()
        attentions = _group_query_attentions(_export_for_ort(model, "cpu")["decode"])
        self.assertTrue(all(node.attributes.get_int("do_rotary", 0) for node in attentions))
        self.assertTrue(all(node.inputs[1] is None and node.inputs[2] is None for node in attentions))

    def test_webgpu_group_query_attention_leaves_rotary(self):
        # WebGPU's rotates at the positions after the past, ignoring `position_ids`.
        model = _ort_model(ort_fusions="webgpu")
        for component, program in _export_for_ort(model, "webgpu").items():
            attentions = _group_query_attentions(program)
            self.assertFalse(any(node.attributes.get_int("do_rotary", 0) for node in attentions), component)


# The fused kernels OpenVINO's CPU plugin must compile each layer of a decode graph into. `MatMul` counts the matmuls
# left decomposed (an unfused attention or expert), so it must stay at 0.
OPENVINO_FUSIONS = {"ScaledDotProductAttention": 1, "RoPE": 2, "MatMul": 0}  # RoPE for the queries and the keys
OPENVINO_MOE_FUSIONS = {**OPENVINO_FUSIONS, "GatherMatmul": 3}  # gate, up and down, computing only the routed experts


def _decode(model, multi_token_decode=True, quantizer=None):
    inputs = {"input_ids": torch.tensor([[1, 5, 7, 9, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    exported = OpenVINOExporter().export_for_generation(
        model,
        inputs,
        OpenVINOConfig(dynamic=True, quantizer=quantizer),
        generation_config=GenerationConfig(max_new_tokens=3),
        multi_token_decode=multi_token_decode,
    )
    return exported.components["decode"].program


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
class OpenVINOFusionTest(MemoryCleanupMixin, unittest.TestCase):
    """The kernels OpenVINO's CPU plugin compiles the decode graph into: the attention fused with its KV cache, RoPE
    and compressed-weight matmuls."""

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
    @parameterized.expand(list(TESTERS))
    def test_decode_fuses(self, model_type):
        model = _model(model_type, torch.bfloat16)
        fusions = OPENVINO_MOE_FUSIONS if model_type in MOE_MODELS else OPENVINO_FUSIONS
        self.assert_decode_fused(model, _decode(model), fusions)

    def test_float32_decode_fuses(self):
        model = _model("llama")
        self.assert_decode_fused(model, _decode(model), OPENVINO_FUSIONS)

    def test_single_token_decode_fuses(self):
        model = _model("llama", torch.bfloat16)
        self.assert_decode_fused(model, _decode(model, multi_token_decode=False), OPENVINO_FUSIONS)

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

    def test_dynamic_repeated_keys_broadcast_in_numpy_mode(self):
        # The plugin runs a bidirectional broadcast to a computed shape in float32 reference code, outside the graph's
        # 16-bit precision; `repeat_kv`'s expand, for grouped-query attention, is one per key and value.
        model = _model("llama", torch.bfloat16, num_attention_heads=4, num_key_value_heads=2)
        inputs = {"input_ids": torch.tensor([[1, 5, 7, 9, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
        program = OpenVINOExporter().export(model, inputs, OpenVINOConfig(dynamic=True)).components["model"].program
        modes = collections.Counter(
            op.get_attributes()["mode"] for op in program.get_ordered_ops() if op.get_type_name() == "Broadcast"
        )
        self.assertEqual(modes["bidirectional"], 0)
        self.assertGreaterEqual(modes["numpy"], 2 * model.config.num_hidden_layers)

    def test_rank_adding_expand_keeps_bidirectional_broadcast(self):
        # An expand that adds leading dimensions: its target and input shapes don't line up elementwise.
        class LeadingExpand(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, hidden_states):
                return {"expanded": hidden_states.expand(2, *hidden_states.shape) * self.scale}

        model, inputs = LeadingExpand().eval(), {"hidden_states": torch.randn(4, 5)}
        exported = OpenVINOExporter().export(model, dict(inputs), OpenVINOConfig(dynamic=True))
        torch.testing.assert_close(exported.to("cpu")(**inputs)["expanded"], model(**inputs)["expanded"])
