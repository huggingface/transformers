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
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaConfig, LlamaForCausalLM, set_seed
from transformers.testing_utils import require_onnxruntime, require_onnxscript, require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import numpy as np
    import torch

    from transformers.exporters import OnnxConfig, OnnxExporter


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
    return AutoModelForCausalLM.from_config(model_config).to(dtype or torch.float32).eval()


def _ops(program) -> collections.Counter:
    return collections.Counter((node.domain, node.op_type) for node in program.model.graph.all_nodes())


def _run(program, feeds):
    import onnx_ir as ir
    import onnxruntime as ort

    session = ort.InferenceSession(ir.to_proto(program.model).SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {item.name: feeds[item.name] for item in session.get_inputs()})


@require_torch
@require_onnxscript
@require_onnxruntime
class OnnxRuntimeFusionTest(unittest.TestCase):
    """`OnnxConfig(ort_fusions=...)`: the ONNX Runtime ops of the fused graph, for each provider (building a `"cuda"` or
    `"webgpu"` graph needs no GPU), and on CPU its outputs against the plain export."""

    @slow
    @parameterized.expand(
        [(model_type, ort_fusions) for model_type in TESTERS for ort_fusions in ("cpu", "cuda", "webgpu")]
    )
    def test_generation_fuses(self, model_type, ort_fusions):
        """Every layer of the prefill and the decode runs ONNX Runtime's attention (CUDA's prefill, with a mask,
        `MultiHeadAttention`; otherwise `GroupQueryAttention`, rotating its query and key itself except on WebGPU) and,
        on CUDA, its `MoE`, with no rotary (`Neg`), attention (`Softmax`) or RMS norm (`ReduceMean`) left
        decomposed."""
        # A CUDA graph is fused in float16, which the per-head query and key norms need to fold into
        # `GroupQueryAttention`. The experts are SiLU-gated, as checkpoints are: ONNX Runtime's `MoE` miscomputes
        # gated GELU experts (microsoft/onnxruntime#33863), so those stay unfused.
        dtype = torch.float16 if ort_fusions == "cuda" else None
        model = _model(model_type, dtype, **({"hidden_act": "silu"} if model_type in MOE_MODELS else {}))
        programs = OnnxExporter().export_for_generation(
            model,
            {"input_ids": torch.tensor([[1, 5, 7, 9, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)},
            OnnxConfig(dynamic=True, ort_fusions=ort_fusions),
            GenerationConfig(max_new_tokens=3),
        )
        layers = model.config.num_hidden_layers
        moe = model_type in MOE_MODELS
        for component, program in programs.items():
            nodes = list(program.model.graph.all_nodes())
            ops = collections.Counter(node.op_type for node in nodes)
            attention = (
                "MultiHeadAttention" if ort_fusions == "cuda" and component == "prefill" else "GroupQueryAttention"
            )
            expected = {
                attention: layers,
                "MoE": layers if moe and ort_fusions == "cuda" else 0,
                # The CPU and WebGPU keep the router's softmax: they have no `MoE` (microsoft/onnxruntime#33866).
                "Softmax": layers if moe and ort_fusions != "cuda" else 0,
                # WebGPU's `GroupQueryAttention` rotates at the positions after the past, not at `position_ids`.
                "RotaryEmbedding": 2 * layers if ort_fusions == "webgpu" else ops["RotaryEmbedding"],
                # A float16 OLMo2 rotates in float32: fused, it would rotate in float16.
                "Neg": 2 * layers if model_type == "olmo2" and ort_fusions == "cuda" else 0,
                "ReduceMean": 0,
            }
            self.assertEqual({op: ops[op] for op in expected}, expected, component)
            if ort_fusions == "webgpu":
                rotating = [node for node in nodes if node.attributes.get_int("do_rotary", 0)]
                self.assertEqual(rotating, [], component)
                # WebGPU has no `FusedMatMul`: it would run on the CPU.
                self.assertEqual(ops["FusedMatMul"], 0, component)

    def setUp(self):
        torch.manual_seed(0)
        # GroupQueryAttention only applies rotary itself for head sizes that are multiples of 16.
        config = LlamaConfig(
            vocab_size=99,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        self.model = LlamaForCausalLM(config).eval()
        # Left-pad the first row so the test covers the padding mask.
        self.inputs = {
            "input_ids": torch.randint(0, 99, (2, 7)),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1] * 7]),
        }

    def test_forward_fuses_norms_and_attention(self):
        """The norms, residual adds and attention of a forward export should be fused, and its logits should match
        the plain export's. The plain export should contain no ONNX Runtime ops."""
        fused = OnnxExporter().export(self.model, dict(self.inputs), OnnxConfig(dynamic=False, ort_fusions="cpu"))
        plain = OnnxExporter().export(self.model, dict(self.inputs), OnnxConfig(dynamic=False))
        ops = _ops(fused)
        for op in ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "GroupQueryAttention"):
            self.assertTrue(any(op_type == op for _, op_type in ops), f"no {op} in {dict(ops)}")
        for op in ("Neg", "Softmax", "ReduceMean"):
            self.assertNotIn(("", op), ops)
        self.assertFalse(any(domain == "com.microsoft" for domain, _ in _ops(plain)))
        feeds = {name: value.numpy() for name, value in self.inputs.items()}
        np.testing.assert_allclose(_run(fused, feeds)[0], _run(plain, feeds)[0], atol=1e-4, rtol=1e-4)

    def test_decode_fuses_cache_update_and_rotary_into_group_query_attention(self):
        """Each layer of a decode step should become a single GroupQueryAttention that updates the cache, applies
        rotary itself and reads the packed query, key and value projection. Its logits and updated cache should match
        the plain export's."""
        generation_config = GenerationConfig(max_new_tokens=4, do_sample=False)
        fused, plain = (
            OnnxExporter().export_for_generation(
                self.model, dict(self.inputs), OnnxConfig(dynamic=False, ort_fusions=fusions), generation_config
            )["decode"]
            for fusions in ("cpu", None)
        )
        attentions = [node for node in fused.model.graph.all_nodes() if node.op_type == "GroupQueryAttention"]
        self.assertEqual(len(attentions), self.model.config.num_hidden_layers)
        self.assertTrue(all(node.attributes.get_int("do_rotary", 0) for node in attentions))
        self.assertTrue(all(node.inputs[1] is None and node.inputs[2] is None for node in attentions))
        self.assertNotIn(("com.microsoft", "RotaryEmbedding"), _ops(fused))
        # Feed one new token after a random past cache.
        generator = np.random.default_rng(0)
        inputs = {value.name: [int(dim) for dim in value.shape] for value in plain.model.graph.inputs}
        past_length = next(shape[2] for name, shape in inputs.items() if "past_key_values" in name)
        feeds = {}
        for name, shape in inputs.items():
            if "past_key_values" in name:
                feeds[name] = generator.standard_normal(shape).astype(np.float32)
            elif "mask" in name:
                feeds[name] = np.ones(shape, dtype=np.int64)
            elif "position" in name:
                feeds[name] = np.full(shape, past_length, dtype=np.int64)
            else:
                feeds[name] = generator.integers(0, 99, shape).astype(np.int64)
        for fused_output, plain_output in zip(_run(fused, feeds), _run(plain, feeds)):
            np.testing.assert_allclose(fused_output, plain_output, atol=1e-4, rtol=1e-4)
