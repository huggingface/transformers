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
"""`OnnxConfig(ort_fusions="cpu")`: which ONNX Runtime ops a tiny Llama's forward and decode graphs fuse into, and that
the fused graphs compute what the plain ones do. The per-model export tests (`test_export.py`) check the outputs
against eager on every architecture; these catch a fusion that silently stops applying."""

import collections
import unittest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.testing_utils import require_onnxruntime, require_onnxscript, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import numpy as np
    import torch

    from transformers.exporters import OnnxConfig, OnnxExporter


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
class OnnxRuntimeFusionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Head size 16: what `GroupQueryAttention` rotates itself; two key-value heads for four query heads.
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
        self.inputs = {
            "input_ids": torch.randint(0, 99, (2, 7)),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1] * 7]),
        }

    def test_forward(self):
        fused = OnnxExporter().export(self.model, dict(self.inputs), OnnxConfig(dynamic=False, ort_fusions="cpu"))
        plain = OnnxExporter().export(self.model, dict(self.inputs), OnnxConfig(dynamic=False))
        ops = _ops(fused)
        for op in ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "GroupQueryAttention"):
            self.assertTrue(any(op_type == op for _, op_type in ops), f"no {op} in {dict(ops)}")
        # Nothing left the fusions should have taken: rotate-half, softmax, the norms' mean.
        for op in ("Neg", "Softmax", "ReduceMean"):
            self.assertNotIn(("", op), ops)
        # Not ONNX Runtime's: a plain export has standard ops only.
        self.assertFalse(any(domain == "com.microsoft" for domain, _ in _ops(plain)))
        feeds = {name: value.numpy() for name, value in self.inputs.items()}
        np.testing.assert_allclose(_run(fused, feeds)[0], _run(plain, feeds)[0], atol=1e-4, rtol=1e-4)

    def test_decode(self):
        generation_config = GenerationConfig(max_new_tokens=4, do_sample=False)
        fused, plain = (
            OnnxExporter().export_for_generation(
                self.model, dict(self.inputs), OnnxConfig(dynamic=False, ort_fusions=fusions), generation_config
            )["decode"]
            for fusions in ("cpu", None)
        )
        attentions = [node for node in fused.model.graph.all_nodes() if node.op_type == "GroupQueryAttention"]
        self.assertEqual(len(attentions), 2)
        self.assertTrue(all(node.attributes.get_int("do_rotary", 0) for node in attentions))
        self.assertNotIn(("com.microsoft", "RotaryEmbedding"), _ops(fused))
        generator = np.random.default_rng(0)
        feeds = {}
        for value in plain.model.graph.inputs:
            shape = [int(dim) for dim in value.shape]
            if "past_key_values" in value.name:
                feeds[value.name] = generator.standard_normal(shape).astype(np.float32)
            elif "mask" in value.name:
                feeds[value.name] = np.ones(shape, dtype=np.int64)
            elif "position" in value.name:
                feeds[value.name] = np.full(shape, shape[-1] + 6, dtype=np.int64)
            else:
                feeds[value.name] = generator.integers(0, 99, shape).astype(np.int64)
        for fused_output, plain_output in zip(_run(fused, feeds), _run(plain, feeds)):
            np.testing.assert_allclose(fused_output, plain_output, atol=1e-4, rtol=1e-4)
