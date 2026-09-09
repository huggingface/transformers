# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import gc
import os
import unittest

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GemmaQuantizationConfig,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


# Fill in once the released hub repo is published.
MODEL_ID = ""


class GemmaQuantizationConfigTest(unittest.TestCase):
    def test_to_dict_round_trip(self):
        cfg = GemmaQuantizationConfig(num_bits=8, quantize_embeddings=True)
        d = cfg.to_dict()
        for key, value in d.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(d["quant_method"], "gemma")


class ReplaceWithQuantLayersTest(unittest.TestCase):
    def test_replaces_linear_and_embedding(self):
        from transformers.integrations.gemma_quant import (
            QuantizedEmbedding,
            QuantizedLinear,
            replace_with_quant_layers,
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 4, bias=False)
                self.emb = torch.nn.Embedding(16, 8)

        model = Model()
        cfg = GemmaQuantizationConfig(quantize_embeddings=True)
        replace_with_quant_layers(model, quantization_config=cfg)
        self.assertIsInstance(model.lin, QuantizedLinear)
        self.assertIsInstance(model.emb, QuantizedEmbedding)


@require_torch_accelerator
class GemmaQuantKernelsTest(unittest.TestCase):
    """Unit tests verifying fused Triton low-bit GEMV/GEMM kernels against eager PyTorch."""

    def _create_layer(self, in_features, out_features, num_bits, bias=False, dtype=torch.bfloat16):
        from transformers.integrations.gemma_quant import QuantizedLinear

        layer = QuantizedLinear(in_features, out_features, bias=bias, num_bits=num_bits).to(
            device=torch_device, dtype=dtype
        )
        layer.weight_scale.data = torch.randn(out_features, 1, dtype=torch.float32, device=torch_device)
        if num_bits == 2:
            layer.weight.data = torch.randint(
                0, 256, (out_features, (in_features + 3) // 4), dtype=torch.uint8, device=torch_device
            )
        elif num_bits == 4:
            layer.weight.data = torch.randint(
                0, 256, (out_features, (in_features + 1) // 2), dtype=torch.uint8, device=torch_device
            )
        else:
            layer.weight.data = torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8, device=torch_device
            )
        if bias:
            layer.bias.data = torch.randn(out_features, dtype=dtype, device=torch_device)
        return layer

    def test_gemv_int4(self):
        layer = self._create_layer(512, 256, num_bits=4, bias=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device=torch_device)

        # Fused Triton output
        out_fused = layer(x)

        # Eager reference output
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_gemv_int2(self):
        layer = self._create_layer(512, 256, num_bits=2, bias=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device=torch_device)

        out_fused = layer(x)
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_gemv_int8(self):
        layer = self._create_layer(512, 256, num_bits=8, bias=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device=torch_device)

        out_fused = layer(x)
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_gemm_int4(self):
        layer = self._create_layer(512, 256, num_bits=4, bias=True)
        x = torch.randn(16, 512, dtype=torch.bfloat16, device=torch_device)

        out_fused = layer(x)
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_gemm_int2(self):
        layer = self._create_layer(512, 256, num_bits=2, bias=True)
        x = torch.randn(16, 512, dtype=torch.bfloat16, device=torch_device)

        out_fused = layer(x)
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_gemm_int8(self):
        layer = self._create_layer(512, 256, num_bits=8, bias=True)
        x = torch.randn(16, 512, dtype=torch.bfloat16, device=torch_device)

        out_fused = layer(x)
        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)

        cos_sim = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_multidimensional_and_unaligned(self):
        # 3D input: (B=2, S=7, K=330) to test batching and unaligned dimensions
        layer = self._create_layer(330, 150, num_bits=4, bias=True)
        x = torch.randn(2, 7, 330, dtype=torch.bfloat16, device=torch_device)

        out = layer(x)
        self.assertEqual(out.shape, (2, 7, 150))

        w_ref = layer._dequantize_weights(x.dtype)
        out_ref = torch.nn.functional.linear(x, w_ref, layer.bias)
        cos_sim = torch.nn.functional.cosine_similarity(out.float().flatten(), out_ref.float().flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_dispatch_fallback_equivalence(self):
        import os

        layer = self._create_layer(512, 256, num_bits=4, bias=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device=torch_device)

        # 1. Fast path (Triton)
        out_triton = layer(x)

        # 2. Disabled via env var -> Eager fallback
        os.environ["TRANSFORMERS_GEMMA_DISABLE_TRITON"] = "1"
        try:
            out_eager = layer(x)
        finally:
            del os.environ["TRANSFORMERS_GEMMA_DISABLE_TRITON"]

        cos_sim = torch.nn.functional.cosine_similarity(
            out_triton.float().flatten(), out_eager.float().flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_dtypes_supported(self):
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            layer = self._create_layer(256, 128, num_bits=4, bias=False, dtype=dtype)
            x = torch.randn(1, 256, dtype=dtype, device=torch_device)
            out = layer(x)
            self.assertEqual(out.dtype, dtype)

    def test_quantized_experts_equivalence(self):
        from transformers.integrations.gemma_quant import QuantizedGemma4TextExperts

        num_experts = 4
        hidden = 128
        inter = 64
        experts = QuantizedGemma4TextExperts(
            num_experts=num_experts,
            hidden_dim=hidden,
            intermediate_dim=inter,
            num_bits=4,
        ).to(device=torch_device, dtype=torch.bfloat16)

        x = torch.randn(2, hidden, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.tensor([[0, 2], [1, 3]], device=torch_device)
        top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.bfloat16, device=torch_device)

        out_triton = experts(x, top_k_index, top_k_weights)

        os.environ["TRANSFORMERS_GEMMA_DISABLE_TRITON"] = "1"
        try:
            out_eager = experts(x, top_k_index, top_k_weights)
        finally:
            del os.environ["TRANSFORMERS_GEMMA_DISABLE_TRITON"]

        cos_sim = torch.nn.functional.cosine_similarity(
            out_triton.float().flatten(), out_eager.float().flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)


@slow
@require_torch_accelerator
@require_accelerate
@unittest.skipUnless(MODEL_ID, "MODEL_ID is empty — fill in once the released hub repo is published.")
class GemmaQuantInferenceTest(unittest.TestCase):
    """End-to-end smoke test against a freshly-converted local checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(MODEL_ID)
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=torch_device)
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantized_linears_installed(self):
        from transformers.integrations.gemma_quant import QuantizedLinear

        q_proj = self.model.get_submodule("model.language_model.layers.0.self_attn.q_proj")
        self.assertIsInstance(q_proj, QuantizedLinear)

    def test_greedy_generation_capital_of_france(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
        text = self.processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        self.assertIn("Paris", text)
