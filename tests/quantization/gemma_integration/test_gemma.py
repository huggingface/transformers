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
