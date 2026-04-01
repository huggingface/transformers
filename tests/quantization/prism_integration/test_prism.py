# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file

from transformers import AutoModelForCausalLM, PrismQuantConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations import PrismEmbedding, PrismLinear, replace_with_prism_modules, unpack_prism_weights
from transformers.quantizers.auto import infer_legacy_quantization_config, get_hf_quantizer
from transformers.testing_utils import require_accelerate
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


GROUP_SIZE = 128


def _pack_prism_weight(weight: "torch.Tensor", group_size: int = GROUP_SIZE):
    if weight.shape[-1] % group_size != 0:
        raise ValueError(f"Expected input dimension divisible by {group_size}, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    num_groups = cols // group_size
    blocks = weight.reshape(rows, num_groups, group_size)
    mins = blocks.min(dim=-1, keepdim=True).values
    maxs = blocks.max(dim=-1, keepdim=True).values
    scales = (maxs - mins).clamp(min=1e-7)
    quantized = ((blocks - mins) / scales).round().clamp(0, 1).to(torch.uint8)

    shifts = torch.arange(32, dtype=torch.int64).view(1, 1, 1, 32)
    packed = ((quantized.reshape(rows, num_groups, group_size // 32, 32).to(torch.int64) << shifts).sum(dim=-1)).to(
        torch.uint32
    )
    dequantized = quantized.to(weight.dtype) * scales + mins
    return (
        packed.reshape(rows, -1).contiguous(),
        scales.squeeze(-1).to(weight.dtype).contiguous(),
        mins.squeeze(-1).to(weight.dtype).contiguous(),
        dequantized.reshape(rows, cols).contiguous(),
    )


def _build_tiny_prism_checkpoint(model_dir: Path):
    torch.manual_seed(7)

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    config.quantization = {"bits": 1, "group_size": GROUP_SIZE}
    config.save_pretrained(model_dir)

    base_model = Qwen3ForCausalLM(config).eval()
    reference_model = Qwen3ForCausalLM(config).eval()
    quantized_state = {}

    with torch.no_grad():
        for name, tensor in base_model.state_dict().items():
            if name == "lm_head.weight":
                continue

            if name.endswith(".weight"):
                module_name = name.removesuffix(".weight")
                module = base_model.get_submodule(module_name)
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    packed, scales, biases, dequantized = _pack_prism_weight(tensor.detach().to(torch.float16))
                    quantized_state[name] = packed
                    quantized_state[f"{module_name}.scales"] = scales
                    quantized_state[f"{module_name}.biases"] = biases

                    reference_module = reference_model.get_submodule(module_name)
                    reference_module.weight.copy_(dequantized.to(reference_module.weight.dtype))
                    continue

            quantized_state[name] = tensor.detach().clone()
            if name in reference_model.state_dict():
                target_tensor = reference_model.state_dict()[name]
                target_tensor.copy_(tensor)

        reference_model.tie_weights()

    save_file(quantized_state, str(model_dir / "model.safetensors"))
    return config, reference_model


class PrismQuantConfigTest(unittest.TestCase):
    def test_basic_config(self):
        config = PrismQuantConfig()
        self.assertEqual(config.quant_method.value, "prism")
        self.assertEqual(config.bits, 1)
        self.assertEqual(config.group_size, GROUP_SIZE)

    def test_legacy_quantization_config_is_inferred(self):
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        config.quantization = {"bits": 1, "group_size": GROUP_SIZE}

        inferred = infer_legacy_quantization_config(config)
        self.assertEqual(inferred, {"quant_method": "prism", "bits": 1, "group_size": GROUP_SIZE})

        hf_quantizer, config, _ = get_hf_quantizer(config, None, None, True, {})
        self.assertEqual(type(hf_quantizer).__name__, "PrismHfQuantizer")
        self.assertEqual(config.quantization_config.bits, 1)
        self.assertEqual(config.quantization_config.group_size, GROUP_SIZE)


class PrismModuleTest(unittest.TestCase):
    def test_replace_with_prism_modules_replaces_linears_and_embeddings(self):
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)

        replace_with_prism_modules(model, quantization_config=PrismQuantConfig())

        self.assertIsInstance(model.model.embed_tokens, PrismEmbedding)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, PrismLinear)
        self.assertIsInstance(model.lm_head, PrismLinear)
        self.assertEqual(model._tied_weights_keys["lm_head.scales"], "model.embed_tokens.scales")
        self.assertEqual(model._tied_weights_keys["lm_head.biases"], "model.embed_tokens.biases")

    def test_unpack_prism_weights_round_trip(self):
        weight = torch.randn(8, GROUP_SIZE, dtype=torch.float16)
        packed, scales, biases, dequantized = _pack_prism_weight(weight)
        unpacked = unpack_prism_weights(packed).reshape(weight.shape[0], scales.shape[1], GROUP_SIZE).to(torch.float16)
        reconstructed = unpacked * scales.unsqueeze(-1) + biases.unsqueeze(-1)
        self.assertTrue(torch.allclose(reconstructed.reshape_as(dequantized), dequantized, atol=0, rtol=0))


@require_accelerate
class PrismIntegrationTest(unittest.TestCase):
    def test_auto_model_loads_legacy_prism_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            _, reference_model = _build_tiny_prism_checkpoint(model_dir)

            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16)
            model.eval()

            self.assertIsInstance(model.model.embed_tokens, PrismEmbedding)
            self.assertIsInstance(model.model.layers[0].self_attn.q_proj, PrismLinear)
            self.assertIsInstance(model.lm_head, PrismLinear)

            input_ids = torch.tensor([[1, 7, 9, 11]], dtype=torch.long)
            with torch.inference_mode():
                expected = reference_model(input_ids=input_ids).logits
                actual = model(input_ids=input_ids).logits

            self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3))
