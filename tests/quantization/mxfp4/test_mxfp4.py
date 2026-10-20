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

"""MXFP4 (`Mxfp4Config`): GPT-OSS's blocks layout and the packed per-expert one it loads and saves back, the
generate path, and the published GPT-OSS checkpoints."""

import tempfile
import unittest
from unittest.mock import patch

import torch

from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM, Mxfp4Config
from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_torch_accelerator_memory,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...integrations.finegrained.utils import (
    E,
    H,
    I,
    _ChainAssertions,
    _load_and_save,
    _per_expert,
    _quantizer_for,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    MXFP4,
    BlockScaledFormatTesterMixin,
    _cheap_tuning,
)


class Mxfp4EnvironmentTest(unittest.TestCase):
    def test_weight_only_mxfp4_keeps_existing_cuda_architectures(self):
        for capability in ((7, 5), (8, 0), (8, 6)):
            with self.subTest(capability=capability):
                config = Mxfp4Config()
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=capability),
                    patch("transformers.quantizers.block_scaled_utils.is_accelerate_available", return_value=True),
                ):
                    _quantizer_for(config).validate_environment(device_map="cuda")
                self.assertFalse(config.dequantize)

    def test_quantized_activations_keep_the_fp8_hardware_requirement(self):
        for activation_format in ("mxfp4", "mxfp8"):
            with self.subTest(activation_format=activation_format):
                config = Mxfp4Config(activation_format=activation_format)
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=(8, 0)),
                    patch("transformers.quantizers.block_scaled_utils.is_accelerate_available", return_value=True),
                ):
                    _quantizer_for(config).validate_environment(device_map="cuda")
                self.assertTrue(config.dequantize)


@require_torch_accelerator
@require_kernels
class Mxfp4GenerateTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(_cheap_tuning())

    def test_generate_matches_forward_logits(self):
        """`generate` switches the experts to the batched kernels, while a plain forward runs the grouped ones: on
        a tiny GPT-OSS with biased experts, quantized on the fly (W4A16), every decoding step's logits must match
        a forward over the same prefix. Large biases make differences between the two kernels' bias handling visible."""
        torch.manual_seed(0)
        config = GptOssConfig(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=256,
            head_dim=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = GptOssForCausalLM(config).to(torch.bfloat16)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.experts.gate_up_proj_bias.normal_()
                layer.mlp.experts.down_proj_bias.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            quantized = GptOssForCausalLM.from_pretrained(
                tmp, quantization_config=Mxfp4Config(), dtype=torch.bfloat16, device_map=torch_device
            )

        prompt = torch.arange(1, 17, device=torch_device).unsqueeze(0)
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache), torch.no_grad():
                out = quantized.generate(
                    prompt,
                    max_new_tokens=3,
                    do_sample=False,
                    use_cache=use_cache,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                for step, logits in enumerate(out.logits):
                    prefix = out.sequences[:, : prompt.shape[1] + step]
                    expected = quantized(prefix).logits[:, -1].float()
                    torch.testing.assert_close(logits.float(), expected, atol=5e-2, rtol=5e-2)


@require_torch_large_accelerator
@require_kernels
@slow
class Mxfp4ModelTest(unittest.TestCase):
    """The published GPT-OSS checkpoint, quantized and dequantized, and saved back."""

    model_name = "openai/gpt-oss-20b"
    input_text = "Once upon a time"
    EXPECTED_OUTPUTS = {"Once upon a time, in a small town, there lived a young"}
    # what the published checkpoint keeps in full precision
    MODULES_TO_NOT_CONVERT = ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def check_inference_correctness_quantized(self, model, tokenizer):
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_sequences = model.generate(
                **encoded_input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        self.assertIn(generated_text, self.EXPECTED_OUTPUTS)

    def test_gpt_oss_model_loading_quantized_with_device_map(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.check_inference_correctness_quantized(model, tokenizer)

    # ~39 GiB for the dequantized bfloat16 weights.
    @require_torch_accelerator_memory(memory=48)
    def test_gpt_oss_model_loading_dequantized_with_device_map(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=Mxfp4Config(dequantize=True),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.check_inference_correctness_quantized(model, tokenizer)

    # Holds the packed (~13 GiB) and the dequantized (~39 GiB) model at the same time, by design.
    @require_torch_accelerator_memory(memory=56)
    def test_memory_footprint_comparison(self):
        quantization_config = Mxfp4Config(dequantize=True)
        quantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        dequantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        quantized_mem = quantized_model.get_memory_footprint()
        dequantized_mem = dequantized_model.get_memory_footprint()
        self.assertLess(quantized_mem, dequantized_mem)

    # Keeps the packed model alive while reloading the saved copy, dequantized (~13 + ~39 GiB).
    @require_torch_accelerator_memory(memory=56)
    def test_save_mxfp4(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            cleanup(torch_device, gc_collect=True)
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantize=True),
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

    # Quantizes from a bfloat16 checkpoint, so it holds the bfloat16 source and the mxfp4 result at once.
    @require_torch_accelerator_memory(memory=56)
    def test_save_mxfp4_non_quantized(self):
        non_quantized_model_name = "hf-internal-testing/gpt-oss-20b-bf16"
        tokenizer = AutoTokenizer.from_pretrained(non_quantized_model_name)
        loaded_model = GptOssForCausalLM.from_pretrained(
            non_quantized_model_name,
            quantization_config=Mxfp4Config(modules_to_not_convert=self.MODULES_TO_NOT_CONVERT),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loaded_model.save_pretrained(tmp)
            cleanup(torch_device, gc_collect=True)
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantize=True),
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)


@require_torch
class Mxfp4ChainTest(_ChainAssertions, unittest.TestCase):
    """MXFP4 as GPT-OSS ships it: blocks (E, N, K/32, 16) uint8 low-nibble-first E2M1 + biased-127
    exponent scales, gate|up rows INTERLEAVED."""

    FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def _reference_dequant(self, blocks, scales):
        # the mxfp4.py reference semantics: lut[lo], lut[hi] interleaved along K, x 2^(scale-127)
        lut = torch.tensor(self.FP4_VALUES)
        lo = lut[(blocks & 0xF).long()]
        hi = lut[(blocks >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], -1)  # (E, N, K/32, 32)
        exp = (scales.long() - 127).unsqueeze(-1)
        return (vals * torch.pow(torch.tensor(2.0), exp)).reshape(*blocks.shape[:2], -1)

    def test_packed_experts(self):
        """Packed E2M1 per expert (two values per `int8` byte along K) with an E8M0 scale per 32, as
        DeepSeek-V4 ships its experts, merged onto the stacked experts."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randint(-128, 128, (n, k // 2), dtype=torch.int8),
                "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        )
        model, info, saved = _load_and_save(checkpoint, Mxfp4Config())
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)

    def test_gpt_oss_blocks_and_scales(self):
        """GPT-OSS's `{proj}_blocks` `(E, N, K/32, 16)` hold the packed `(E, N, K/2)` bytes the kernels read,
        regrouped per row, beside E8M0 exponent bytes; both save back as they shipped."""
        torch.manual_seed(0)
        checkpoint = {"model.layers.0.self_attn.q_proj.weight": torch.randn(H, H)}
        for proj, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
            prefix = f"model.layers.0.mlp.experts.{proj}"
            checkpoint[f"{prefix}_blocks"] = torch.randint(0, 256, (E, rows, cols // 32, 16), dtype=torch.uint8)
            checkpoint[f"{prefix}_scales"] = torch.randint(120, 130, (E, rows, cols // 32), dtype=torch.uint8)
        model, info, saved = _load_and_save(checkpoint, Mxfp4Config(modules_to_not_convert=["q_proj"]))
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        blocks = checkpoint["model.layers.0.mlp.experts.down_proj_blocks"]
        self.assertEqual(experts.down_proj.dtype, torch.int8)
        self.assertTrue(torch.equal(experts.down_proj.view(torch.uint8), blocks.reshape(E, H, -1)))
        self.assertSavesBack(saved, checkpoint)

    def test_blocks_dequantize_to_the_unquantized_experts_orientation(self):
        """`dequantize=True` lands the blocks as bf16 in the `(E, hidden, 2I)` orientation the unquantized
        GPT-OSS experts hold."""
        from transformers.core_model_loading import Transpose
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks

        torch.manual_seed(0)
        num_experts, rows, hidden = 2, 8, 64
        blocks = torch.randint(0, 256, (num_experts, rows, hidden // 32, 16), dtype=torch.uint8)
        scales = torch.randint(120, 134, (num_experts, rows, hidden // 32), dtype=torch.uint8)
        reference = self._reference_dequant(blocks, scales).transpose(1, 2)

        sources = ["gate_up_proj_blocks$", "gate_up_proj_scales$"]
        tensors = {sources[0]: blocks.clone(), sources[1]: scales.clone()}
        for op in (FineGrainedPackedBlocks(None), FineGrainedDequantize(None), Transpose(1, 2)):
            tensors = op.convert(tensors, source_patterns=sources, target_patterns=["gate_up_proj"])
        self.assertEqual(len(tensors), 1)  # the scale is consumed, not passed down the chain
        torch.testing.assert_close(next(iter(tensors.values())).float(), reference, rtol=0, atol=0)


class Mxfp4Test(BlockScaledFormatTesterMixin, unittest.TestCase):
    spec = MXFP4


class Mxfp4TinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = MXFP4


class Mxfp4LoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = MXFP4
