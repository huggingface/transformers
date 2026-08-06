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
"""MoE experts of a packed-FP4 compressed-tensors checkpoint run without being decompressed at load time."""

import gc
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.testing_utils import (
    backend_empty_cache,
    require_compressed_tensors,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_triton,
    torch_device,
)
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import CompressedTensorsConfig


if is_torch_available():
    import torch


def _tiny_moe_config():
    return Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def _quantization_config(format, group_size, scale_dtype):
    """A config shaped like the ones RedHat and Moonshot publish: `Linear` targets minus an ignore list."""
    return {
        "quant_method": "compressed-tensors",
        "format": format,
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "format": format,
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "group" if group_size == 32 else "tensor_group",
                    "group_size": group_size,
                    "scale_dtype": scale_dtype,
                    "dynamic": False,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ["lm_head", r"re:.*self_attn.*"],
    }


def _compressor(format, group_size, scale_dtype):
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationStrategy

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy=QuantizationStrategy.GROUP if group_size == 32 else QuantizationStrategy.TENSOR_GROUP,
            group_size=group_size,
            scale_dtype=scale_dtype,
        ),
        format=format,
    )
    return BaseCompressor.load_from_registry(format), scheme


def _build_checkpoints(directory, format):
    """Write a tiny packed-FP4 MoE checkpoint and its dequantized twin.

    The packed one holds the experts in the per-expert `weight_packed` / `weight_scale` layout
    compressed-tensors writes; the twin holds exactly the same values decompressed, in the fused layout
    the model uses. Any difference between the two at inference therefore comes from the runtime, not
    from the quantization.
    """
    from safetensors.torch import save_file

    is_mxfp4 = format == "mxfp4-pack-quantized"
    group_size = 32 if is_mxfp4 else 16
    scale_dtype = torch.uint8 if is_mxfp4 else torch.float8_e4m3fn
    compressor, scheme = _compressor(format, group_size, scale_dtype)

    def compress(weight):
        groups = weight.float().reshape(weight.shape[0], -1, group_size)
        state = {"weight": weight}
        if is_mxfp4:
            # mxfp4 scales are powers of two, so the checkpoint keeps only their e8m0 exponent.
            state["weight_scale"] = torch.exp2(torch.floor(torch.log2(groups.abs().amax(-1) / 6.0))).clamp(
                min=2.0**-127
            )
        else:
            # nvfp4 scales are e4m3, which is why they also need a per-tensor factor to stay in range.
            # Shaped `(1,)`, the way published checkpoints store it — a scalar would change how the
            # dequantization promotes dtypes.
            global_scale = (
                (torch.finfo(torch.float8_e4m3fn).max * 6.0) / weight.float().abs().amax().clamp(min=1e-12)
            ).reshape(1)
            state["weight_global_scale"] = global_scale
            scale = (groups.abs().amax(-1) / 6.0 * global_scale).clamp(min=1e-12)
            state["weight_scale"] = scale.to(torch.float8_e4m3fn).to(torch.float32)
        compressed = compressor.compress(state, scheme)
        dequantized = compressor.decompress(dict(compressed), scheme)["weight"].to(weight.dtype)
        return compressed, dequantized

    torch.manual_seed(0)
    config = _tiny_moe_config()
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.normal_(0, 0.05)

    packed_state, dense_state = {}, {}
    for key, value in model.state_dict().items():
        if ".mlp.experts." not in key:
            packed_state[key] = dense_state[key] = value
            continue
        prefix, proj = key.rsplit(".", 1)
        dequantized_experts = []
        for expert in range(config.num_experts):
            if proj == "gate_up_proj":
                gate, up = value[expert].chunk(2, dim=0)
                parts = {"gate_proj": gate, "up_proj": up}
            else:
                parts = {"down_proj": value[expert]}
            dequantized = {}
            for name, weight in parts.items():
                compressed, dense = compress(weight.contiguous())
                for component, tensor in compressed.items():
                    packed_state[f"{prefix}.{expert}.{name}.{component}"] = tensor
                dequantized[name] = dense
            dequantized_experts.append(torch.cat(list(dequantized.values()), dim=0))
        dense_state[key] = torch.stack(dequantized_experts)

    packed_dir, dense_dir = os.path.join(directory, "packed"), os.path.join(directory, "dense")
    for path, state, quantization_config in (
        (packed_dir, packed_state, _quantization_config(format, group_size, str(scale_dtype))),
        (dense_dir, dense_state, None),
    ):
        os.makedirs(path)
        serialized = config.to_dict()
        serialized["dtype"] = "bfloat16"
        if quantization_config is not None:
            serialized["quantization_config"] = quantization_config
        with open(os.path.join(path, "config.json"), "w") as file:
            json.dump(serialized, file)
        save_file(
            {k: v.contiguous() for k, v in state.items()},
            os.path.join(path, "model.safetensors"),
            metadata={"format": "pt"},
        )
    return packed_dir, dense_dir


def _expert_weight_bytes(model):
    """Bytes held by the expert projections, whether they are dense parameters or packed weights."""

    def nbytes(tensor):
        data = tensor if isinstance(tensor, torch.Tensor) else tensor.storage.data
        return data.numel() * data.element_size()

    total = 0
    for name, module in model.named_modules():
        if not name.endswith(".experts"):
            continue
        for proj in ("gate_up_proj", "down_proj"):
            total += nbytes(getattr(module, proj))
            precision_config = getattr(module, f"{proj}_precision_config", None)
            if precision_config is not None:
                total += nbytes(precision_config.weight_scale)
            for suffix in ("_scale", "_global_scale"):
                extra = getattr(module, f"{proj}{suffix}", None)
                if extra is not None:
                    total += nbytes(extra)
    return total


class PackedExpertsTestMixin:
    """Shared checks; `format` and `implementation` are set by the per-format subclasses."""

    format: str
    implementation: str

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.packed_dir, cls.dense_dir = _build_checkpoints(cls.directory.name, cls.format)
        cls.input_ids = torch.randint(0, 256, (2, 16)).to(torch_device)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def _logits(self, path, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs).to(torch_device).eval()
        with torch.no_grad():
            logits = model(self.input_ids).logits.float()
        return model, logits

    def test_experts_stay_packed_by_default(self):
        model, _ = self._logits(self.packed_dir)
        experts = model.model.layers[0].mlp.experts

        self.assertNotIsInstance(experts.gate_up_proj, torch.nn.Parameter)
        self.assertEqual(experts.config._experts_implementation, self.implementation)
        # Two 4-bit values per byte, so the packed experts must be several times smaller than bf16 ones.
        dequantized, _ = self._logits(self.packed_dir, quantization_config=CompressedTensorsConfig(dequantize=True))
        self.assertLess(_expert_weight_bytes(model) * 3, _expert_weight_bytes(dequantized))

    def test_packed_matches_dequantized_reference(self):
        _, packed = self._logits(self.packed_dir)
        _, dense = self._logits(self.dense_dir)
        # Packed and dense accumulate differently, so this is a tolerance and not an equality.
        torch.testing.assert_close(packed, dense, atol=5e-3, rtol=5e-2)

    def test_dequantize_reproduces_decompressed_weights(self):
        model, logits = self._logits(self.packed_dir, quantization_config=CompressedTensorsConfig(dequantize=True))
        experts = model.model.layers[0].mlp.experts
        self.assertIsInstance(experts.gate_up_proj, torch.nn.Parameter)
        self.assertEqual(experts.gate_up_proj.dtype, torch.bfloat16)

        # Decompressing the checkpoint has to give back exactly the weights it was built from.
        reference, reference_logits = self._logits(self.dense_dir)
        torch.testing.assert_close(
            experts.gate_up_proj, reference.model.layers[0].mlp.experts.gate_up_proj, atol=0, rtol=0
        )
        torch.testing.assert_close(logits, reference_logits, atol=0, rtol=0)

    def test_packed_experts_are_not_saveable(self):
        model, _ = self._logits(self.packed_dir)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                model.save_pretrained(directory)


@require_torch
@require_torch_accelerator
@require_compressed_tensors
@require_kernels
@require_triton(min_version="3.4.0")
class Mxfp4PackedExpertsTest(PackedExpertsTestMixin, unittest.TestCase):
    format = "mxfp4-pack-quantized"
    implementation = "mxfp4"

    def test_falls_back_to_decompression_without_kernels(self):
        # The mxfp4 runtime is the Triton `matmul_ogs` kernel, fetched through `kernels`.
        with patch("transformers.quantizers.quantizer_compressed_tensors.is_kernels_available", return_value=False):
            model, logits = self._logits(self.packed_dir)
        experts = model.model.layers[0].mlp.experts
        self.assertIsInstance(experts.gate_up_proj, torch.nn.Parameter)
        self.assertEqual(experts.gate_up_proj.dtype, torch.bfloat16)

        _, dequantized = self._logits(self.packed_dir, quantization_config=CompressedTensorsConfig(dequantize=True))
        torch.testing.assert_close(logits, dequantized, atol=0, rtol=0)


@require_torch
@require_torch_accelerator
@require_compressed_tensors
class Nvfp4PackedExpertsTest(PackedExpertsTestMixin, unittest.TestCase):
    format = "nvfp4-pack-quantized"
    implementation = "nvfp4"

    def test_expanding_packed_experts_matches_the_decompressed_weights(self):
        """Both load paths must agree on the weights themselves, not just on what the model outputs.

        The per-tensor global scale makes this easy to get wrong by an ulp, depending on whether the
        division against the group scales happens at the weight dtype or in fp32.
        """
        from transformers.integrations.nvfp4 import dequantize_nvfp4

        packed, _ = self._logits(self.packed_dir)
        dequantized, _ = self._logits(self.packed_dir, quantization_config=CompressedTensorsConfig(dequantize=True))

        experts = packed.model.layers[0].mlp.experts
        reference = dequantized.model.layers[0].mlp.experts
        self.assertGreater(experts.gate_up_proj_global_scale.min().item(), 1.0)

        for proj in ("gate_up_proj", "down_proj"):
            for expert in range(experts.num_experts):
                expanded = dequantize_nvfp4(
                    getattr(experts, proj)[expert],
                    getattr(experts, f"{proj}_scale")[expert],
                    getattr(experts, f"{proj}_global_scale")[expert],
                    torch.bfloat16,
                )
                torch.testing.assert_close(expanded, getattr(reference, proj)[expert], atol=0, rtol=0)

    def test_packed_weights_keep_checkpoint_dtypes(self):
        model, _ = self._logits(self.packed_dir)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj.dtype, torch.uint8)
        self.assertEqual(experts.gate_up_proj_scale.dtype, torch.float8_e4m3fn)
        # One e4m3 scale per group of 16, against one packed byte per 2 values.
        self.assertEqual(experts.gate_up_proj_scale.shape[-1] * 8, experts.gate_up_proj.shape[-1])
