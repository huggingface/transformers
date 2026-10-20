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
"""Helpers shared by the fine-grained quantization tests.

They build format configs, quantizers, checkpoint configs, a mocked kernel bundle and a tiny MoE model.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

import torch
from torch import nn

import transformers.integrations.finegrained.layers as fg
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    convert_and_load_state_dict_in_model,
    revert_weight_conversion,
)
from transformers.integrations.moe import use_experts_implementation
from transformers.modeling_utils import LoadStateDictConfig
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import BLOCK_SCALED_CONFIGS


def _format_config(weight_format, flat=False, **options):
    """Return the config for `weight_format`.

    By default FP8 is per-tensor and MXFP4 is W4A4. Pass `flat=True` to keep the config class defaults.
    """
    defaults = (
        {}
        if flat
        else {"fp8": {"weight_block_size": None}, "mxfp4": {"activation_format": None}}.get(weight_format, {})
    )
    return BLOCK_SCALED_CONFIGS[weight_format](**{**defaults, **options})


class _Cfg:
    hidden_size = 64
    num_local_experts = 4
    intermediate_size = 32
    hidden_act = "silu"
    _experts_implementation = None


def _tiny_config(config_cls, **overrides):
    """Return a one-layer `config_cls` with two experts and one expert per token."""
    defaults = {
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "num_experts": 2,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
    }
    return config_cls(**{**defaults, **overrides})


def _quantizer_for(config):
    """Return the quantizer `from_pretrained` picks for this config."""
    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

    method = getattr(config.quant_method, "value", config.quant_method)
    return AUTO_QUANTIZER_MAPPING[method](config)


def _replace(model, config, modules_to_not_convert=None):
    """Swap the modules of `model` the way a load with `config` does."""
    return _quantizer_for(config).replace_modules(model, modules_to_not_convert=modules_to_not_convert)


def _targets(conv):
    t = conv.target_patterns
    return t if isinstance(t, (list, tuple)) else [t]


def _modelopt(**fields):
    return AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "NVFP4", **fields})


def _compressed_tensors(weights, activations=None, fmt="nvfp4-pack-quantized", **fields):
    group = {
        "weights": weights,
        "targets": ["Linear"],
        **({"input_activations": activations} if activations else {}),
    }
    return {"quant_method": "compressed-tensors", "format": fmt, "config_groups": {"group_0": group}, **fields}


@dataclass
class _Call:
    args: tuple
    kwargs: dict


class _Recorder:
    """Records every call and returns a correctly-shaped zero tensor for matmuls."""

    def __init__(self):
        self.calls = {}

    def _op(self, name):
        def op(*args, **kwargs):
            self.calls.setdefault(name, []).append(_Call(args, kwargs))
            if name == "swizzle_mx_scales":
                return args[0]
            if name.startswith("moe_fused"):  # (hidden, top_k_index, top_k_weights, gate_up, down, ...)
                hidden, down = args[0], kwargs["down_proj"]
                return hidden.new_zeros(hidden.shape[0], down.shape[1], dtype=torch.bfloat16)
            a, b = args[0], args[1]
            n = b.shape[-2]
            gather = kwargs.get("gather_idx")
            rows = gather.shape[0] if gather is not None else a.shape[0]
            return a.new_zeros(rows, n, dtype=torch.bfloat16)

        return op


def fake_finegrained_kernel():
    """Return a fake `finegrained-kernels` bundle and the recorder of its calls."""
    rec = _Recorder()
    kernel = mock.Mock()
    for name in (
        "matmul_2d",
        "matmul_grouped",
        "moe_fused_batched",
        "moe_fused_grouped",
        "swizzle_mx_scales",
        "unswizzle_mx_scales",
        "mxfp8_act_quant",
        "mxfp4_act_quant",
        "nvfp4_act_quant",
    ):
        setattr(kernel, name, rec._op(name))
    kernel.get_supported_act_fns = lambda: ("silu", "gelu", "relu")
    kernel.get_supported_norms = lambda: ("rms_norm", "centered_rms_norm", "input_scaled_rms_norm")
    return kernel, rec


@contextmanager
def patch_finegrained_kernel(kernel):
    """Make the finegrained integration load `kernel` as its bundle."""
    with (
        mock.patch.object(fg, "_FINEGRAINED", None),
        mock.patch.object(fg, "is_kernels_available", return_value=True),
        mock.patch.object(fg, "lazy_load_kernel", return_value=kernel),
    ):
        yield


E, H, I = 2, 64, 32  # experts, hidden size, intermediate size


@use_experts_implementation
class _Experts(nn.Module):
    """Unquantized experts that the quantizer replaces with `FineGrainedExperts`."""

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(E, H, I))


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(H, H, bias=False)
        self.mlp = nn.Module()
        self.mlp.experts = _Experts(config)


class _MoeModel(PreTrainedModel):
    """Tiny model with one decoder layer. It holds a dense `q_proj` and a MoE."""

    base_model_prefix = "model"
    config: PreTrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(config)])
        self.post_init()

    def _init_weights(self, module):
        pass  # every weight here comes from the state dict


def _model_mapping():
    """Return the conversion mapping that merges per-expert weights."""
    return get_checkpoint_conversion_mapping("qwen2_moe")


def _conversion_model(quantizer):
    """Build the tiny model's parameter tree before asking the quantizer for its conversions."""
    with torch.device("meta"), mock.patch.object(fg, "is_sm100", return_value=False):
        model = _MoeModel(_moe_config())
        quantizer._process_model_before_weight_loading(model)
    return model


def _moe_config():
    config = PreTrainedConfig()
    config.hidden_size, config.intermediate_size, config.num_local_experts, config.hidden_act = H, I, E, "silu"
    return config


def _load_and_save(checkpoint, quantization_config, model=None, mapping=None, pre_quantized=True):
    """Load `checkpoint` into `model` through the quantizer, then save it back.

    The load runs off SM100, so the scales stay affine and no kernel is needed. `model` defaults to the tiny MoE.
    Pass `pre_quantized=False` to quantize a full-precision checkpoint. Return `(model, loading_info, saved)`.
    """
    quantizer = _quantizer_for(quantization_config)
    quantizer.pre_quantized = pre_quantized
    model = _MoeModel(_moe_config()) if model is None else model
    with mock.patch.object(fg, "is_sm100", return_value=False):
        quantizer._process_model_before_weight_loading(model)
        load_config = LoadStateDictConfig(
            weight_mapping=quantizer.update_weight_conversions(
                _model_mapping() if mapping is None else mapping, model
            ),
            hf_quantizer=quantizer,
        )
        info, _ = convert_and_load_state_dict_in_model(
            model, {k: v.clone() for k, v in checkpoint.items()}, load_config
        )
        model.hf_quantizer = quantizer
        saved = revert_weight_conversion(model, model.state_dict())
    return model, info, saved


def _raw(tensor):
    return tensor.reshape(-1).view(torch.uint8) if tensor.element_size() == 1 else tensor


def _per_expert(make):
    """Build a per-expert checkpoint. `make(n, k)` returns one projection's `{suffix: tensor}`."""
    checkpoint = {}
    for expert in range(E):
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            for suffix, tensor in make(*shape).items():
                checkpoint[f"model.layers.0.mlp.experts.{expert}.{proj}.{suffix}"] = tensor
    for suffix, tensor in make(H, H).items():
        checkpoint[f"model.layers.0.self_attn.q_proj.{suffix}"] = tensor
    return checkpoint


class _ChainAssertions:
    def assertLoadsCleanly(self, info):
        self.assertEqual(
            (info.missing_keys, info.unexpected_keys, info.mismatched_keys, info.conversion_errors),
            (set(), set(), set(), {}),
        )

    def assertSavesBack(self, saved, checkpoint):
        """Check that the save writes the checkpoint's own keys, dtypes and bytes."""
        self.assertEqual(sorted(saved), sorted(checkpoint))
        for key, tensor in checkpoint.items():
            with self.subTest(key=key):
                self.assertEqual((saved[key].dtype, saved[key].shape), (tensor.dtype, tensor.shape))
                self.assertTrue(torch.equal(_raw(saved[key]), _raw(tensor)))

    def assertHoldsInterleaved(self, experts, checkpoint, suffix, held):
        """Check that the experts hold gate and up rows interleaved as `[g0, u0, g1, u1, ...]`."""
        self.assertTrue(experts.holds_interleaved_gate_up)
        for expert in range(E):
            gate = checkpoint[f"model.layers.0.mlp.experts.{expert}.gate_proj.{suffix}"]
            up = checkpoint[f"model.layers.0.mlp.experts.{expert}.up_proj.{suffix}"]
            expected = torch.stack([_raw(gate).view(gate.shape), _raw(up).view(up.shape)], dim=1)
            got = _raw(held[expert]).view(held[expert].shape)
            self.assertTrue(torch.equal(got, expected.reshape(-1, gate.shape[-1])), f"expert {expert} {suffix}")
