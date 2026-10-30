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
"""Native K2 FP8 regressions using tiny local checkpoints and independently scaled weights."""

import copy
import json
from pathlib import Path

import pytest

from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        CompressedTensorsConfig,
        FineGrainedFP8Config,
        K2HorizonConfig,
        K2HorizonForCausalLM,
        LlamaConfig,
    )
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
else:
    pytest.skip("PyTorch is required for K2 FP8 tests", allow_module_level=True)


def tiny_config(variant="moe", **kwargs):
    settings = {
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 384,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "rope_head_dim": 64,
        "layernorm_num_groups": 2,
        "query_key_norm": variant != "dense",
        "num_experts": 2 if variant != "dense" else 0,
        "num_experts_per_tok": 1 if variant != "dense" else 0,
        "moe_intermediate_size": 384,
        "num_shared_experts": 1 if variant != "dense" else 0,
        "moe_gate_bias": variant != "dense",
        "router_score_func": "sigmoid",
        "mova_num_experts": 2 if variant == "mova" else 0,
        "mova_num_experts_per_tok": 1 if variant == "mova" else 0,
        "attention_gate_func": "softplus" if variant == "mova" else None,
        "mlp_only_layers": [0],
        "bos_token_id": 0,
        "eos_token_id": None,
        "pad_token_id": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }
    settings.update(kwargs)
    return K2HorizonConfig(**settings)


def finegrained_config(ignored):
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ignored,
    }


def compressed_tensors_config(ignored=None):
    # The dense 7B and sparse 375B checkpoints use weight_scale; MoVA uses weight_scale_inv.
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"] if ignored is None else ignored,
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "block",
                    "block_structure": [128, 128],
                    "dynamic": False,
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
    }


def save_checkpoint(folder, variant):
    torch.manual_seed(42)
    # Initialize in the compute dtype without downcasting the float32 rotary buffers.
    reference = K2HorizonForCausalLM._from_config(
        tiny_config(variant), dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    state = reference.state_dict()
    quantized_names = []
    ignored = []
    for name, module in reference.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        quantize = name != "lm_head" if variant == "dense" else ".mlp.experts." in name
        if not quantize:
            ignored.append(name)
            continue
        weight = module.weight.detach().float()
        rows, cols = weight.shape
        blocks = weight.reshape(rows // 128, 128, cols // 128, 128)
        scales = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
        if variant != "mova":
            scales = scales.to(torch.bfloat16)
        expanded_scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        packed = (weight / expanded_scales).to(torch.float8_e4m3fn)
        state[name + ".weight"] = packed
        scale_name = "weight_scale_inv" if variant == "mova" else "weight_scale"
        state[name + "." + scale_name] = scales
        with torch.no_grad():
            module.weight.copy_(packed.float() * expanded_scales)
        quantized_names.append(name)
    quantization = finegrained_config(ignored) if variant == "mova" else compressed_tensors_config(ignored)
    config_dict = reference.config.to_dict()
    config_dict["quantization_config"] = quantization
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
        "AutoModel": "modeling_k2_horizon.K2HorizonModel",
        "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM",
    }
    reference.config = K2HorizonConfig.from_dict(config_dict)
    reference.save_pretrained(folder, state_dict=state.copy())
    # Start from the published metadata spelling, before K2 configuration normalization.
    config_path = Path(folder) / "config.json"
    checkpoint_config = json.loads(config_path.read_text())
    checkpoint_config["quantization_config"] = quantization
    config_path.write_text(json.dumps(checkpoint_config))
    return reference, quantized_names, state


def assert_loading_info(loading, auto_class):
    assert not loading["missing_keys"]
    assert not loading["mismatched_keys"]
    assert set(loading["unexpected_keys"]) == ({"lm_head.weight"} if auto_class is AutoModel else set())


def reference_group_fp8_activations(module, inputs):
    # Independent W8A8 input reference for the published group-128 dynamic scheme.
    value = inputs[0]
    groups = value.unflatten(-1, (value.shape[-1] // 128, 128))
    scale = groups.abs().amax(dim=-1, keepdim=True) / 448.0
    scale = torch.where(scale == 0, torch.finfo(value.dtype).eps, scale)
    packed = (groups / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return ((packed.to(value.dtype) * scale).reshape_as(value),)


def test_fp8_skip_names_do_not_match_numbered_siblings_or_children():
    raw = finegrained_config(["lm_head", "model.layers.1.self_attn.v_experts.1", "model.layers.1.mlp.gate"])
    original = copy.deepcopy(raw)
    config = tiny_config("mova", mova_num_experts=12, quantization_config=raw)
    model = K2HorizonForCausalLM(config)
    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config.from_dict(config.quantization_config), pre_quantized=True
    )
    quantizer._process_model_before_weight_loading(model)
    assert raw == original
    layer = model.model.layers[1]
    assert isinstance(layer.mlp.experts, nn.ModuleList)
    assert not any(isinstance(module, FP8Experts) for module in model.modules())
    assert isinstance(layer.mlp.experts[0].gate_proj, FP8Linear)
    assert isinstance(layer.mlp.shared_experts.gate_proj, FP8Linear)
    assert layer.mlp.gate.weight.dtype == torch.float32
    assert type(layer.self_attn.v_experts[1]) is nn.Linear
    assert isinstance(layer.self_attn.v_experts[10], FP8Linear)
    assert type(model.lm_head) is nn.Linear


def test_fp8_configuration_changes_are_local_and_idempotent():
    raw = finegrained_config(["model.layers.1.mlp.gate", "lm_head"])
    original = copy.deepcopy(raw)
    config = tiny_config(quantization_config=raw)
    restored = K2HorizonConfig.from_dict(config.to_dict())
    assert restored.quantization_config == config.quantization_config
    assert raw == original
    assert LlamaConfig(quantization_config=raw).quantization_config == original
    assert not hasattr(tiny_config(), "quantization_config")
    compressed = compressed_tensors_config()
    assert tiny_config("dense", quantization_config=compressed).quantization_config == compressed


def test_fp8_loading_overrides_preserve_checkpoint_exclusions():
    raw = finegrained_config(["model.layers.1.self_attn.v_experts.1", "lm_head"])
    config = tiny_config("mova", quantization_config=raw)
    expected = copy.deepcopy(config.quantization_config["modules_to_not_convert"])
    for dequantize in (True, False):
        override = FineGrainedFP8Config(dequantize=dequantize)
        config.quantization_config = override
        assert config.quantization_config["modules_to_not_convert"] == expected
        assert config.quantization_config["dequantize"] is dequantize
        assert override.modules_to_not_convert is None


def test_explicit_fp8_skip_regex_retains_subtree_semantics():
    pattern = r"^model\.layers\.1\.mlp\.shared_experts(?:\.|$)"
    config = tiny_config(quantization_config=FineGrainedFP8Config(modules_to_not_convert=[pattern, "lm_head"]))
    model = K2HorizonForCausalLM(config)
    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config.from_dict(config.quantization_config), pre_quantized=True
    )
    quantizer._process_model_before_weight_loading(model)
    assert type(model.model.layers[1].mlp.shared_experts.gate_proj) is nn.Linear
    assert isinstance(model.model.layers[1].mlp.experts[0].gate_proj, FP8Linear)


@pytest.mark.parametrize("auto_class", [AutoModel, AutoModelForCausalLM])
def test_fp8_cpu_dequantization_matches_stored_block_scales(tmp_path, auto_class):
    pytest.importorskip("accelerate")
    reference, _, _ = save_checkpoint(tmp_path, "mova")
    actual, loading = auto_class.from_pretrained(
        tmp_path,
        device_map="cpu",
        dtype=torch.bfloat16,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        output_loading_info=True,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    assert_loading_info(loading, auto_class)
    if auto_class is AutoModel:
        reference = reference.model
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter, reference.get_parameter(name), rtol=0, atol=0)
    reference.set_attn_implementation("eager")
    ids = torch.tensor([[2, 3, 4]])
    output_key = "last_hidden_state" if auto_class is AutoModel else "logits"
    with torch.no_grad():
        torch.testing.assert_close(actual(ids)[output_key], reference(ids)[output_key], rtol=0, atol=0)
        if auto_class is AutoModelForCausalLM:
            torch.testing.assert_close(
                actual.generate(ids, max_new_tokens=2, do_sample=False),
                reference.generate(ids, max_new_tokens=2, do_sample=False),
            )


@pytest.mark.parametrize("auto_class", [AutoModel, AutoModelForCausalLM])
def test_fp8_storage_save_and_reload_on_cpu(tmp_path, monkeypatch, auto_class):
    pytest.importorskip("accelerate")
    _, quantized_names, state = save_checkpoint(tmp_path / "checkpoint", "mova")
    # Only bypass the accelerator check: exercise real replacement, loading, and serialization.
    # No FP8 matmul executes on CPU in this storage test.
    monkeypatch.setattr(FineGrainedFP8HfQuantizer, "validate_environment", lambda *args, **kwargs: None)
    source = tmp_path / "checkpoint"
    for _ in range(2):
        actual, loading = auto_class.from_pretrained(
            source,
            device_map="cpu",
            dtype=torch.bfloat16,
            output_loading_info=True,
            trust_remote_code=False,
        )
        assert_loading_info(loading, auto_class if source.name == "checkpoint" else AutoModelForCausalLM)
        backbone = actual if auto_class is AutoModel else actual.model
        assert isinstance(backbone.layers[1].mlp.experts, nn.ModuleList)
        assert not any(isinstance(module, FP8Experts) for module in actual.modules())
        assert backbone.layers[1].mlp.gate.weight.dtype == torch.bfloat16
        assert type(backbone.layers[1].mlp.shared_experts.gate_proj) is nn.Linear
        assert type(backbone.layers[1].self_attn.v_experts[0]) is nn.Linear
        for name in quantized_names:
            module = actual.get_submodule(name.removeprefix("model.") if auto_class is AutoModel else name)
            assert isinstance(module, FP8Linear)
            assert module.weight.dtype == torch.float8_e4m3fn
            assert module.weight_scale_inv.dtype == torch.float32
            torch.testing.assert_close(module.weight.float(), state[name + ".weight"].float(), rtol=0, atol=0)
            torch.testing.assert_close(module.weight_scale_inv, state[name + ".weight_scale_inv"], rtol=0, atol=0)
        source = tmp_path / "reloaded"
        actual.save_pretrained(source)


@pytest.mark.parametrize("dequantize", [False, True])
@pytest.mark.parametrize("variant", ["dense", "moe"])
@pytest.mark.parametrize("auto_class", [AutoModel, AutoModelForCausalLM])
def test_compressed_tensors_cpu_loading_and_generate(tmp_path, dequantize, variant, auto_class):
    pytest.importorskip("accelerate")
    pytest.importorskip("compressed_tensors")
    reference, quantized_names, state = save_checkpoint(tmp_path / "checkpoint", variant)
    actual, loading = auto_class.from_pretrained(
        tmp_path / "checkpoint",
        device_map="cpu",
        dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(dequantize=dequantize),
        output_loading_info=True,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    assert_loading_info(loading, auto_class)
    backbone = actual if auto_class is AutoModel else actual.model
    if auto_class is AutoModelForCausalLM:
        assert actual.lm_head.weight.dtype == torch.bfloat16
    if variant == "moe":
        assert isinstance(backbone.layers[1].mlp.experts, nn.ModuleList)
        assert backbone.layers[1].mlp.gate.weight.dtype == torch.bfloat16
        assert type(backbone.layers[1].mlp.shared_experts.gate_proj) is nn.Linear
    for name in quantized_names:
        module = actual.get_submodule(name.removeprefix("model.") if auto_class is AutoModel else name)
        if dequantize:
            torch.testing.assert_close(module.weight, reference.get_submodule(name).weight, rtol=0, atol=0)
        else:
            assert module.weight.dtype == torch.float8_e4m3fn
            torch.testing.assert_close(module.weight.float(), state[name + ".weight"].float(), rtol=0, atol=0)
        torch.testing.assert_close(module.weight_scale, state[name + ".weight_scale"], rtol=0, atol=0)
        reference.get_submodule(name).register_forward_pre_hook(reference_group_fp8_activations)
    ids = torch.tensor([[2, 3, 4]])
    if auto_class is AutoModel:
        reference = reference.model
    reference.set_attn_implementation("eager")
    output_key = "last_hidden_state" if auto_class is AutoModel else "logits"
    with torch.no_grad():
        output = actual(ids)[output_key]
        torch.testing.assert_close(output, reference(ids)[output_key], rtol=0, atol=0)
        if auto_class is AutoModelForCausalLM:
            assert actual.generate(ids, max_new_tokens=2, do_sample=False).shape == (1, 5)


@pytest.mark.parametrize("variant", ["dense", "moe"])
@pytest.mark.parametrize("auto_class", [AutoModel, AutoModelForCausalLM])
def test_compressed_tensors_fp8_storage_round_trip(tmp_path, variant, auto_class):
    pytest.importorskip("accelerate")
    pytest.importorskip("compressed_tensors")
    _, quantized_names, state = save_checkpoint(tmp_path / "checkpoint", variant)
    source = tmp_path / "checkpoint"
    for _ in range(2):
        actual, loading = auto_class.from_pretrained(
            source,
            device_map="cpu",
            dtype=torch.bfloat16,
            output_loading_info=True,
            trust_remote_code=False,
        )
        assert_loading_info(loading, auto_class if source.name == "checkpoint" else AutoModelForCausalLM)
        for name in quantized_names:
            module = actual.get_submodule(name.removeprefix("model.") if auto_class is AutoModel else name)
            assert module.weight.dtype == torch.float8_e4m3fn
            assert module.weight_scale.dtype == torch.bfloat16
            torch.testing.assert_close(module.weight.float(), state[name + ".weight"].float(), rtol=0, atol=0)
            torch.testing.assert_close(module.weight_scale, state[name + ".weight_scale"], rtol=0, atol=0)
        source = tmp_path / "reloaded"
        actual.save_pretrained(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 matmul requires a compatible accelerator")
def test_fp8_gpu_forward_generate_and_reload(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("kernels")
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 matmul requires compute capability >= 8.9")
    reference, _, _ = save_checkpoint(tmp_path / "checkpoint", "mova")
    reference.cuda().set_attn_implementation("eager")
    actual = AutoModelForCausalLM.from_pretrained(
        tmp_path / "checkpoint", device_map="cuda", dtype=torch.bfloat16, attn_implementation="eager"
    )
    assert actual.model.layers[1].mlp.experts[0].gate_proj.weight.dtype == torch.float8_e4m3fn
    ids = torch.tensor([[2, 3, 4]], device="cuda")
    with torch.no_grad():
        output = actual(ids).logits
        torch.testing.assert_close(output, reference(ids).logits, rtol=0.05, atol=0.05)
        tokens = actual.generate(ids, max_new_tokens=2, do_sample=False)
        actual.save_pretrained(tmp_path / "reloaded")
        reloaded = AutoModelForCausalLM.from_pretrained(
            tmp_path / "reloaded", device_map="cuda", dtype=torch.bfloat16, attn_implementation="eager"
        )
        torch.testing.assert_close(reloaded(ids).logits, output, rtol=0, atol=0)
        torch.testing.assert_close(reloaded.generate(ids, max_new_tokens=2, do_sample=False), tokens)
