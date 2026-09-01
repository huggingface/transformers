# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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


import dataclasses

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.native_config import (
    Llama4Scaling,
    MistralNativeConfig,
    MOEModelArgs,
    QFormat,
    QuantizationArgs,
    VisionEncoderArgs,
    YarnArgs,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import QuantizationConfigMixin


# A minimal raw `params.json` payload, as read straight off disk.
RAW_MISTRAL_PARAMS = {
    "dim": 4096,
    "n_layers": 32,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-5,
    "head_dim": 128,
    "vocab_size": 32000,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "max_position_embeddings": 32768,
}


def make_hf_fp8_quant_config(activation_scheme: str = "static") -> QuantizationConfigMixin:
    return AutoQuantizationConfig.from_dict(
        {
            "quant_method": "fp8",
            "activation_scheme": activation_scheme,
            "modules_to_not_convert": ["lm_head"],
            "weight_block_size": None,
        }
    )


def make_non_reversible_quant_config() -> QuantizationConfigMixin:
    return AutoQuantizationConfig.from_dict(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
        }
    )


def base_native_config() -> MistralNativeConfig:
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=32768,
    )


def yarn_args() -> YarnArgs:
    return YarnArgs(factor=16.0, original_max_position_embeddings=16384, beta=32.0, alpha=1.0, apply_scale=False)


def llama4_scaling() -> Llama4Scaling:
    return Llama4Scaling(original_max_position_embeddings=16384, beta=0.1)


def vision_encoder_args() -> VisionEncoderArgs:
    return VisionEncoderArgs(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=14,
        image_size=1540,
        intermediate_size=4096,
        num_channels=3,
        max_image_size=1540,
        rope_theta=10000.0,
        mm_projector_id="patch_merge",
        add_pre_mm_projector_layer_norm=True,
        adapter_bias=False,
        spatial_merge_size=2,
        image_token_id=10,
        image_break_token_id=12,
        image_end_token_id=13,
    )


def moe_args() -> MOEModelArgs:
    return MOEModelArgs(
        num_experts=128,
        num_experts_per_tok=4,
        expert_hidden_dim=2048,
        first_k_dense_replace=0,
        num_shared_experts=1,
        routed_scale=1.0,
        num_expert_groups=1,
        num_expert_groups_per_tok=1,
    )


def ministral3_native_config() -> MistralNativeConfig:
    _yarn_args = yarn_args()
    _llama4_scaling = llama4_scaling()
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=1000000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=262144,
        tied_embeddings=True,
        yarn=_yarn_args,
        llama_4_scaling=_llama4_scaling,
    )


def mistral3_native_config() -> MistralNativeConfig:
    _vision_encoder_args = vision_encoder_args()
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=1000000000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=131072,
        vision_encoder=_vision_encoder_args,
    )


def mistral4_native_config() -> MistralNativeConfig:
    _moe_args = moe_args()
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=1048576,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        kv_lora_rank=256,
        v_head_dim=128,
        yarn=YarnArgs(factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0, apply_scale=False),
        llama_4_scaling=Llama4Scaling(original_max_position_embeddings=8192, beta=0.1),
        moe=_moe_args,
    )


def mistral4_asymmetric_mla_native_config() -> MistralNativeConfig:
    """A Mistral4 (MLA) native config with an asymmetric `qk_rope_head_dim` / `qk_nope_head_dim`
    split, so `partial_rotary_factor` cannot be produced by the swapped
    `qk_nope / (qk_nope + qk_rope)` formula. Every other MLA fixture in this module has an equal
    rope/nope split, which makes the correct and swapped formulas indistinguishable.
    """
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=1048576,
        q_lora_rank=1024,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        kv_lora_rank=256,
        v_head_dim=128,
        moe=moe_args(),
    )


def expected_mistral_hf_config() -> MistralConfig:
    return MistralConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=32768,
        sliding_window=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        quantization_config=None,
    )


def expected_ministral3_hf_config() -> Ministral3Config:
    return Ministral3Config(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=262144,
        sliding_window=None,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
        quantization_config=None,
    )


def expected_mistral4_hf_config() -> Mistral4Config:
    return Mistral4Config(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=1048576,
        sliding_window=None,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        kv_lora_rank=256,
        v_head_dim=128,
        n_routed_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=2048,
        first_k_dense_replace=0,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        quantization_config=None,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 128.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "partial_rotary_factor": 0.5,
        },
    )


def expected_mistral3_hf_config() -> Mistral3Config:
    text_config = MistralConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=131072,
        sliding_window=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000000.0,
        },
        quantization_config=None,
    )
    vision_config = PixtralVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=14,
        image_size=1540,
        intermediate_size=4096,
        num_channels=3,
        hidden_act="silu",
        rope_theta=10000.0,
    )
    return Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        multimodal_projector_bias=False,
        image_token_id=10,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        quantization_config=None,
        tie_word_embeddings=False,
    )


def perturbed_mistral_native_config() -> MistralNativeConfig:
    """A base Mistral native config with every applicable field set to a non-default value."""
    return MistralNativeConfig(
        dim=5120,
        n_layers=40,
        head_dim=160,
        hidden_dim=18432,
        n_heads=40,
        n_kv_heads=10,
        rope_theta=50000.0,
        norm_eps=1e-6,
        vocab_size=33000,
        max_position_embeddings=65536,
        sliding_window=8192,
        tied_embeddings=True,
        quantization=QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act="TENSOR"),
    )


def perturbed_ministral3_native_config() -> MistralNativeConfig:
    """A Ministral3 native config with every applicable field set to a non-default value."""
    return MistralNativeConfig(
        dim=5120,
        n_layers=40,
        head_dim=160,
        hidden_dim=18432,
        n_heads=40,
        n_kv_heads=10,
        rope_theta=2000000.0,
        norm_eps=1e-6,
        vocab_size=33000,
        max_position_embeddings=1048576,
        sliding_window=8192,
        tied_embeddings=False,
        yarn=YarnArgs(factor=32.0, original_max_position_embeddings=32768, beta=16.0, alpha=2.0, apply_scale=True),
        llama_4_scaling=Llama4Scaling(original_max_position_embeddings=32768, beta=0.25),
        quantization=QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act="TENSOR"),
    )


def perturbed_mistral4_native_config() -> MistralNativeConfig:
    """A Mistral4 (MLA + MoE) native config with every field set to a non-default value.

    `head_dim` is kept consistent with `qk_nope_head_dim + qk_rope_head_dim`, as anything
    else is rejected.
    """
    return MistralNativeConfig(
        dim=5120,
        n_layers=48,
        head_dim=192,
        hidden_dim=20480,
        n_heads=48,
        n_kv_heads=48,
        rope_theta=20000.0,
        norm_eps=1e-6,
        vocab_size=34000,
        max_position_embeddings=1048576,
        sliding_window=4096,
        tied_embeddings=True,
        q_lora_rank=2048,
        qk_rope_head_dim=96,
        qk_nope_head_dim=96,
        kv_lora_rank=512,
        v_head_dim=192,
        yarn=YarnArgs(factor=64.0, original_max_position_embeddings=16384, beta=16.0, alpha=2.0, apply_scale=True),
        llama_4_scaling=Llama4Scaling(original_max_position_embeddings=16384, beta=0.2),
        moe=MOEModelArgs(
            first_k_dense_replace=2,
            num_experts=256,
            num_experts_per_tok=8,
            num_expert_groups=4,
            num_expert_groups_per_tok=2,
            routed_scale=2.5,
            expert_hidden_dim=4096,
            num_shared_experts=2,
            expert_parallel=4,
            expert_model_parallel=2,
            route_every_n=3,
        ),
        quantization=QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act="TENSOR"),
    )


def perturbed_vision_encoder_args() -> VisionEncoderArgs:
    """A vision encoder config with every field set to a non-default value.

    `mm_projector_id` and `add_pre_mm_projector_layer_norm` are kept at their only supported
    values, as anything else is rejected.
    """
    return VisionEncoderArgs(
        hidden_size=2048,
        num_hidden_layers=32,
        num_attention_heads=32,
        patch_size=16,
        image_size=2048,
        intermediate_size=8192,
        num_channels=4,
        max_image_size=1024,
        rope_theta=20000.0,
        mm_projector_id="patch_merge",
        add_pre_mm_projector_layer_norm=True,
        adapter_bias=True,
        spatial_merge_size=4,
        image_token_id=20,
        image_break_token_id=99,
        image_end_token_id=100,
    )


def perturbed_mistral3_native_config() -> MistralNativeConfig:
    """A Mistral3 (VLM) native config with every field set to a non-default value."""
    return MistralNativeConfig(
        dim=5120,
        n_layers=40,
        head_dim=160,
        hidden_dim=18432,
        n_heads=40,
        n_kv_heads=10,
        rope_theta=5000000.0,
        norm_eps=1e-6,
        vocab_size=33000,
        max_position_embeddings=262144,
        sliding_window=6144,
        tied_embeddings=True,
        quantization=QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act="TENSOR"),
        vision_encoder=perturbed_vision_encoder_args(),
    )


def perturbed_mistral3_moe_native_config() -> MistralNativeConfig:
    """A Mistral3 (VLM) native config whose text sub-config is itself Mistral4 (MLA + MoE).

    Has both `vision_encoder` and `moe` set, with every field at a non-default value, so the
    preserved extras must carry both sections at once.
    """
    return dataclasses.replace(
        perturbed_mistral4_native_config(),
        vision_encoder=perturbed_vision_encoder_args(),
    )
