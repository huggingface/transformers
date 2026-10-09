# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The Institute of Foundation Models and the HuggingFace Inc. team. All rights reserved.
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
"""K2 Horizon model configuration."""

import copy
import re

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


def prepare_k2_quantization_config(quantization_config, previous_config=None):
    quant_method = (
        quantization_config.get("quant_method")
        if isinstance(quantization_config, dict)
        else getattr(quantization_config, "quant_method", None)
    )
    if quant_method not in ("fp8", "compressed-tensors"):
        return quantization_config
    if not isinstance(quantization_config, dict):
        serialized = quantization_config.to_dict()
        if quant_method == "compressed-tensors":
            # The compressed-tensors serializer omits these runtime loading options.
            serialized["dequantize"] = quantization_config.dequantize
            serialized["use_optimized_inference"] = quantization_config.use_optimized_inference
        quantization_config = serialized

    quantization_config = copy.deepcopy(quantization_config)
    if quant_method == "compressed-tensors":
        if quantization_config.get("ignore") is not None:
            # Checkpoints include the causal-LM prefix; AutoModel loads the same backbone without it.
            quantization_config["ignore"] = [
                rf"re:^(?:model\.)?{re.escape(name.removeprefix('model.'))}$" if name.startswith("model.") else name
                for name in quantization_config["ignore"]
            ]
        return quantization_config

    modules = quantization_config.get("modules_to_not_convert")
    if modules is None:
        ignored_layers = quantization_config.get("ignored_layers")
        if ignored_layers is not None:
            # Published exclusions are literal module paths, not regex prefixes (e.g. gate vs gate_proj).
            paths = "|".join(re.escape(name.removeprefix("model.")) for name in ignored_layers)
            modules = [rf"^(?:model\.)?(?:{paths})$"] if paths else []
        elif isinstance(previous_config, dict) and previous_config.get("quant_method") == "fp8":
            # Loading overrides such as FineGrainedFP8Config(dequantize=True) carry an empty skip list.
            modules = previous_config.get("modules_to_not_convert")
    modules = list(modules) if modules is not None else ["lm_head"]
    # Preserve the ModuleList container while letting the stock quantizer replace its Linear children.
    expert_container = r"^(?:model\.)?layers\.[0-9]+\.mlp\.experts$"
    if expert_container not in modules:
        modules.append(expert_container)
    quantization_config["modules_to_not_convert"] = modules
    quantization_config.pop("ignored_layers", None)
    return quantization_config


@auto_docstring(checkpoint="IFM/K2-Horizon-0.9B")
@strict
class K2HorizonConfig(PreTrainedConfig):
    r"""
    use_sliding_window (`bool`, *optional*, defaults to `False`):
        Whether every attention layer uses a sliding window.
    layernorm_num_groups (`int`, *optional*, defaults to 1):
        Number of independently normalized groups in each decoder and final RMS normalization.
    query_key_norm (`bool`, *optional*, defaults to `False`):
        Whether to normalize each query and key head, using separate weights for every head.
    rope_head_dim (`int`, *optional*):
        Number of dimensions per attention head to rotate. Defaults to `head_dim`.
    attention_gate_func (`str`, *optional*):
        Activation for the optional post-attention gate, either `"silu"` or `"softplus"`.
    num_experts (`int`, *optional*, defaults to 0):
        Number of routed MLP experts. Zero selects a dense model.
    decoder_sparse_step (`int`, *optional*, defaults to 1):
        Place sparse layers at every `decoder_sparse_step` layers, counting from one.
    mlp_only_layers (`list[int]`, *optional*):
        Zero-based layer indices that use dense MLPs and ordinary value projections.
    moe_gate_bias (`bool`, *optional*, defaults to `False`):
        Whether routers have a correction bias used only for expert selection, not mixture weights.
    num_shared_experts (`int`, *optional*, defaults to 0):
        Number of shared experts, combined into one MLP that processes every token.
    router_score_func (`str`, *optional*, defaults to `"softmax"`):
        Router score activation, either `"softmax"` or `"sigmoid"`.
    router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Multiplier for selected expert weights, after any normalization.
    mova_num_experts (`int`, *optional*, defaults to 0):
        Number of routed value projections in sparse attention layers. Zero disables MoVA.
    mova_num_experts_per_tok (`int`, *optional*, defaults to 0):
        Number of value experts selected per token. MoVA normalizes selected weights when this exceeds one.
    padding_idx (`int`, *optional*):
        Embedding row whose gradient is disabled. Unlike `pad_token_id`, this changes embedding training behavior.

    ```python
    >>> from transformers import K2HorizonConfig, K2HorizonModel

    >>> configuration = K2HorizonConfig()
    >>> model = K2HorizonModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "k2_horizon"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1_000_000.0

    vocab_size: int = 64256
    hidden_size: int = 1536
    intermediate_size: int = 5120
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_sliding_window: bool = False
    sliding_window: int | None = None
    layernorm_num_groups: int = 1
    query_key_norm: bool = False
    rope_head_dim: int | None = None
    attention_gate_func: str | None = None
    num_experts: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] | None = None
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = False
    moe_gate_bias: bool = False
    num_shared_experts: int = 0
    router_score_func: str = "softmax"
    router_scaling_factor: float | None = 1.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0
    padding_idx: int | None = None
    pad_token_id: int | None = 64255
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1

    def __setattr__(self, key, value):
        if key == "quantization_config":
            value = prepare_k2_quantization_config(value, self.__dict__.get(key))
        super().__setattr__(key, value)

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        if self.router_scaling_factor is None:
            self.router_scaling_factor = 1.0
        if self.rope_head_dim is None:
            self.rope_head_dim = self.head_dim
        if self.rope_parameters is None and kwargs.get("rope_scaling") is None:
            self.rope_parameters = {
                "rope_type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "attention_factor": 1.2772588722239782,
                "beta_fast": 128.0,
                "beta_slow": 4.0,
                "truncate": True,
            }
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.num_experts < 0 or self.mova_num_experts < 0 or self.num_shared_experts < 0:
            raise ValueError("Expert counts must be nonnegative.")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive.")
        if any(layer < 0 or layer >= self.num_hidden_layers for layer in self.mlp_only_layers):
            raise ValueError("mlp_only_layers must contain valid zero-based decoder layer indices.")
        if self.num_experts > 0:
            if not 1 <= self.num_experts_per_tok <= self.num_experts:
                raise ValueError("num_experts_per_tok must be between 1 and num_experts.")
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive.")
        if self.mova_num_experts > 0:
            if self.num_experts == 0:
                raise ValueError("MoVA requires sparse MLP layers (num_experts > 0).")
            if not 1 <= self.mova_num_experts_per_tok <= self.mova_num_experts:
                raise ValueError("mova_num_experts_per_tok must be between 1 and mova_num_experts.")
        if self.router_score_func not in ("softmax", "sigmoid"):
            raise ValueError("router_score_func must be 'softmax' or 'sigmoid'.")
        if self.layernorm_num_groups <= 0 or self.hidden_size % self.layernorm_num_groups != 0:
            raise ValueError("hidden_size must be divisible by layernorm_num_groups.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.rope_head_dim <= 0 or self.rope_head_dim > self.head_dim or self.rope_head_dim % 2 != 0:
            raise ValueError("rope_head_dim must be a positive even integer no greater than head_dim.")
        if self.attention_gate_func not in (None, "silu", "softplus"):
            raise ValueError("attention_gate_func must be None, 'silu', or 'softplus'.")


__all__ = ["K2HorizonConfig"]
