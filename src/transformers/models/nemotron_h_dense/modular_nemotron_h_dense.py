# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import math

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...models.jamba.modeling_jamba import (
    JambaAttention,
    JambaAttentionDecoderLayer,
    JambaMambaDecoderLayer,
    JambaModel,
    JambaPreTrainedModel,
)
from ...models.llama.modeling_llama import LlamaRMSNorm
from ...models.nemotron.modeling_nemotron import NemotronMLP
from ...models.zamba.modeling_zamba import ZambaForCausalLM
from ...models.zamba2.modeling_zamba2 import Zamba2MambaMixer, Zamba2RMSNormGated
from ...utils import is_torchdynamo_compiling, logging
from .configuration_nemotron_h_dense import NemotronHDenseConfig


logger = logging.get_logger(__name__)

is_fast_path_available = False


class NemotronHDenseRMSNorm(LlamaRMSNorm):
    pass


class NemotronHDenseMLP(NemotronMLP):
    def __init__(self, config, intermediate_size: int | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]


class NemotronHDenseMamba(Zamba2MambaMixer):
    def __init__(self, config: NemotronHDenseConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]
        self.use_mem_eff_path = True

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.num_heads = config.mamba_num_heads

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)
        self.norm = Zamba2RMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // self.n_groups, eps=config.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(
        self,
        hidden_states,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type and not is_torchdynamo_compiling():
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
        return self.torch_forward(hidden_states, cache_params, attention_mask)


class NemotronHDenseAttention(JambaAttention):
    pass


class NemotronHDenseMambaDecoderLayer(JambaMambaDecoderLayer):
    def __init__(self, config: NemotronHDenseConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.mamba = NemotronHDenseMamba(config, layer_idx=layer_idx)
        self.feed_forward = NemotronHDenseMLP(config)
        self.input_layernorm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_ff_layernorm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class NemotronHDenseAttentionDecoderLayer(JambaAttentionDecoderLayer):
    """Hybrid decoder layer: norm → self-attention → residual → norm → mlp → residual."""

    def __init__(self, config: NemotronHDenseConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.self_attn = NemotronHDenseAttention(config, layer_idx)
        self.feed_forward = NemotronHDenseMLP(config)
        self.input_layernorm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_ff_layernorm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


ALL_DECODER_LAYER_TYPES = {
    "mamba": NemotronHDenseMambaDecoderLayer,
    "attention": NemotronHDenseAttentionDecoderLayer,
}


class NemotronHDensePreTrainedModel(JambaPreTrainedModel):
    config: NemotronHDenseConfig
    base_model_prefix = "model"
    _no_split_modules = ["NemotronHDenseMambaDecoderLayer", "NemotronHDenseAttentionDecoderLayer"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": [NemotronHDenseMambaDecoderLayer, NemotronHDenseAttentionDecoderLayer],
        "attentions": NemotronHDenseAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        # Skip Jamba's _init_weights (references JambaMambaMixer / JambaExperts); run nn-level init directly.
        from ...modeling_utils import PreTrainedModel  # local import to avoid cycles in modular converter

        PreTrainedModel._init_weights(self, module)
        if isinstance(module, NemotronHDenseMamba):
            A = torch.arange(1, self.config.mamba_num_heads + 1)
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)

            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name == "out_proj.weight":
                    init.kaiming_uniform_(p, a=math.sqrt(5))
                    p_new = p / math.sqrt(self.config.num_hidden_layers)
                    init.copy_(p, p_new)


class NemotronHDenseModel(JambaModel):
    def __init__(self, config: NemotronHDenseConfig):
        JambaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                ALL_DECODER_LAYER_TYPES[config.layer_types[i]](config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()


class NemotronHDenseForCausalLM(ZambaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NemotronHDenseConfig):
        JambaPreTrainedModel.__init__(self, config)
        self.model = NemotronHDenseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "NemotronHDensePreTrainedModel",
    "NemotronHDenseModel",
    "NemotronHDenseForCausalLM",
]
