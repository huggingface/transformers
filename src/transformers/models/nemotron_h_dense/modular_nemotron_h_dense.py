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
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.jamba.modeling_jamba import JambaAttention
from ...models.llama.modeling_llama import LlamaRMSNorm
from ...models.nemotron.modeling_nemotron import NemotronMLP
from ...models.zamba.modeling_zamba import ZambaForCausalLM
from ...models.zamba2.modeling_zamba2 import Zamba2MambaMixer, Zamba2RMSNormGated
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_nemotron_h_dense import NemotronHDenseConfig


logger = logging.get_logger(__name__)

is_fast_path_available = False


class NemotronHDenseMamba2Mixer(Zamba2MambaMixer):
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

        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )

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


class NemotronHDenseRMSNorm(LlamaRMSNorm):
    pass


class NemotronHDenseMLP(NemotronMLP):
    def __init__(self, config, intermediate_size=None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]


class NemotronHDenseAttention(JambaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return super().forward(hidden_states, attention_mask, past_key_values, **kwargs)


MIXER_TYPES = {
    "mamba": NemotronHDenseMamba2Mixer,
    "attention": NemotronHDenseAttention,
    "mlp": NemotronHDenseMLP,
}


class NemotronHDenseBlock(GradientCheckpointingLayer):
    """
    A single transformer block in the dense Nemotron-H model. Each block holds one of
    Mamba, Attention, or MLP mixer, applies pre-normalization then the mixer, and adds
    a residual connection.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layer_types[layer_idx]

        mixer_kwargs = {"config": config}
        if self.block_type != "mlp":
            mixer_kwargs["layer_idx"] = layer_idx
        self.mixer = MIXER_TYPES[self.block_type](**mixer_kwargs)

    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        elif self.block_type == "attention":
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                user_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class NemotronHDensePreTrainedModel(PreTrainedModel):
    config: NemotronHDenseConfig
    base_model_prefix = "model"
    _no_split_modules = ["NemotronHDenseBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": NemotronHDenseBlock,
        "attentions": NemotronHDenseAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NemotronHDenseMamba2Mixer):
            A = torch.arange(1, self.config.mamba_num_heads + 1)
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)

            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                init.copy_(module.dt_bias, inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name == "out_proj.weight":
                    init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p_new = p / math.sqrt(self.config.num_hidden_layers)
                        init.copy_(p, p_new)


class NemotronHDenseModel(NemotronHDensePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHDenseBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.norm_f = NemotronHDenseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        hidden_states = inputs_embeds

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, past_key_values)

        block_type_to_mask = {
            "mamba": mamba_mask,
            "attention": causal_mask,
        }

        for layer_idx, mixer_block in enumerate(self.layers):
            # Parameter-free blocks (mlp / moe) do not need a mask.
            layer_mask = block_type_to_mask.get(mixer_block.block_type)

            hidden_states = mixer_block(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm_f(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _update_mamba_mask(self, attention_mask, past_key_values):
        mamba_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            mamba_mask = None
        return mamba_mask


class NemotronHDenseForCausalLM(ZambaForCausalLM):
    _tied_weights_keys = {}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "NemotronHDensePreTrainedModel",
    "NemotronHDenseModel",
    "NemotronHDenseForCausalLM",
]
