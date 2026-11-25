# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Zamba model."""

import math
from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations import use_kernelized_func
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..jamba.modeling_jamba import JambaAttention
from ..llama.modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification, eager_attention_forward
from ..mamba.modeling_mamba import (
    MambaRMSNorm,
    causal_conv1d_fn,
    causal_conv1d_forward,
    causal_conv1d_update,
    mamba_selective_scan,
    mamba_selective_state_update,
)
from ..mamba2.modeling_mamba2 import apply_mask_to_padding_states
from ..mistral.modeling_mistral import MistralMLP
from .configuration_zamba import ZambaConfig


logger = logging.get_logger(__name__)


class ZambaRMSNorm(MambaRMSNorm):
    pass


class ZambaAttention(JambaAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".

    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
    The input dimension here is attention_hidden_size = 2 * hidden_size, and head_dim = attention_hidden_size // num_heads.
    The extra factor of 2 comes from the input being the concatenation of original_hidden_states with the output of the previous (mamba) layer
    (see fig. 2 in https://huggingface.co/papers/2405.16712).
    Additionally, replaced
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) with
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim/2)
    """

    def __init__(self, config: ZambaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_hidden_size = config.attention_hidden_size
        self.scaling = (self.head_dim / 2) ** -0.5
        self.q_proj = nn.Linear(config.attention_hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=False)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if cache is not None:
            # Diff from Jamba: one attention block is shared across layers, so `layer_idx` comes from the caller
            key_states, value_states = cache.update(key_states, value_states, layer_idx, kwargs=kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernelized_func(
    [mamba_selective_scan, mamba_selective_state_update, causal_conv1d_forward, causal_conv1d_fn, causal_conv1d_update]
)
class ZambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)

    This module differs from `transformers.models.mamba.modeling_mamba.MambaMixer` in two ways:
    - Added multi-head: the output of `self.in_proj` is split into `self.n_mamba_heads` heads, and each head
    undergoes an independent forward pass, identical to the original `MambaMixer`, up until the pre-activations of
    `self.out_proj`. The pre-activations, coming from different mamba heads, are then concatenated and fed into `self.out_proj`.
    """

    def __init__(self, config: ZambaConfig, layer_idx):
        super().__init__()
        if not config.use_mamba_kernels:
            logger.warning_once(
                "`use_mamba_kernels=False` is deprecated and has no effect. The implementation is selected "
                "automatically: Hub kernels when loading with `use_kernels=True`, otherwise the `mamba-ssm` and "
                "`causal-conv1d` packages if installed, otherwise the PyTorch implementation."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.n_mamba_heads = config.n_mamba_heads
        self.mamba_head_dim = self.intermediate_size // self.n_mamba_heads
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_mamba_act
        self.act = ACT2FN[config.hidden_mamba_act]

        self.use_associative_scan = config.use_associative_scan

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # weight associated to the selective projection used to make dt, B and C input dependent
        # each mamba head is processed independently
        self.x_proj_weight = nn.Parameter(
            torch.zeros(
                self.n_mamba_heads,
                self.time_step_rank + self.ssm_state_size * 2,
                self.mamba_head_dim,
            )
        )
        # time step projection (discretization)
        self.dt_proj_weight = nn.Parameter(
            (torch.zeros(self.n_mamba_heads, self.mamba_head_dim, self.time_step_rank) - 0.5)
            * 2
            / self.time_step_rank**0.5
        )
        self.dt_proj_bias = nn.Parameter(torch.zeros(self.n_mamba_heads, self.mamba_head_dim))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A).reshape(self.n_mamba_heads, self.mamba_head_dim, -1))
        self.D = nn.Parameter(torch.ones(self.n_mamba_heads, self.mamba_head_dim))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv1d")
    @deprecate_kwarg("cache_params", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        # Retrieve the recurrent state if there is a cache. It may be None if the layer is not initialized.
        if cache is not None:
            recurrent_state = cache.get_recurrent_state(self.layer_idx, state_idx=0)
        else:
            recurrent_state = None

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # Key difference: Split the projection into independent Mamba heads
        hidden_states_B_C, gate = projected_states.view(batch_size, -1, 2, seq_len).chunk(2, dim=2)
        hidden_states_B_C = hidden_states_B_C.squeeze(2).contiguous()
        gate = gate.reshape(batch_size, self.n_mamba_heads, self.mamba_head_dim, seq_len).transpose(0, 1)

        # 2. Convolution sequence transformation
        hidden_states_B_C = causal_conv1d_forward(
            self.conv1d,
            hidden_states_B_C,
            cache,
            self.layer_idx,
            activation=self.activation,
            **kwargs,
        )

        # 3. SSM transformation
        hidden_states_B_C = apply_mask_to_padding_states(
            hidden_states_B_C.transpose(1, 2),
            attention_mask,
        )
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)
        hidden_states_B_C = hidden_states_B_C.reshape(
            batch_size,
            self.n_mamba_heads,
            self.mamba_head_dim,
            seq_len,
        ).transpose(0, 1)

        # Key difference: x_proj and dt_proj have separate weights for each Mamba head
        ssm_parameters = (self.x_proj_weight[:, None] @ hidden_states_B_C).transpose(-1, -2)
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        time_step = self.dt_proj_weight[:, None] @ time_step.transpose(-1, -2)
        time_proj_bias = self.dt_proj_bias.float() if self.dt_proj_bias is not None else None
        A = -torch.exp(self.A_log.float())

        # Key difference: per head scans
        # Recurrent form
        if recurrent_state is not None and seq_len == 1:
            scan_outputs = []

            for head_idx in range(self.n_mamba_heads):
                scan_output = mamba_selective_state_update(
                    recurrent_state[:, head_idx],
                    hidden_states_B_C[head_idx, ..., 0],
                    time_step[head_idx, ..., 0],
                    A[head_idx],
                    B[head_idx, :, 0],
                    C[head_idx, :, 0],
                    self.D[head_idx],
                    z=gate[head_idx, ..., 0],
                    dt_bias=time_proj_bias[head_idx] if time_proj_bias is not None else None,
                    dt_softplus=True,
                ).unsqueeze(-1)
                scan_outputs.append(scan_output)

            scan_output = torch.cat(scan_outputs, dim=1)

        # Full sequence form
        else:
            output_final_state = cache is not None
            scan_outputs = []
            final_states = []

            for head_idx in range(self.n_mamba_heads):
                scan_result = mamba_selective_scan(
                    hidden_states_B_C[head_idx],
                    time_step[head_idx],
                    A[head_idx],
                    B[head_idx].transpose(1, 2),
                    C[head_idx].transpose(1, 2),
                    D=self.D[head_idx].float(),
                    z=gate[head_idx],
                    delta_bias=time_proj_bias[head_idx] if time_proj_bias is not None else None,
                    delta_softplus=True,
                    return_last_state=output_final_state,
                    # Old model: only when user request it explicitly
                    use_mambapy=False,
                    use_associative_scan=self.use_associative_scan,
                )

                if output_final_state:
                    head_output, final_state = scan_result
                    scan_outputs.append(head_output)
                    final_states.append(final_state)
                else:
                    scan_outputs.append(scan_result)

            scan_output = torch.cat(scan_outputs, dim=1)

            if output_final_state:
                final_state = torch.stack(final_states, dim=1)
                cache.update_recurrent_state(final_state, self.layer_idx)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2).to(dtype))
        return contextualized_states


class ZambaMLP(MistralMLP):
    pass


class ZambaAttentionDecoderLayer(nn.Module):
    def __init__(self, config: ZambaConfig, layer_idx: int | None = None):
        super().__init__()
        self.self_attn = ZambaAttention(config, layer_idx)

        self.feed_forward = ZambaMLP(config)
        self.input_layernorm = ZambaRMSNorm(config.attention_hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = ZambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            hidden_states (`torch.Tensor`): the output of the previous Mamba layer concatenated with the word
                embedding output, of shape `(batch, seq_len, 2 * embed_dim)` (see fig. 2 in
                https://huggingface.co/papers/2405.16712).
            layer_idx (`int`): layer_idx in the forward pass. Used to distinguish Zamba's tied transformer layers.
            attention_mask (`torch.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            cache (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
        """
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
            **kwargs,
        )
        # feed-forward (MLP)
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        return hidden_states


class ZambaMambaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ZambaConfig, layer_idx: int):
        super().__init__()
        self.mamba = ZambaMambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = ZambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        transformer_hidden_states: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            cache (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
        """

        residual = hidden_states

        # `transformer_hidden_states` is the output from shared transformer + linear layer (see fig. 2 in https://huggingface.co/papers/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba layer below (as described in eq. (6) of https://huggingface.co/papers/2405.16712).
        hidden_states = (
            hidden_states + transformer_hidden_states if transformer_hidden_states is not None else hidden_states
        )
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache=cache,
            attention_mask=attention_mask,
            **kwargs,
        )
        # residual connection after mamba
        hidden_states = residual + hidden_states

        return hidden_states


class ZambaHybridLayer(GradientCheckpointingLayer):
    def __init__(self, shared_transf: ZambaAttentionDecoderLayer, linear: nn.Linear, mamba: ZambaMambaDecoderLayer):
        super().__init__()
        self.shared_transf = shared_transf
        self.linear = linear
        self.mamba_decoder = mamba

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.Tensor`): word embedding output that will be concatenated with
            hidden activations to form the input of the shared transformer layer.
            layer_idx (`int`): layer number.
            attention_mask (`torch.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            cache (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
        """

        transformer_hidden_states = self.shared_transf(
            torch.concatenate([hidden_states, original_hidden_states], dim=-1),
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            cache=cache,
            use_cache=use_cache,
            **kwargs,
        )
        transformer_hidden_states = self.linear(transformer_hidden_states)

        hidden_states = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
            **kwargs,
        )

        return hidden_states


@auto_docstring
class ZambaPreTrainedModel(PreTrainedModel):
    config: ZambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ZambaHybridLayer", "ZambaMambaDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": ZambaMambaDecoderLayer,
        "attentions": ZambaAttention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        super()._init_weights(module)
        if isinstance(module, ZambaMambaMixer):
            init.normal_(module.x_proj_weight, mean=0.0, std=std)
            dt_init_std = self.config.mamba_dt_rank**-0.5
            init.uniform_(module.dt_proj_weight, -dt_init_std, dt_init_std)

            mamba_head_dim = self.config.mamba_expand * self.config.hidden_size // self.config.n_mamba_heads
            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads, mamba_head_dim)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_proj_bias, inv_dt)

            A = torch.arange(1, module.ssm_state_size + 1, dtype=torch.float32)[None, :]
            A = A.expand(module.intermediate_size, -1).contiguous()
            init.copy_(module.A_log, torch.log(A).reshape(module.n_mamba_heads, module.mamba_head_dim, -1))
            init.ones_(module.D)


@auto_docstring
class ZambaModel(ZambaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ZambaDecoderLayer`]

    Args:
        config: ZambaConfig
    """

    def __init__(self, config: ZambaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers_block_type = config.layers_block_type
        layers = []
        self._tied_weights_keys = None
        for layer_id, layer_type in enumerate(self.layers_block_type):
            mamba = ZambaMambaDecoderLayer(config, layer_idx=layer_id)
            if layer_type == "hybrid":
                linear = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
                layers.append(ZambaHybridLayer(ZambaAttentionDecoderLayer(config), linear, mamba))
                if self._tied_weights_keys is None:
                    self._tied_weights_keys = {
                        rf"layers.(?!{layer_id}\.)\d+.shared_transf": f"layers.{layer_id}.shared_transf"
                    }
            else:
                layers.append(mamba)
        self.layers = nn.ModuleList(layers)

        self.final_layernorm = ZambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        original_hidden_states = torch.clone(inputs_embeds)
        # original_hidden_states: word embedding output that will be concatenated with hidden activations to form the input of the shared transformer layer

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                original_hidden_states,
                layer_idx,
                causal_mask_mapping["linear_attention"],
                causal_mask_mapping["full_attention"],
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache if use_cache else None,
        )


class ZambaForCausalLM(LlamaForCausalLM):
    _tp_plan = AttributeError()

    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, ZambaForCausalLM

        >>> model = ZambaForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1")
        >>> tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        kwargs["logits_to_keep"] = self.config.num_logits_to_keep
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        return model_inputs


class ZambaForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = ["ZambaForCausalLM", "ZambaForSequenceClassification", "ZambaModel", "ZambaPreTrainedModel"]
