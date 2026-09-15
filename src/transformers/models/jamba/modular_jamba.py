# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GenericForSequenceClassification, GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..falcon_mamba.modeling_falcon_mamba import FalconMambaMixer
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm, eager_attention_forward
from ..mamba.modeling_mamba import (
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    mamba_selective_scan,
    mamba_selective_state_update,
)
from ..mistral.modeling_mistral import MistralMLP
from ..mixtral.modeling_mixtral import MixtralExperts, MixtralForCausalLM
from .configuration_jamba import JambaConfig


logger = logging.get_logger(__name__)


class JambaRMSNorm(LlamaRMSNorm):
    pass


class JambaAttention(LlamaAttention):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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


class JambaMambaMixer(FalconMambaMixer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
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

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependent
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        self.use_mambapy = config.use_mambapy
        self.use_associative_scan = config.use_associative_scan

    def init_jamba_weights(self):
        raise NotImplementedError("Not needed for jamba")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # Key difference to falcon mamba: mamba inner fn applies its dt norm at a different time
        # leading to incompatible implementations
        A = -torch.exp(self.A_log.float())

        hidden_states_B_C, gate = projected_states.chunk(2, dim=1)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # 2. Convolution sequence transformation
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
        else:
            if cache_params is not None:
                hidden_states_B_C = cache_params.update_conv_state(
                    hidden_states_B_C,
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )

            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                hidden_states_B_C = hidden_states_B_C[:, :, -seq_len:]

        # 3. SSM transformation
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C.transpose(1, 2), attention_mask)
        time_step, B, C = torch.split(
            self.x_proj(hidden_states_B_C),
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        # Key difference to (falcon) mamba: Additional norms on B, C, and dt
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # In case the model has been quantized, we need a hack to properly call the `nn.Linear` module
        # at the price of a small overhead.
        if hasattr(self.config, "_is_quantized"):
            time_step = (self.dt_proj(time_step) - self.dt_proj.bias).transpose(1, 2)
        else:
            time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        time_proj_bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None

        # Recurrent form
        if use_precomputed_states and seq_len == 1:
            scan_output = mamba_selective_state_update(
                recurrent_state,
                hidden_states_B_C.transpose(1, 2)[..., 0],
                time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                z=gate[..., 0],
                dt_bias=time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)

        # Full sequence form
        else:
            output_final_state = cache_params is not None
            scan_result = mamba_selective_scan(
                hidden_states_B_C.transpose(1, 2),
                time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                D=self.D.float(),
                z=gate,
                delta_bias=time_proj_bias,
                delta_softplus=True,
                return_last_state=output_final_state,
                use_mambapy=self.use_mambapy,
                use_associative_scan=self.use_associative_scan,
            )

            if output_final_state:
                scan_output, final_state = scan_result
                cache_params.update_recurrent_state(final_state, self.layer_idx)
            else:
                scan_output = scan_result

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2).to(dtype))
        return contextualized_states


class JambaMLP(MistralMLP):
    pass


class JambaExperts(MixtralExperts):
    pass


class JambaSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = JambaExperts(config)

    def route_tokens_to_experts(self, hidden_states, router_logits):
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
        return top_k_index, top_k_weights.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(hidden_states, router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class JambaAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx] if config.layers_num_experts else 1
        self.self_attn = JambaAttention(config, layer_idx)

        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class JambaMambaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx] if config.layers_num_experts else 1
        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)
        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


ALL_DECODER_LAYER_TYPES = {"attention": JambaAttentionDecoderLayer, "mamba": JambaMambaDecoderLayer}


class JambaPreTrainedModel(PreTrainedModel):
    config: JambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _is_stateful = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": [JambaAttentionDecoderLayer, JambaMambaDecoderLayer],
        "attentions": JambaAttention,
        "router_logits": OutputRecorder(nn.Linear, layer_name="router"),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, JambaMambaMixer):
            A = torch.arange(1, module.ssm_state_size + 1)[None, :]
            A = A.expand(module.intermediate_size, -1).contiguous()
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)
        elif isinstance(module, JambaExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class JambaModel(JambaPreTrainedModel):
    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            decoder_layers.append(layer_class(config, layer_idx=i))
        self.layers = nn.ModuleList(decoder_layers)

        self.final_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class JambaForCausalLM(MixtralForCausalLM):
    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.num_experts = config.num_experts

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, JambaForCausalLM

        >>> model = JambaForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            logits_to_keep,
            **kwargs,
        )


class JambaForSequenceClassification(GenericForSequenceClassification, JambaPreTrainedModel):
    pass


__all__ = ["JambaForCausalLM", "JambaForSequenceClassification", "JambaModel", "JambaPreTrainedModel"]
