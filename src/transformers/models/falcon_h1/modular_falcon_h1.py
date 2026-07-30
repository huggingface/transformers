# Copyright 2025 Technology Innovation Institute and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch FalconH1 model."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bamba.modeling_bamba import BambaMixer
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mamba2.modeling_mamba2 import (
    MambaRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from .configuration_falcon_h1 import FalconH1Config


logger = logging.get_logger(__name__)


class FalconH1RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class FalconH1Attention(LlamaAttention):
    def __init__(self, config: FalconH1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.key_multiplier = config.key_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) * self.key_multiplier
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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


class FalconH1RMSNormGated(MambaRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, dim = hidden_states.shape
        else:
            batch_size, dim = hidden_states.shape
            seq_len = 1
        hidden_states = hidden_states.to(torch.float32)

        hidden_states = hidden_states.view(batch_size, seq_len, self.n_groups, int(dim // self.n_groups))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight.view(self.n_groups, int(dim // self.n_groups)) * hidden_states
        hidden_states = hidden_states.view(batch_size, seq_len, dim)

        if seq_len == 1:
            hidden_states = hidden_states.squeeze(1)

        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class FalconH1Mixer(BambaMixer):
    """
    FalconH1Mixer is identical to classic Mamba2 mixer classes but differs on two different things
    - Users can pass custom intermediate_size through `config.mamba_d_ssm`
    - The use of gated RMS normalization layer is optional
    """

    def __init__(self, config: FalconH1Config, layer_idx: int, initialize_mixer_weights: bool = True):
        super().__init__(config, layer_idx, initialize_mixer_weights)
        self.intermediate_size = (
            int(config.mamba_expand * self.hidden_size) if config.mamba_d_ssm is None else config.mamba_d_ssm
        )
        self.groups_time_state_size = config.mamba_n_groups * self.ssm_state_size
        self.mamba_rms_norm = config.mamba_rms_norm
        self.norm = (
            FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )
            if config.mamba_rms_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=config.projectors_bias)
        self.ssm_in_multiplier = config.ssm_in_multiplier

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        # Key difference 1: Additional Multipliers
        hidden_states = hidden_states * self.ssm_in_multiplier
        projected_states = self.in_proj(hidden_states)
        projected_states = projected_states * self.mup_vector

        A = -torch.exp(self.A_log.float())
        fused_kwargs = (
            kwargs | {} if self.time_step_limit == (0.0, float("inf")) else kwargs | {"dt_limit": self.time_step_limit}
        )
        if self.training and cache_params is None:
            fused_output = falcon_h1_split_conv1d_scan_combined(  # noqa F821
                projected_states,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.mamba_rms_norm else None,
                rmsnorm_eps=self.norm.variance_epsilon if self.mamba_rms_norm else None,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.head_dim,
                ngroups=self.n_groups,
                norm_before_gate=False,
                return_final_states=False,
                **fused_kwargs,
            )

            # Only kernels can use this shortcircuit, fallback to normal torch otherwise
            if fused_output is not None:
                return fused_output

        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # 2. Convolution sequence transformation
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)
        if use_precomputed_states and seq_len == 1:
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
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # Recurrent form
        if use_precomputed_states and seq_len == 1:
            hidden_states = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            dt = dt.transpose(1, 2).expand(-1, -1, self.head_dim)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            B = B.view(batch_size, self.n_groups, B.shape[2] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[2] // self.n_groups)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)

            scan_output = falcon_h1_selective_state_update(  # noqa F821
                recurrent_state,
                hidden_states,
                dt,
                A,
                B,
                C,
                D,
                # Key difference 2: Potential z gate into kernel
                z=gate.view(batch_size, self.num_heads, self.head_dim) if not self.mamba_rms_norm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            scan_output = scan_output.view(batch_size, 1, self.num_heads * self.head_dim)

            # Key difference 3: Norm based handling as optional between z and norm
            if self.mamba_rms_norm:
                scan_output = self.norm(scan_output, gate)

        # Chunk form
        else:
            output_final_state = cache_params is not None
            scan_result = falcon_h1_chunk_scan(  # noqa F821
                hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                dt,
                A,
                B.view(batch_size, seq_len, self.n_groups, -1),
                C.view(batch_size, seq_len, self.n_groups, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                return_final_states=output_final_state,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                initial_states=recurrent_state if use_precomputed_states else None,
                dt_limit=self.time_step_limit,
                **kwargs,
            )

            if output_final_state:
                scan_output, ssm_state = scan_result
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)
            else:
                scan_output = scan_result

            scan_output = scan_output.view(batch_size, seq_len, -1)

            # Key difference 3: Norm based handling as optional between z and norm
            if self.mamba_rms_norm:
                scan_output = self.norm(scan_output, gate)
            else:
                scan_output = scan_output * torch.nn.functional.silu(gate)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


class FalconH1MLP(LlamaMLP):
    def __init__(self, config: FalconH1Config):
        super().__init__(config)
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def forward(self, x):
        y = self.up_proj(x) * self.act_fn(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1RMSNorm(LlamaRMSNorm):
    pass


class FalconH1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FalconH1Config, layer_idx: int):
        super().__init__()
        self.feed_forward = FalconH1MLP(config)

        head_dim = config.hidden_size // config.num_attention_heads
        self.channels_attn = config.num_attention_heads * head_dim + 2 * config.num_key_value_heads * head_dim

        self.mamba = FalconH1Mixer(config=config, layer_idx=layer_idx)

        self.self_attn = FalconH1Attention(config, layer_idx)

        self.attention_in_multiplier = config.attention_in_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.input_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mamba_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=mamba_attention_mask,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        attention_hidden_states, _ = self.self_attn(
            hidden_states=hidden_states * self.attention_in_multiplier,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

        hidden_states = mamba_hidden_states + attention_hidden_states

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


@auto_docstring
class FalconH1PreTrainedModel(PreTrainedModel):
    config: FalconH1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FalconH1DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _is_stateful = True
    _can_compile_fullgraph = False  # StaticCache has no entry for ``"hybrid"`` layers (KV + SSM state).

    _can_record_outputs = {
        "hidden_states": FalconH1DecoderLayer,
        "attentions": FalconH1Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FalconH1Mixer):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.log(torch.arange(1, module.num_heads + 1)))
            init.ones_(module.D)
        elif isinstance(module, FalconH1Model):
            mup_vector = compute_mup_vector(module.config)
            for layer in module.layers:
                init.copy_(layer.mamba.mup_vector, mup_vector)


def compute_mup_vector(config):
    """
    Computes the MuP vector based on model configuration.

    FalconH1 applies different MuP multiplier for each dimension of the hidden states.
    The MuP vector is partitioned into chunks, and each chunk is multiplied with its
    corresponding projected dimension.

    Args:
        config: FalconH1Config object

    Returns:
        torch.Tensor: The computed MuP vector
    """
    # We'll need some values from the config to compute the vector dimensions
    intermediate_size = (
        config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
    )
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    zxbcdt_multipliers = config.ssm_multipliers

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    mup_vector = torch.ones(1, 1, vector_shape)

    # Apply multipliers to different sections of the vector
    mup_vector[:, :, :intermediate_size] *= zxbcdt_multipliers[0]
    mup_vector[:, :, intermediate_size : 2 * intermediate_size] *= zxbcdt_multipliers[1]
    mup_vector[:, :, 2 * intermediate_size : 2 * intermediate_size + groups_time_state_size] *= zxbcdt_multipliers[2]
    mup_vector[
        :, :, 2 * intermediate_size + groups_time_state_size : 2 * intermediate_size + 2 * groups_time_state_size
    ] *= zxbcdt_multipliers[3]
    mup_vector[:, :, 2 * intermediate_size + 2 * groups_time_state_size :] *= zxbcdt_multipliers[4]

    return mup_vector


@auto_docstring
# Adapted from transformers.models.jamba.modeling_jamba.JambaModel
class FalconH1Model(FalconH1PreTrainedModel):
    def __init__(self, config: FalconH1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(FalconH1DecoderLayer(config, layer_idx=i))
        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = FalconH1RotaryEmbedding(config=config)

        self.embedding_multiplier = config.embedding_multiplier
        self.lm_head_multiplier = config.lm_head_multiplier

        self.gradient_checkpointing = False
        # Compute the MuP vector once and register it for all layers
        mup_vector = compute_mup_vector(config)
        for layer in self.layers:
            layer.mamba.register_buffer("mup_vector", mup_vector.clone(), persistent=False)

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
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier
        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                mamba_attention_mask=causal_mask_mapping["linear_attention"],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FalconH1ForCausalLM(LlamaForCausalLM):
    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **_):
        # Every FalconH1 decoder layer is hybrid (attention + mamba in the same block), so the layer-type
        # dispatch table can't enumerate sub-patterns. We return both masks the layer needs as a dict.
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }

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
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, FalconH1ForCausalLM

        >>> model = FalconH1ForCausalLM.from_pretrained("...")
        >>> tokenizer = AutoTokenizer.from_pretrained("...")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.model.lm_head_multiplier

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        kwargs["logits_to_keep"] = self.config.num_logits_to_keep
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        return model_inputs


__all__ = ["FalconH1Model", "FalconH1ForCausalLM", "FalconH1PreTrainedModel"]
