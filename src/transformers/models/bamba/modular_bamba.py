# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Bamba model."""

from typing import TypedDict

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults, no_inherit_decorator
from ...utils.output_capturing import capture_outputs
from ..jamba.modeling_jamba import JambaAttentionDecoderLayer
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    rotate_half,
)
from ..mamba2.modeling_mamba2 import (
    Mamba2Mixer,
    MambaRMSNormGated,
)
from .configuration_bamba import BambaConfig


logger = logging.get_logger(__name__)


class BambaFlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for advanced Flash Attention, causal-conv1d, and mamba_ssm kernel usage.
    Use cases include padding-free training and fewer `torch.compile` graph breaks.

    cu_seq_lens_q (`torch.LongTensor`):
        Gets cumulative sequence length for query state.
    cu_seq_lens_k (`torch.LongTensor`):
        Gets cumulative sequence length for key state.
    max_length_q (`int`):
        Maximum sequence length for query state.
    max_length_k (`int`):
        Maximum sequence length for key state.
    seq_idx (`torch.IntTensor`):
        Index of each packed sequence.
    """

    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor


class BambaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


@no_inherit_decorator
class BambaAttention(LlamaAttention):
    pass


class BambaRMSNormGated(MambaRMSNormGated):
    pass


class BambaMixer(Mamba2Mixer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)

    The are a few differences between this and Mamba2Mixer:
    - The variable use_precomputed_states is slightly different due to the hybrid cache structure
    - Some extra variables that our layer doesn't need have been removed
    """

    def __init__(self, config: BambaConfig, layer_idx: int, initialize_mixer_weights: bool = True):
        self.use_bias = config.mamba_proj_bias
        super().__init__(config, layer_idx, initialize_mixer_weights)
        self.num_heads = config.mamba_n_heads
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.use_conv_bias = config.mamba_conv_bias
        self.layer_norm_epsilon = config.rms_norm_eps
        self.n_groups = config.mamba_n_groups
        self.head_dim = config.mamba_d_head
        self.chunk_size = config.mamba_chunk_size
        self.norm = BambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.mamba_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=self.use_bias,
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)
        del self.time_step_floor
        del self.time_step_rank
        del self.use_bias
        del self.time_step_min
        del self.time_step_max

    @torch.no_grad()
    def init_bamba_weights(self):
        A = torch.arange(1, self.num_heads + 1, device=self.A_log.device, dtype=torch.float32)
        init.copy_(self.A_log, torch.log(A))
        init.ones_(self.D)
        init.ones_(self.dt_bias)


class BambaMLP(LlamaMLP):
    pass


class BambaRMSNorm(LlamaRMSNorm):
    pass


class BambaDecoderLayer(JambaAttentionDecoderLayer):
    def __init__(self, config: BambaConfig, layer_idx: int, layer_type: str = "linear_attention"):
        super().__init__(config, layer_idx)

        del self.self_attn

        num_experts = 1
        ffn_layer_class = BambaMLP if num_experts == 1 else None
        self.feed_forward = ffn_layer_class(config)

        self.block_type = layer_type
        if layer_type == "linear_attention":
            self.mamba = BambaMixer(config=config, layer_idx=layer_idx)
        elif layer_type == "full_attention":
            self.self_attn = BambaAttention(config, layer_idx)
        else:
            raise ValueError(f"Invalid layer_type: {layer_type!r}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[BambaFlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.mamba(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
            self_attn_weights = None
        elif self.block_type == "full_attention":
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights


@auto_docstring
class BambaPreTrainedModel(PreTrainedModel):
    config: BambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BambaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _is_stateful = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": BambaDecoderLayer,
        "attentions": BambaAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BambaMixer):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.log(torch.arange(1, module.num_heads + 1)))
            init.ones_(module.D)


@auto_docstring
class BambaModel(BambaPreTrainedModel):
    def __init__(self, config: BambaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(BambaDecoderLayer(config, layer_idx=i, layer_type=config.layers_block_type[i]))
        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = BambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BambaRotaryEmbedding(config=config)

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
        **kwargs: Unpack[BambaFlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

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

        for i, decoder_layer in enumerate(self.layers):
            hidden_states, attn_weights = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layers_block_type[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class BambaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.z_loss_coefficient = config.z_loss_coefficient

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BambaForCausalLM

        >>> model = BambaForCausalLM.from_pretrained("...")
        >>> tokenizer = AutoTokenizer.from_pretrained("...")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if self.z_loss_coefficient > 0:
                z_loss = logits.logsumexp(dim=-1).to(dtype=loss.dtype).pow(2).mean()
                loss = loss + self.z_loss_coefficient * z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        kwargs["logits_to_keep"] = self.config.num_logits_to_keep
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        return model_inputs


__all__ = ["BambaModel", "BambaForCausalLM", "BambaPreTrainedModel"]
