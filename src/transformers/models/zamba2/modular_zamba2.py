# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
#
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
import math
from collections.abc import Callable
from itertools import cycle

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bamba.modeling_bamba import BambaMixer
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..zamba.modeling_zamba import (
    ZambaAttention,
    ZambaAttentionDecoderLayer,
    ZambaForCausalLM,
    ZambaForSequenceClassification,
    ZambaHybridLayer,
    ZambaMambaDecoderLayer,
    ZambaModel,
    ZambaRMSNorm,
    eager_attention_forward,
)
from .configuration_zamba2 import Zamba2Config


logger = logging.get_logger(__name__)


class Zamba2RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, group_count * self.group_size)
        return self.weight * hidden_states.to(input_dtype)


class Zamba2RMSNorm(ZambaRMSNorm):
    pass


class Zamba2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Zamba2Attention(ZambaAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
    The input dimension here is attention_hidden_size = 2 * hidden_size, and head_dim = attention_hidden_size // num_heads.
    The extra factor of 2 comes from the input being the concatenation of original_hidden_states with the output of the previous (mamba) layer
    (see fig. 2 in https://huggingface.co/papers/2405.16712).
    Additionally, replaced
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) with
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim/2)
    Finally, this attention layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this
    layer is tied, un-tied adapters (formally the same as LoRA but used in the base model) modules are added to the q, k, v projectors to increase
    expressivity with a small memory overhead (see Fig. 2 of https://huggingface.co/papers/2411.15242).
    """

    def __init__(
        self,
        config: Zamba2Config,
        layer_idx: int | None = None,
        num_fwd_mem_blocks: int | None = None,
        block_id: int | None = None,
    ):
        super().__init__(config, layer_idx)
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.layer_block_map = config.hybrid_layer_ids
        self.block_id = block_id

        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    linear_q_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_k_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_v_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()
                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        self.layer_dic = {value: index for index, value in enumerate(self.layer_block_map)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.config.use_shared_attention_adapter:
            adapter_layer_idx = self.layer_dic[layer_idx]
            query_states = query_states + self.linear_q_adapter_list[adapter_layer_idx](hidden_states)
            key_states = key_states + self.linear_k_adapter_list[adapter_layer_idx](hidden_states)
            value_states = value_states + self.linear_v_adapter_list[adapter_layer_idx](hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

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


class Zamba2MambaMixer(BambaMixer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Zamba2Config, layer_idx: int | None = None, initialize_mixer_weights: bool = True):
        self.config = config
        super().__init__(config, layer_idx, initialize_mixer_weights)
        self.use_conv_bias = config.use_conv_bias
        self.activation = "silu"
        self.act = nn.SiLU()
        self.n_groups = config.mamba_ngroups
        self.head_dim = config.mamba_headdim
        self.num_heads = self.config.n_mamba_heads
        self.chunk_size = config.chunk_size
        # No upper limit
        self.time_step_limit = (config.time_step_min, float("inf"))

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.add_bias_linear,
        )
        self.norm = Zamba2RMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // self.n_groups, eps=1e-5
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)
        del self.layer_norm_epsilon
        del self.use_bias


class Zamba2MLP(nn.Module):
    def __init__(self, config: Zamba2Config, num_fwd_mem_blocks=None, block_id: int | None = None):
        """
        This MLP layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this layer
        is tied, un-tied adapter modules (formally same as LoRA, but used in the base model) are added to the up and gate projectors to increase expressivity with a small memory overhead.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.block_id = block_id

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.add_bias_linear)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for i in range(self.num_fwd_mem_blocks):
            if i % config.num_mem_blocks == block_id:
                gate_up_proj_adapter = nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.adapter_rank, bias=False),
                    nn.Linear(self.config.adapter_rank, 2 * self.intermediate_size, bias=False),
                )
            else:
                gate_up_proj_adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

        layer_block_map = config.hybrid_layer_ids
        self.layer_dic = {value: index for index, value in enumerate(layer_block_map)}

    def forward(self, hidden_state, layer_idx=None):
        gate_up_state = self.gate_up_proj(hidden_state)
        layer_idx = self.layer_dic[layer_idx]
        gate_up_state = gate_up_state + self.gate_up_proj_adapter_list[layer_idx](hidden_state)

        gate_up_state = torch.chunk(gate_up_state, 2, dim=-1)
        hidden_state = self.act_fn(gate_up_state[0]) * gate_up_state[1]
        output = self.down_proj(hidden_state)
        return output


class Zamba2AttentionDecoderLayer(ZambaAttentionDecoderLayer):
    def __init__(self, config: Zamba2Config, block_id: int | None = None, layer_idx: int | None = None):
        self.block_id = block_id
        num_gs = len(config.hybrid_layer_ids)
        super().__init__(config, layer_idx)
        self.self_attn = Zamba2Attention(config, layer_idx=-1, num_fwd_mem_blocks=num_gs, block_id=block_id)
        self.feed_forward = Zamba2MLP(config, num_fwd_mem_blocks=num_gs, block_id=block_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`.
                This is concatenated with `hidden_states` (which is the output of the previous (mamba) layer). The
                concatenated tensor is then used as input of the pre-attention RMSNorm
                (see fig. 2 in https://huggingface.co/papers/2405.16712).
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, layer_idx)

        return hidden_states


class Zamba2MambaDecoderLayer(ZambaMambaDecoderLayer):
    def __init__(self, config: Zamba2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mamba = Zamba2MambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Zamba2HybridLayer(ZambaHybridLayer):
    def __init__(
        self, shared_transformer: Zamba2AttentionDecoderLayer, linear: nn.Linear, mamba: Zamba2MambaDecoderLayer
    ):
        super().__init__(shared_transformer, linear, mamba)
        del self.shared_transf
        self.shared_transformer = shared_transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output that will be concatenated with
            hidden activations to form the input of the shared transformer layer.
            layer_idx (`int`): layer number.
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        transformer_hidden_states = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            **kwargs,
        )
        transformer_hidden_states = self.linear(transformer_hidden_states)

        hidden_states = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        return hidden_states


@auto_docstring
class Zamba2PreTrainedModel(PreTrainedModel):
    config: Zamba2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Zamba2HybridLayer", "Zamba2MambaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_sdpa = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": Zamba2MambaDecoderLayer,
        "attentions": Zamba2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Zamba2MambaMixer):
            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)

            A = torch.arange(1, module.num_heads + 1)
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)


class Zamba2Model(ZambaModel, Zamba2PreTrainedModel):
    """
    Model consisting of *config.num_hidden_layers* layers.

    Args:
        config: Zamba2Config
    """

    def __init__(self, config: Zamba2Config):
        Zamba2PreTrainedModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers_block_type = config.layers_block_type
        self.layers = self.get_layers()

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_mem_rope:
            if config.use_long_context:
                logger.warning_once(
                    "`use_long_context` set to `True`: using rescaled `rope_theta` and extended `max_position_embeddings`."
                )
            self.rotary_emb = Zamba2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_layers(self):
        layers = []
        self._tied_weights_keys = {}
        self.first_transformer_layer_id = 0
        unique_hybrid_blocks = []

        for layer_id, layer_type in enumerate(self.layers_block_type):
            mamba_layer = Zamba2MambaDecoderLayer(self.config, layer_idx=layer_id)
            if layer_type == "hybrid":
                prefix_pattern = f"layers.{layer_id}.shared_transformer"

                # Zamba ties Hybrid module weights by repeating blocks after every
                # `num_mem_blocks`. So if `num_mem_blocks=2`, the blocks looks like
                # [1, 2, 1, 2, 1, 2] where all "ones" share the same set of weights.
                if (
                    not isinstance(unique_hybrid_blocks, list)
                    or len(unique_hybrid_blocks) >= self.config.num_mem_blocks
                ):
                    if isinstance(unique_hybrid_blocks, list):
                        unique_hybrid_blocks = cycle(unique_hybrid_blocks)
                    target_pattern = next(unique_hybrid_blocks)
                    self._tied_weights_keys.update({prefix_pattern: target_pattern})
                else:
                    # Store source patterns to which the subsequent modules will be tied
                    unique_hybrid_blocks.append(prefix_pattern)

                block_id = layer_id % self.config.num_mem_blocks
                attn_block = Zamba2AttentionDecoderLayer(self.config, block_id=block_id)
                linear_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
                layers.append(Zamba2HybridLayer(attn_block, linear_layer, mamba_layer))
            else:
                layers.append(mamba_layer)
        return nn.ModuleList(layers)

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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        original_hidden_states = torch.clone(inputs_embeds)
        # original_hidden_states: word embedding output that will be concatenated with hidden activations to form the input of the shared transformer layer

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        # create position embeddings to be shared across the decoder layers
        if self.config.use_mem_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        else:
            position_embeddings = None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                original_hidden_states,
                layer_idx,
                causal_mask_mapping["linear_attention"],
                causal_mask_mapping["full_attention"],
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Zamba2ForCausalLM(ZambaForCausalLM):
    pass


class Zamba2ForSequenceClassification(ZambaForSequenceClassification):
    def __init__(self, config: Zamba2Config):
        super().__init__(config)
        self.model = Zamba2Model(config)
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=pooled_logits, labels=labels, pooled_logits=pooled_logits, config=self.config, **kwargs
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Zamba2Model",
    "Zamba2PreTrainedModel",
]
