# coding=utf-8
# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from ...utils import auto_docstring, can_return_tuple, logging
from ..bamba.configuration_bamba import BambaConfig
from ..bamba.modeling_bamba import BambaMixer, BambaRMSNormGated, HybridMambaAttentionDynamicCache
from ..granitemoeshared.modeling_granitemoeshared import (
    GraniteMoeSharedAttention,
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSharedMLP,
    GraniteMoeSharedModel,
    GraniteMoeSharedPreTrainedModel,
)
from .configuration_granitemoehybrid import GraniteMoeHybridConfig


logger = logging.get_logger(__name__)


class GraniteMoeHybridAttention(GraniteMoeSharedAttention):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class GraniteMoeHybridMambaLayer(BambaMixer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(BambaConfig(config), layer_idx)


class GraniteMoeHybridRMSNormGated(BambaRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class GraniteMoeHybridMLP(GraniteMoeSharedMLP):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)


class GraniteMoeHybridDecoderLayer(GraniteMoeSharedDecoderLayer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.shared_mlp = GraniteMoeHybridMLP(config)
        # Either attention or mamba will be initialized, depending on the layer type.
        self.self_attn = None
        self.mamba = None

        if config.layers_block_type[layer_idx] == "mamba":
            self.mamba = GraniteMoeHybridMambaLayer(config, layer_idx)
        else:
            self.self_attn = GraniteMoeHybridAttention(config, layer_idx)
        self.layer_type = config.layers_block_type[layer_idx]

        # Accept 0 experts: skip MoE if num_local_experts == 0
        self.has_experts = getattr(config, "num_local_experts", 0) > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.mamba is not None:
            hidden_states = self.mamba(
                hidden_states=hidden_states,
                cache_position=cache_position,
                cache_params=past_key_value,
                attention_mask=attention_mask,
            )
            # No attention weights for state space layers
            self_attn_weights = None
        else:
            hidden_states, self_attn_weights, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states * self.residual_multiplier

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.has_experts:
            moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
        else:
            hidden_states = self.shared_mlp(hidden_states)
            router_logits = None

        hidden_states = residual + hidden_states * self.residual_multiplier

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class GraniteMoeHybridPreTrainedModel(GraniteMoeSharedPreTrainedModel):
    config_class = GraniteMoeHybridConfig
    _no_split_modules = ["GraniteMoeHybridDecoderLayer"]
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights()
        # Initialize Mamba modules
        if isinstance(module, (nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, GraniteMoeHybridMambaLayer):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data = torch.log(torch.arange(1, module.num_heads + 1))
            module.D.data.fill_(1.0)
        elif isinstance(module, GraniteMoeHybridRMSNormGated):
            module.weight.data.fill_(1.0)


class GraniteMoeHybridModel(GraniteMoeSharedModel):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeHybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        ## overwritten because `HybridMambaAttentionDynamicCache` is needed
        if use_cache and past_key_values is None:
            logger.warning_once(
                "GraniteMoeHybrid requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. "
                "Because one was not provided, no cache will be returned."
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        position_embeddings = None
        # create position embeddings to be shared across the decoder layers
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            layer_mask = mamba_mask if decoder_layer.layer_type == "mamba" else causal_mask

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                output_router_logits=output_router_logits,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                if layer_outputs[1] is not None:
                    # append attentions only of attention layers. Mamba layers return `None` as the attention weights
                    all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                if layer_outputs[-1] is not None:
                    # append router logits only of expert layers. Regular MLP layers return `None` as the router logits
                    all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = next_decoder_cache if use_cache else None

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


class GraniteMoeHybridForCausalLM(GraniteMoeSharedForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)
        self.model = GraniteMoeHybridModel(config)
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- has a unique cache type, `HybridMambaAttentionDynamicCache`

        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if (
                inputs_embeds is not None  # Exception 1
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    def _supports_default_dynamic_cache(self) -> bool:
        """
        Function overwritten as this class uses its own `HybridMambaAttentionDynamicCache`
        and do not need to initialize the Cache in advance in order to save memory
        (because no back and forth `to_legacy_cache` and `from_legacy_cache` will be performed
        for `HybridMambaAttentionDynamicCache`).
        """
        return False


__all__ = ["GraniteMoeHybridForCausalLM", "GraniteMoeHybridModel", "GraniteMoeHybridPreTrainedModel"]
