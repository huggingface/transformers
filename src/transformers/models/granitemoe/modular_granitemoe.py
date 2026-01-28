# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, check_model_inputs
from ..granite.modeling_granite import GraniteRMSNorm, GraniteRotaryEmbedding
from ..jetmoe.modeling_jetmoe import JetMoeParallelExperts, JetMoeTopKGating
from ..llama.modeling_llama import LlamaAttention, LlamaPreTrainedModel
from ..mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralForCausalLM, MixtralModel, load_balancing_loss_func
from .configuration_granitemoe import GraniteMoeConfig


class GraniteMoeRMSNorm(GraniteRMSNorm):
    pass


class GraniteMoeRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class GraniteMoeParallelExperts(JetMoeParallelExperts):
    pass


class GraniteMoeTopKGating(JetMoeTopKGating):
    pass


@use_kernel_forward_from_hub("ScatterMoEGatedMLP")
class GraniteMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: GraniteMoeConfig):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.intermediate_size
        self.activation = ACT2FN[config.hidden_act]
        self.input_linear = GraniteMoeParallelExperts(config.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = GraniteMoeParallelExperts(config.num_local_experts, self.hidden_size, self.input_size)

        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size, _ = self.router(layer_input)

        expert_inputs = layer_input[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        return layer_output


class GraniteMoeAttention(LlamaAttention):
    def __init__(self, config: GraniteMoeConfig, layer_idx: int):
        super().__init__(self, config, layer_idx)
        self.scaling = config.attention_multiplier  # Only diff with llama


class GraniteMoeDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: GraniteMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteMoeAttention(config=config, layer_idx=layer_idx)
        self.block_sparse_moe = GraniteMoeMoE(config)
        self.input_layernorm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        del self.mlp
        self.block_sparse_moe = GraniteMoeMoE(config)
        self.residual_multiplier = config.residual_multiplier  # Only diff with mixtral!

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier  # diff
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier  # diff
        return hidden_states


@auto_docstring
class GraniteMoePreTrainedModel(LlamaPreTrainedModel, PreTrainedModel):
    config: GraniteMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraniteMoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False  # TopK gating fails fullgraph compilation at "expert_size = expert_size.tolist()"

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteMoeParallelExperts):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class GraniteMoeModel(MixtralModel):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(  # ONLY DIFF WITH MIXTRAL: NO SLIDING
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        inputs_embeds = inputs_embeds * self.embedding_multiplier
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteMoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__(config)
        self.model = GraniteMoeModel(config)
        self.logits_scaling = config.logits_scaling

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GraniteMoeForCausalLM

        >>> model = GraniteMoeForCausalLM.from_pretrained("ibm/PowerMoE-3b")
        >>> tokenizer = AutoTokenizer.from_pretrained("ibm/PowerMoE-3b")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # Only compute necessary logits
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits / self.config.logits_scaling

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["GraniteMoeForCausalLM", "GraniteMoeModel", "GraniteMoePreTrainedModel"]
