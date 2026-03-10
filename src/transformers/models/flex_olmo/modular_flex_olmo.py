# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..mixtral.modeling_mixtral import MixtralModel, MixtralPreTrainedModel
from ..olmo2.modeling_olmo2 import Olmo2Attention, Olmo2RMSNorm, Olmo2RotaryEmbedding
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
    OlmoeMLP,
    OlmoeSparseMoeBlock,
    OlmoeTopKRouter,
)


@auto_docstring(checkpoint="allenai/FlexOlmo-7x7B-1T")
class FlexOlmoConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import FlexOlmoModel, FlexOlmoConfig

    >>> # Initializing a FlexOlmo style configuration
    >>> configuration = FlexOlmoConfig()

    >>> # Initializing a model from the FlexOlmo style configuration
    >>> model = FlexOlmoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flex_olmo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the added norm on q and k
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 100352,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-06,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 5,
        num_experts: int | None = 7,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.01,
        norm_topk_prob: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


# FlexOlmo RMS norm reuses Olmo2 RMS norm, which handles low precision slightly differently than the original Olmoe.
class FlexOlmoRMSNorm(Olmo2RMSNorm):
    pass


# FlexOlmo RMS norm reuses Olmo2 RMS norm, so that the output cos and sin are returned
# as float32 rather than the input type.
class FlexOlmoRotaryEmbedding(Olmo2RotaryEmbedding):
    pass


class FlexOlmoMLP(OlmoeMLP):
    pass


# FlexOlmo uses Olmo2 attention instead of OlmoE Attention since its `apply_rotary_pos_emb`
# implementation handles lower precision more faithfully to the Olmo codebase.
class FlexOlmoAttention(Olmo2Attention):
    pass


class FlexOlmoTopKRouter(OlmoeTopKRouter):
    pass


class FlexOlmoSparseMoeBlock(OlmoeSparseMoeBlock):
    pass


# FlexOlmo decoder layer is identical to OlmoE decoder layer except:
# - Norm is applied after attention/feedforward rather than before.
class FlexOlmoDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: FlexOlmoConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = FlexOlmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = FlexOlmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = FlexOlmoAttention(config=config, layer_idx=layer_idx)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# FlexOlmo uses Mixtral model as its base instead of OlmoE model since Mixtral is more up-to-date with the rest
# of the transformers library. For example, it uses the newer mechanisms of recording submodule outputs.
class FlexOlmoPreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(FlexOlmoTopKRouter, index=0),
        "hidden_states": FlexOlmoDecoderLayer,
        "attentions": FlexOlmoAttention,
    }


# FlexOlmo uses Mixtral model as its base instead of OlmoE model since Mixtral is more up-to-date with the rest
# of the transformers library. For example, it uses the newer mechanisms of recording submodule outputs.
# FlexOlmo model is identical to Mixtral model except:
# - FlexOlmo does not use sliding window attention.
class FlexOlmoModel(MixtralModel):
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

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

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
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlexOlmoForCausalLM(OlmoeForCausalLM):
    pass


__all__ = [
    "FlexOlmoConfig",
    "FlexOlmoForCausalLM",
    "FlexOlmoModel",
    "FlexOlmoPreTrainedModel",
]
