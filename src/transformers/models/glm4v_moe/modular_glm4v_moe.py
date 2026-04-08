# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3NaiveMoe
from ..glm4.modeling_glm4 import Glm4Attention
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeMLP,
    Glm4MoeMoE,
    Glm4MoePreTrainedModel,
    Glm4MoeTopkRouter,
    eager_attention_forward,
)
from ..glm4v.configuration_glm4v import Glm4vConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vTextModel,
    Glm4vVisionModel,
    Glm4vVisionRotaryEmbedding,
)
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from ..qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeModelOutputWithPast,
    load_balancing_loss_func,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-4.5V")
@strict
class Glm4vMoeTextConfig(Glm4MoeConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                                \--k dense layers--/

    Example:

    ```python
    >>> from transformers import Glm4vMoeTextModel, Glm4vMoeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4vMoeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4vMoeTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Glm4vMoe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    vocab_size: int = 151424
    max_position_embeddings: int = 65536
    attention_bias: bool = True
    router_aux_loss_coef: float = 0.0001
    use_qk_norm = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(self, **kwargs)


@auto_docstring(checkpoint="zai-org/GLM-4.5V")
@strict
class Glm4vMoeConfig(Glm4vConfig):
    r"""
    image_start_token_id (`int`, *optional*, defaults to 151339):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 151340):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 151341):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 151342):
        The video end token index to encode the end of video.

    ```python
    >>> from transformers import Glm4vMoeForConditionalGeneration, Glm4vMoeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4vMoeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4vMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    image_token_id: int = 151363
    video_token_id: int = 151364


class Glm4vMoeTextAttention(Glm4Attention):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.rope_parameters = config.rope_parameters

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

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


class Glm4vMoeTextTopkRouter(Glm4MoeTopkRouter, nn.Module):
    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)


class Glm4vMoeTextNaiveMoe(DeepseekV3NaiveMoe):
    pass


class Glm4vMoeTextMoE(Glm4MoeMoE):
    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)
        self.config = config
        self.experts = Glm4vMoeTextNaiveMoe(config)
        self.gate = Glm4vMoeTextTopkRouter(config)
        self.shared_experts = Glm4vMoeTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm4vMoeTextMLP(Glm4MoeMLP):
    pass


class Glm4vMoeTextDecoderLayer(Glm4MoeDecoderLayer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class Glm4vMoePreTrainedModel(Glm4MoePreTrainedModel):
    config: Glm4vMoeConfig
    base_model_prefix = "model"
    input_modalities = ("text", "image", "video")
    _no_split_modules = ["Glm4vMoeTextDecoderLayer", "Glm4vMoeVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _can_record_outputs = {}

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Glm4vMoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Glm4vMoeCausalLMOutputWithPast(Qwen3VLMoeCausalLMOutputWithPast):
    pass


class Glm4vMoeVisionRotaryEmbedding(Glm4vVisionRotaryEmbedding):
    pass


@auto_docstring
class Glm4vMoeVisionModel(Glm4vVisionModel):
    pass


@auto_docstring
class Glm4vMoeTextModel(Glm4vTextModel):
    _can_record_outputs = {
        "hidden_states": Glm4vMoeTextDecoderLayer,
        "attentions": Glm4vMoeTextAttention,
        "router_logits": Glm4vMoeTextTopkRouter,
    }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Glm4vMoeModelOutputWithPast(Qwen3VLMoeModelOutputWithPast):
    pass


class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.text_config.num_local_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

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
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Glm4vMoeCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Glm4vMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Glm4vMoeConfig",
    "Glm4vMoeVisionConfig",  # noqa: F822
    "Glm4vMoeTextConfig",
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",  # noqa: F822
    "Glm4vMoePreTrainedModel",
    "Glm4vMoeTextModel",
    "Glm4vMoeVisionModel",
]
