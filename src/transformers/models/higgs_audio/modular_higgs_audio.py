# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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

from typing import Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
)
from ..csm.modeling_csm import CsmBackboneModelEmbeddings
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
)
from .configuration_higgs_audio import HiggsAudioConfig
from .generation_higgs_audio import HiggsAudioGenerationMixin


if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.text_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(
            config.hidden_size, config.audio_num_codebooks * (config.audio_codebook_size + 2), bias=False
        )

    def forward(
        self,
        hidden_states,
        audio_out_mask,
    ):
        logits = self.text_lm_head(hidden_states)
        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        return logits, audio_logits


class HiggsAudioMLP(LlamaMLP):
    pass


class HiggsAudioRMSNorm(LlamaRMSNorm):
    pass


class HiggsAudioDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.audio_mlp = HiggsAudioMLP(config)
        self.audio_input_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        audio_out_mask = audio_out_mask.to(hidden_states.device)
        hidden_states = hidden_states.masked_scatter(
            audio_out_mask.unsqueeze(-1),
            self.audio_input_layernorm(hidden_states[audio_out_mask]).to(hidden_states.device),
        )
        hidden_states = hidden_states.masked_scatter(
            ~audio_out_mask.unsqueeze(-1),
            self.input_layernorm(hidden_states[~audio_out_mask]).to(hidden_states.device),
        )

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Dual-path FFN
        text_hidden_states = self.post_attention_layernorm(hidden_states[~audio_out_mask])
        audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[audio_out_mask])

        text_hidden_states = self.mlp(text_hidden_states)
        hidden_states[~audio_out_mask] += text_hidden_states.to(hidden_states.device)

        audio_hidden_states = self.audio_mlp(audio_hidden_states)
        hidden_states[audio_out_mask] += audio_hidden_states.to(hidden_states.device)

        return hidden_states


@auto_docstring(
    custom_intro="""
    The bare Higgs Audio Model outputting raw hidden-states without any specific head on top.
    """
)
@auto_docstring
class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, HiggsAudioRMSNorm):
            module.weight.data.fill_(1.0)


class HiggsAudioEmbeddings(CsmBackboneModelEmbeddings):
    def forward(self, input_ids):
        inputs_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = inputs_embeds.sum(dim=1)
        return inputs_embeds


class HiggsAudioModel(LlamaModel):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.embed_audio_tokens = HiggsAudioEmbeddings(config)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        audio_input_ids_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        audio_in_token_mask = input_ids == self.config.audio_in_token_idx
        audio_out_token_mask = input_ids == self.config.audio_out_token_idx
        audio_token_mask = audio_in_token_mask | audio_out_token_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

            if audio_input_ids is not None:
                audio_inputs_embeds = self.embed_audio_tokens(audio_input_ids[audio_input_ids_mask])
                inputs_embeds[audio_token_mask] = audio_inputs_embeds.to(inputs_embeds.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                audio_out_mask=audio_token_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The Higgs Audio model, a llama-like auto-regressive transformer model with dual-FFN.
    """
)
class HiggsAudioForConditionalGeneration(HiggsAudioPreTrainedModel, HiggsAudioGenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.config = config
        self.model = HiggsAudioModel(config)
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)
        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights

        self.post_init()

    def prepare_inputs_for_generation(
        self,
        *args,
        audio_input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        # Handle audio_input_ids slicing for generation with past_key_values
        if audio_input_ids is not None and model_inputs.get("past_key_values") is not None:
            current_input_length = (
                model_inputs["inputs_embeds"].shape[1]
                if model_inputs.get("inputs_embeds") is not None
                else model_inputs["input_ids"].shape[1]
            )
            audio_input_ids = audio_input_ids[:, -current_input_length:]
            audio_input_ids = audio_input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["audio_input_ids"] = audio_input_ids
        
        # Handle audio_input_ids_mask slicing for generation with past_key_values
        if audio_input_ids_mask is not None and model_inputs.get("past_key_values") is not None:
            current_input_length = (
                model_inputs["inputs_embeds"].shape[1]
                if model_inputs.get("inputs_embeds") is not None
                else model_inputs["input_ids"].shape[1]
            )
            audio_input_ids_mask = audio_input_ids_mask[:, -current_input_length:]
            audio_input_ids_mask = audio_input_ids_mask.clone(memory_format=torch.contiguous_format)
            model_inputs["audio_input_ids_mask"] = audio_input_ids_mask

        return model_inputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids_mask: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        # TODO: add lost computation back

        return CausalLMOutputWithPast(
            logits=self.audio_decoder_proj.audio_lm_head(last_hidden_state),
        )


__all__ = ["HiggsAudioForConditionalGeneration", "HiggsAudioPreTrainedModel", "HiggsAudioModel"]
