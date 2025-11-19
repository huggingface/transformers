# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
)
from ...utils.generic import check_model_inputs
from ..csm.modeling_csm import CsmBackboneModelEmbeddings
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, LlamaModel, LlamaPreTrainedModel, LlamaRMSNorm
from .generation_higgs_audio_v2 import HiggsAudioV2GenerationMixin


if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)


class HiggsAudioV2Config(LlamaConfig):
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=28,
        num_attention_heads=24,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=128001,
        bos_token_id=1,
        eos_token_id=128009,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=128,
        num_codebooks=8,
        codebook_size=1024,
        audio_token_id=128016,
        audio_bos_token_id=128013,
        audio_delay_token_id=128014,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_token_id = audio_token_id
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_delay_token_id = audio_delay_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id


class HiggsAudioV2MLP(LlamaMLP):
    pass


class HiggsAudioV2RMSNorm(LlamaRMSNorm):
    pass


class HiggsAudioV2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HiggsAudioV2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.audio_mlp = HiggsAudioV2MLP(config)
        self.audio_input_layernorm = HiggsAudioV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        if audio_token_mask is None:
            hidden_states = self.audio_input_layernorm(hidden_states)
        else:
            audio_token_mask = audio_token_mask.to(hidden_states.device)
            hidden_states = hidden_states.masked_scatter(
                audio_token_mask.unsqueeze(-1),
                self.audio_input_layernorm(hidden_states[audio_token_mask]).to(hidden_states.device),
            )
            hidden_states = hidden_states.masked_scatter(
                ~audio_token_mask.unsqueeze(-1),
                self.input_layernorm(hidden_states[~audio_token_mask]).to(hidden_states.device),
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

        if audio_token_mask is None:
            audio_hidden_states = self.audio_post_attention_layernorm(hidden_states)
            audio_hidden_states = self.audio_mlp(audio_hidden_states)
            hidden_states = hidden_states + audio_hidden_states.to(hidden_states.device)
        else:
            text_hidden_states = self.post_attention_layernorm(hidden_states[~audio_token_mask])
            audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[audio_token_mask])

            text_hidden_states = self.mlp(text_hidden_states)
            hidden_states[~audio_token_mask] += text_hidden_states.to(hidden_states.device)

            audio_hidden_states = self.audio_mlp(audio_hidden_states)
            hidden_states[audio_token_mask] += audio_hidden_states.to(hidden_states.device)

        return hidden_states


class HiggsAudioV2Embeddings(CsmBackboneModelEmbeddings):
    def forward(self, input_ids):
        inputs_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = inputs_embeds.sum(dim=-2)
        return inputs_embeds


class HiggsAudioV2PreTrainedModel(LlamaPreTrainedModel):
    pass


class HiggsAudioV2Model(LlamaModel):
    def __init__(self, config: HiggsAudioV2Config):
        super().__init__(config)
        self.embed_audio_tokens = HiggsAudioV2Embeddings(config)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, audio_input_ids_mask: torch.LongTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of audio_input_ids. If the lengths are different, an error is raised.

        If input_ids and inputs_embeds are None, we return None.
        Indeed this means we cannot determine the placeholder mask, the model is to be used in a audio-only mode, hence we return None.
        """
        if input_ids is None and inputs_embeds is None:
            return None

        elif input_ids is None:
            special_audio_mask = inputs_embeds == self.embed_tokens(
                torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = special_audio_mask.all(-1)

        else:
            special_audio_mask = (input_ids == self.config.audio_token_id) | (
                input_ids == self.config.audio_delay_token_id
            )

        if audio_input_ids_mask is not None:
            n_audio_tokens_in_text = special_audio_mask.sum()
            n_audio_tokens_in_audio = audio_input_ids_mask.sum()
            if n_audio_tokens_in_text != n_audio_tokens_in_audio:
                raise ValueError(
                    f"Number of audio tokens in text and audio do not match: in text: {n_audio_tokens_in_text}, in audio: {n_audio_tokens_in_audio}"
                )

        return special_audio_mask

    @check_model_inputs()
    @auto_docstring
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
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Indices of audio codebook tokens.

            Indices can be obtained using [`HiggsAudioV2TokenizerModel.encode`].
        audio_input_ids_mask (`torch.BoolTensor` of shape `(batch_size, num_audio_frames)`, *optional*):
            Indicates which audio frames in `audio_input_ids` are valid.

        Returns:
            [`~models.modeling_outputs.BaseModelOutputWithPast`]:
                Usual decoder outputs with the placeholder positions already substituted by their corresponding
                audio embeddings.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HiggsAudioV2Model
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> processor = AutoProcessor.from_pretrained("eustlb/higgs-v2", device_map=device)
        >>> model = HiggsAudioV2Model.from_pretrained("eustlb/higgs-v2", device_map=device)
        >>> conversation = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "Generate audio following instruction."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "scene",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "Audio is recorded from a quiet room."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "assistant",
        ...         "content": [
        ...             {
        ...                 "type": "audio",
        ...                 "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav"
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
        ...             }
        ...         ]
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(conversation, return_dict=True, tokenize=True, sampling_rate=24000, return_tensors="pt")
        >>> inputs = inputs.to(model.device)
        >>> outputs = model(**inputs)
        ```
        """
        if (input_ids is None) and (inputs_embeds is None) and (audio_input_ids is None):
            raise ValueError("You must specify at least one of input_ids, inputs_embeds, or audio_input_ids")

        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("Only one of input_ids or inputs_embeds can be provided")

        audio_token_mask = self.get_placeholder_mask(input_ids, inputs_embeds, audio_input_ids_mask)

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if audio_input_ids is not None:
            audio_embeds = self.embed_audio_tokens(audio_input_ids)

        if inputs_embeds is not None and audio_input_ids is not None:
            audio_embeds = (
                audio_embeds[audio_input_ids_mask.to(audio_embeds.device)]
                if audio_input_ids_mask is not None
                else audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask[..., None].expand_as(inputs_embeds), audio_embeds.to(inputs_embeds.device)
            )
        elif audio_input_ids is not None:
            inputs_embeds = audio_embeds

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
                audio_token_mask=audio_token_mask,
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
class HiggsAudioV2ForConditionalGeneration(HiggsAudioV2PreTrainedModel, HiggsAudioV2GenerationMixin):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = ["text_lm_head.weight"]

    def __init__(self, config: HiggsAudioV2Config, use_text_head: bool = False):
        super().__init__(config)
        self.model = HiggsAudioV2Model(config)
        self.audio_lm_head = nn.Linear(config.hidden_size, config.num_codebooks * config.codebook_size, bias=False)
        self.text_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) if use_text_head else None

        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        audio_input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)

        if audio_input_ids is not None and model_inputs.get("past_key_values") is not None:
            current_cache_length = model_inputs["cache_position"][0]
            audio_token_mask = (input_ids == self.config.audio_token_id) | (
                input_ids == self.config.audio_delay_token_id
            )
            in_cache_num_audio_input_ids = audio_token_mask[:, :current_cache_length].sum(dim=-1)

            # already cached audio_input_ids should be masked
            # this surmise that audio_input_ids are right padded!
            valid_audio_input_ids = audio_input_ids_mask.cumsum(dim=-1) > in_cache_num_audio_input_ids[:, None]
            audio_input_ids_mask = audio_input_ids_mask & valid_audio_input_ids

        if audio_input_ids_mask is not None and (~audio_input_ids_mask[:, :-1]).all():
            # in decoding mode, we only pass audio_input_ids
            audio_input_ids = audio_input_ids[:, -1:, :].clone(memory_format=torch.contiguous_format)
            model_inputs.pop("input_ids", None)
            audio_input_ids_mask = None

        model_inputs["audio_input_ids"] = audio_input_ids
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Indices of audio codebook tokens.

            Indices can be obtained using [`HiggsAudioV2TokenizerModel.encode`].
        audio_input_ids_mask (`torch.BoolTensor` of shape `(batch_size, num_audio_frames)`, *optional*):
            Indicates which audio frames in `audio_input_ids` are valid.

        Returns:
            [`~models.modeling_outputs.BaseModelOutputWithPast`]:
                Usual decoder outputs with the placeholder positions already substituted by their corresponding
                audio embeddings.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HiggsAudioV2Model
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> processor = AutoProcessor.from_pretrained("eustlb/higgs-v2", device_map=device)
        >>> model = HiggsAudioV2Model.from_pretrained("eustlb/higgs-v2", device_map=device)
        >>> conversation = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "Generate audio following instruction."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "scene",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "Audio is recorded from a quiet room."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "assistant",
        ...         "content": [
        ...             {
        ...                 "type": "audio",
        ...                 "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav"
        ...             }
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "text",
        ...                 "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
        ...             }
        ...         ]
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(conversation, return_dict=True, tokenize=True, sampling_rate=24000, return_tensors="pt")
        >>> inputs = inputs.to(model.device)
        >>> outputs = model(**inputs)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.audio_lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if audio_labels is not None:
            audio_logits = logits.reshape(*logits.shape[:2], self.config.num_codebooks, self.config.codebook_size)
            audio_labels_expanded = input_ids.new_ones((*input_ids.shape[:2], 8)) * -100
            audio_token_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, audio_input_ids_mask)
            audio_labels_expanded[audio_token_mask] = audio_labels[audio_input_ids_mask]

            codebook_losses = []
            for codebook_idx in range(self.config.num_codebooks):
                codebook_logits = audio_logits[:, :, codebook_idx, :]
                codebook_labels = audio_labels_expanded[:, :, codebook_idx]
                codebook_losses.append(
                    self.loss_function(codebook_logits, codebook_labels, self.config.codebook_size, **kwargs)
                )

            loss = sum(codebook_losses)

        if labels is not None:
            if self.text_lm_head is not None:
                text_logits = self.text_lm_head(hidden_states[:, slice_indices, :])
                text_loss = self.loss_function(text_logits, labels, self.config.vocab_size, **kwargs)
                loss = text_loss if loss is None else loss + text_loss
            else:
                logger.warning_once(
                    f"`labels` provided to {self.__class__.__name__} but `text_lm_head` is disabled. "
                    f"Text labels ignored. Set `use_text_head=True` in model init to enable text loss."
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HiggsAudioV2ForConditionalGeneration",
    "HiggsAudioV2PreTrainedModel",
    "HiggsAudioV2Model",
    "HiggsAudioV2Config",
]
