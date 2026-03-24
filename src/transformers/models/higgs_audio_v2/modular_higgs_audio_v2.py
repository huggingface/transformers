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


import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.output_capturing import capture_outputs
from ..csm.modeling_csm import CsmBackboneModelEmbeddings
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, LlamaModel, LlamaPreTrainedModel, LlamaRMSNorm
from .generation_higgs_audio_v2 import HiggsAudioV2GenerationMixin


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="bosonai/higgs-audio-v2-generation-3B-base")
@strict
class HiggsAudioV2Config(LlamaConfig):
    r"""
    audio_bos_token_id (`int`, *optional*, defaults to 128013):
        The token ID for the beginning-of-sequence token for audio output.
    audio_delay_token_id (`int`, *optional*, defaults to 128014):
        The token ID used for audio delay pattern in multi-codebook generation.
    audio_stream_bos_id (`int`, *optional*, defaults to 1024):
        The ID for the beginning-of-stream token in audio sequences.
    audio_stream_eos_id (`int`, *optional*, defaults to 1025):
        The ID for the end-of-stream token in audio sequences.

    Example:

    ```python
    >>> from transformers import HiggsAudioV2Model, HiggsAudioV2Config

    >>> # Initializing a HiggsAudioV2 style configuration
    >>> configuration = HiggsAudioV2Config()

    >>> # Initializing a model from the configuration
    >>> model = HiggsAudioV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    pad_token_id: int | None = 128001
    eos_token_id: int | list[int] | None = 128009
    head_dim: int | None = 128
    num_codebooks: int = 8
    codebook_size: int = 1024
    audio_token_id: int = 128016
    audio_bos_token_id: int = 128013
    audio_delay_token_id: int = 128014
    audio_stream_bos_id: int = 1024
    audio_stream_eos_id: int = 1025

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "factor": 32.0,
                "rope_theta": 500000.0,
                "high_freq_factor": 0.5,
                "low_freq_factor": 0.125,
                "original_max_position_embeddings": 1024,
                "rope_type": "llama3",
            }
        super().__post_init__(**kwargs)


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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None = None,
        audio_token_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
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


class HiggsAudioV2PreTrainedModel(LlamaPreTrainedModel, PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)

        if isinstance(module, HiggsAudioV2Embeddings):
            init.copy_(
                module.audio_tokens_offsets, torch.arange(self.config.num_codebooks) * self.config.codebook_size
            )


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

        return special_audio_mask

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
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
        >>> processor = AutoProcessor.from_pretrained("eustlb/higgs-audio-v2-generation-3B-base", device_map=device)
        >>> model = HiggsAudioV2Model.from_pretrained("eustlb/higgs-audio-v2-generation-3B-base", device_map=device)
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                audio_token_mask=audio_token_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
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
        r"""
        use_text_head (`bool`, *optional*, defaults to False):
            Whether to use a text language model head. Such head is not required for generation,
            but can be used to compute the text loss when training.
        """
        super().__init__(config)
        self.model = HiggsAudioV2Model(config)
        self.audio_lm_head = nn.Linear(config.hidden_size, config.num_codebooks * config.codebook_size, bias=False)
        self.text_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) if use_text_head else None

        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.LongTensor | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)

        if audio_input_ids is not None and model_inputs.get("past_key_values") is not None:
            current_cache_length = model_inputs.get("past_key_values").get_seq_length()
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
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Indices of audio codebook tokens.

            Indices can be obtained using [`HiggsAudioV2TokenizerModel.encode`].
        audio_input_ids_mask (`torch.BoolTensor` of shape `(batch_size, num_audio_frames)`, *optional*):
            Indicates which audio frames in `audio_input_ids` are valid.
        audio_labels (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Labels for the audio codebook tokens for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.codebook_size]. Token with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.codebook_size]`.
            Can be obtained using `output_labels=True` when calling [`HiggsAudioV2Processor`].

        Returns:
            [`~models.modeling_outputs.CausalLMOutputWithPast`]:
                A [`~models.modeling_outputs.CausalLMOutputWithPast`] containing the logits, loss (if labels are provided),
                and other outputs from the model.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration
        >>> model_id = "eustlb/higgs-audio-v2-generation-3B-base"
        >>> processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
        >>> model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
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
