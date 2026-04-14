# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ...utils.output_capturing import OutputRecorder
from ..auto.modeling_auto import AutoModel
from ..clip.modeling_clip import CLIPMLP
from ..moonshine.modeling_moonshine import (
    MoonshineDecoder,
    MoonshineForConditionalGeneration,
    MoonshineModel,
    MoonshinePreTrainedModel,
    eager_attention_forward,
    shift_tokens_right,
)
from .configuration_cohere_asr import CohereAsrConfig


class CohereAsrDecoderMLP(CLIPMLP):
    pass


# Modular automatically inherits RoPE, hence no inheritance for now
class CohereAsrSelfAttention(nn.Module):
    def __init__(self, config: CohereAsrConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            past_key_values = past_key_values.self_attention_cache
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


# Modular automatically inherits RoPE, hence no inheritance for now
class CohereAsrCrossAttention(nn.Module):
    def __init__(self, config: CohereAsrConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = encoder_hidden_states.shape[1]

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_states = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_states, value_states = past_key_values.cross_attention_cache.update(
                    key_states, value_states, self.layer_idx
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

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
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CohereAsrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self_attn = CohereAsrSelfAttention(config=config, layer_idx=layer_idx)
        self.encoder_attn = CohereAsrCrossAttention(config=config, layer_idx=layer_idx)

        self.mlp = CohereAsrDecoderMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        encoder_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CohereAsrPreTrainedModel(MoonshinePreTrainedModel):
    main_input_name = "input_features"
    _keys_to_ignore_on_load_unexpected = [r"preprocessor\.featurizer\..*"]


class CohereAsrDecoder(MoonshineDecoder):
    _can_record_outputs = {
        "attentions": OutputRecorder(CohereAsrSelfAttention, index=1, layer_name="self_attn"),
        "hidden_states": CohereAsrDecoderLayer,
        "cross_attentions": OutputRecorder(CohereAsrCrossAttention, index=1, layer_name="encoder_attn"),
    }

    def __init__(self, config):
        super().__init__(config)
        del self.rotary_emb
        self.norm = nn.LayerNorm(config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layernorm = nn.LayerNorm(config.hidden_size)
        self.proj = nn.Linear(config.encoder_config.hidden_size, config.hidden_size, bias=True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding indices in `encoder_hidden_states`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        """
        encoder_hidden_states = self.proj(encoder_hidden_states)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Fixed sinusoidal position embedding added to token embeddings, then layernorm
        pos_emb = self.pos_emb(position_ids.squeeze(0))
        pos_emb = pos_emb.to(inputs_embeds.device)
        inputs_embeds = self.embedding_layernorm(inputs_embeds + pos_emb)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class CohereAsrModel(MoonshineModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
        decoder_position_ids: tuple[torch.LongTensor] | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Float values of the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, CohereAsrModel
        >>> from datasets import load_dataset

        >>> model = CohereAsrModel.from_pretrained("UsefulSensors/cohere_asr-tiny")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("UsefulSensors/cohere_asr-tiny")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 288]
        ```
        """
        # Main difference: uses `input_features` instead of `input_values`
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(input_features, attention_mask=attention_mask, **kwargs)

        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class CohereAsrForConditionalGeneration(MoonshineForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
        decoder_position_ids: tuple[torch.LongTensor] | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqLMOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Float values of the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("UsefulSensors/cohere_asr-tiny")
        >>> model = CohereAsrForConditionalGeneration.from_pretrained("UsefulSensors/cohere_asr-tiny")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(input_features, max_new_tokens=100)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        # Main difference: uses `input_features` instead of `input_values`
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs: Seq2SeqModelOutput = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(self, *args, audio_chunk_index=None, **kwargs):
        # audio_chunk_index is returned by the processor but not used by the model, absorb it here
        return GenerationMixin.prepare_inputs_for_generation(self, *args, **kwargs)


__all__ = [
    "CohereAsrPreTrainedModel",
    "CohereAsrModel",
    "CohereAsrForConditionalGeneration",
]
