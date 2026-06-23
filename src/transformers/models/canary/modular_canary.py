# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Canary model."""

import math

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoModel
from ..whisper.modeling_whisper import WhisperAttention, WhisperDecoderLayer
from .configuration_canary import CanaryConfig


logger = logging.get_logger(__name__)


class CanaryAttention(WhisperAttention):
    """
    Multi-headed attention for the Canary decoder. Identical to [`WhisperAttention`] except that the key projection
    uses a bias, matching NeMo's `MultiHeadAttention` where `key_net` (like the query, value and output projections)
    is a biased linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: int | None = None,
        config: CanaryConfig | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
            is_causal=is_causal,
            layer_idx=layer_idx,
            config=config,
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


class CanaryDecoderLayer(WhisperDecoderLayer):
    pass


class CanarySinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding from NeMo's `FixedPositionalEncoding`: the standard interleaved sin/cos table
    of [Attention Is All You Need](https://huggingface.co/papers/1706.03762) scaled by `1 / sqrt(embedding_dim)`. No
    other sinusoidal positional embedding in the library applies this scaling, so none can be reused here.
    """

    positional_embeddings: torch.Tensor

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.register_buffer("positional_embeddings", self._build_table(), persistent=False)

    def _build_table(self, device=None) -> torch.Tensor:
        positional_embeddings = torch.zeros(self.num_positions, self.embedding_dim, device=device)
        position = torch.arange(0, self.num_positions, dtype=torch.float, device=device).unsqueeze(1)
        coefficient = -math.log(10000.0) / self.embedding_dim
        div_term = torch.exp(coefficient * torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=device))
        positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        positional_embeddings[:, 1::2] = torch.cos(position * div_term)
        positional_embeddings = positional_embeddings / math.sqrt(self.embedding_dim)
        return positional_embeddings

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.positional_embeddings[position_ids]


@auto_docstring
class CanaryPreTrainedModel(PreTrainedModel):
    config: CanaryConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["ParakeetEncoderBlock", "CanaryDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CanarySinusoidalPositionalEmbedding):
            init.copy_(module.positional_embeddings, module._build_table())


class CanaryDecoder(CanaryPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* [`CanaryDecoderLayer`] layers with fixed sinusoidal
    positional embeddings, an embedding layer norm and cross-attention to the FastConformer encoder outputs.
    """

    main_input_name = "input_ids"
    input_modalities = ("text",)
    _can_record_outputs = {
        "hidden_states": CanaryDecoderLayer,
        "attentions": OutputRecorder(CanaryAttention, index=1, layer_name="self_attn"),
        "cross_attentions": OutputRecorder(CanaryAttention, index=1, layer_name="encoder_attn"),
    }

    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = CanarySinusoidalPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList(
            [CanaryDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding indices of `encoder_hidden_states`. Mask values
            selected in `[0, 1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_key_values_length
            position_ids = position_ids.unsqueeze(0)

        positions = self.embed_positions(position_ids).to(inputs_embeds.dtype)
        hidden_states = self.layernorm_embedding(inputs_embeds * self.embed_scale + positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

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

        for decoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring(
    custom_intro="""
    The bare Canary model (FastConformer encoder + Transformer decoder) outputting raw hidden-states without any
    specific head on top.
    """
)
class CanaryModel(CanaryPreTrainedModel):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.decoder = CanaryDecoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

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
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. For Canary these are the multitask prompt
            tokens (`<|startofcontext|> ... <|startoftranscript|> <source_lang> <target_lang> ...`) followed by the
            transcription/translation, built by [`CanaryProcessor`].
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence token in the sinusoidal positional embeddings.
        """
        # The encoder is a separate sub-model with its own config, so forward these output flags to it explicitly.
        for output_flag in ("output_attentions", "output_hidden_states"):
            if kwargs.get(output_flag) is None:
                kwargs[output_flag] = getattr(self.config, output_flag, False)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                **kwargs,
            )

        encoder_attention_mask = getattr(encoder_outputs, "attention_mask", None)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
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


@auto_docstring(
    custom_intro="""
    The Canary model with a language modeling head. Can be used for multilingual automatic speech recognition and
    speech-to-text translation.
    """
)
class CanaryForConditionalGeneration(CanaryPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = {"proj_out.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.model = CanaryModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

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
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqLMOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary (the multitask prompt prefix followed by the
            target text), built by [`CanaryProcessor`].
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence token in the sinusoidal positional embeddings.
        labels (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[0, ..., config.vocab_size]` or
            `-100`. Tokens set to `-100` are ignored (masked).

        Example:

        ```python
        >>> from transformers import AutoProcessor, CanaryForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> processor = AutoProcessor.from_pretrained("harshaljanjani/canary-1b-v2-hf")
        >>> model = CanaryForConditionalGeneration.from_pretrained("harshaljanjani/canary-1b-v2-hf")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], source_lang="en", target_lang="en", return_tensors="pt")
        >>> generated_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size)

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


__all__ = ["CanaryForConditionalGeneration", "CanaryModel", "CanaryPreTrainedModel"]
