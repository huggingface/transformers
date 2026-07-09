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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..parakeet import ParakeetEncoderConfig
from ..whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperModel,
    WhisperPreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="harshaljanjani/canary-1b-v2-hf")
@strict
class CanaryConfig(PreTrainedConfig):
    r"""
    encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
        The config object or dictionary of the FastConformer encoder ([`ParakeetEncoderConfig`]).
    vocab_size (`int`, *optional*, defaults to 16384):
        Vocabulary size of the Canary decoder.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the decoder layers and the pooler layer.
    decoder_layers (`int`, *optional*, defaults to 8):
        Number of decoder layers.
    decoder_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the decoder.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556)
        for more details.
    activation_function (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function in the decoder feed-forward layers.
    max_target_positions (`int`, *optional*, defaults to 1024):
        The maximum sequence length that the decoder might ever be used with.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the decoder embeddings, attention output, and feed-forward layers.
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for activations inside the decoder feed-forward layer.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Whether to scale the decoder token embeddings by `sqrt(d_model)`.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether the model should return the last key/values attentions.
    is_encoder_decoder (`bool`, *optional*, defaults to `True`):
        Whether the model is used as an encoder/decoder model.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the decoder input embeddings and the language modeling head.
    pad_token_id (`int`, *optional*, defaults to 2):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 4):
        Beginning of stream token id (`<|startoftranscript|>`).
    eos_token_id (`int`, *optional*, defaults to 3):
        End of stream token id (`<|endoftext|>`).
    decoder_start_token_id (`int`, *optional*, defaults to 7):
        The token id that starts decoding (`<|startofcontext|>`, the first token of the multitask prompt).

    Example:

    ```python
    >>> from transformers import CanaryForConditionalGeneration, CanaryConfig

    >>> # Initializing a Canary configuration
    >>> configuration = CanaryConfig()

    >>> # Initializing a model from the configuration
    >>> model = CanaryForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "canary"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": ParakeetEncoderConfig}
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "decoder_attention_heads",
        "num_hidden_layers": "decoder_layers",
    }

    encoder_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 16384
    d_model: int = 1024
    decoder_layers: int = 8
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 4096
    decoder_layerdrop: float | int = 0.0
    activation_function: str = "relu"
    max_target_positions: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    scale_embedding: bool = False
    use_cache: bool = True
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 4
    eos_token_id: int | None = 3
    decoder_start_token_id: int | None = 7
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig(
                num_hidden_layers=32,
                num_mel_bins=128,
                scale_input=False,
                layerdrop=0.0,
                dropout_positions=0.0,
            )
        super().__post_init__(**kwargs)


class CanaryAttention(WhisperAttention):
    """
    Multi-headed attention for the Canary decoder. Identical to [`WhisperAttention`] except that the key projection
    uses a bias, matching NeMo's `MultiHeadAttention` where `key_net` (like the query, value and output projections)
    is a biased linear layer.
    """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


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
class CanaryPreTrainedModel(WhisperPreTrainedModel):
    config: CanaryConfig
    _no_split_modules = ["ParakeetEncoderBlock", "CanaryDecoderLayer"]

    def _get_feat_extract_output_lengths(self):
        raise AttributeError("Not needed for Canary")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, CanarySinusoidalPositionalEmbedding):
            init.copy_(module.positional_embeddings, module._build_table())


class CanaryDecoder(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* [`CanaryDecoderLayer`] layers with fixed sinusoidal
    positional embeddings, an embedding layer norm and cross-attention to the FastConformer encoder outputs.
    """

    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.max_source_positions = None
        self.embed_positions = CanarySinusoidalPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
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
            Mask to avoid performing attention on padding indices in `encoder_hidden_states`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
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
class CanaryModel(WhisperModel):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.post_init()

    def _mask_input_features(self):
        raise AttributeError("Not needed for Canary")

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @can_return_tuple
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        for output_flag in ("output_attentions", "output_hidden_states"):
            if kwargs.get(output_flag) is None:
                kwargs[output_flag] = getattr(self.config, output_flag, False)
        return self.encoder(input_features=input_features, attention_mask=attention_mask, **kwargs)

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
        if encoder_outputs is None:
            encoder_outputs = self.get_audio_features(input_features, attention_mask, **kwargs)

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

        >>> inputs = processor.apply_transcription_request(audio=ds[0]["audio"]["array"], source_language="en")
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


__all__ = ["CanaryConfig", "CanaryForConditionalGeneration", "CanaryModel", "CanaryPreTrainedModel"]
