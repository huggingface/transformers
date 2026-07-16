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

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...configuration_utils import PreTrainedConfig
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
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..moonshine.modeling_moonshine import MoonshineForConditionalGeneration
from ..qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding
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
    max_target_positions (`int`, *optional*, defaults to 1024):
        The maximum sequence length that the decoder might ever be used with.
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
    sub_configs = {"encoder_config": AutoConfig}
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
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "parakeet_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["parakeet_encoder"](
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


class CanaryPositionalEmbedding(SinusoidsPositionEmbedding):
    """
    Identical to [`SinusoidsPositionEmbedding`] except that the timescales and the `1 / sqrt(channels)` scaling match
    NeMo's `FixedPositionalEncoding` and the table is indexed by `position_ids`. The conversion script permutes the
    checkpoint from NeMo's interleaved sin/cos layout to this concatenated layout.
    """

    def __init__(self, length: int, channels: int):
        max_timescale = 10000 ** ((channels - 2) / channels)
        super().__init__(length, channels, max_timescale)

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1) / math.sqrt(self.channels)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.positional_embedding[position_ids]


@auto_docstring
class CanaryPreTrainedModel(WhisperPreTrainedModel):
    config: CanaryConfig
    _no_split_modules = ["ParakeetEncoderBlock", "CanaryDecoderLayer"]

    def _get_feat_extract_output_lengths(self):
        raise AttributeError("Not needed for Canary")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, CanaryPositionalEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.positional_embedding, position_embeddings)


class CanaryDecoder(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* [`CanaryDecoderLayer`] layers with fixed sinusoidal
    positional embeddings, an embedding layer norm and cross-attention to the FastConformer encoder outputs.
    """

    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.max_source_positions = None
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
        # unlike Whisper, NeMo normalizes the summed token and positional embeddings with a LayerNorm
        hidden_states = self.layernorm_embedding(inputs_embeds * self.embed_scale + positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        # unlike Whisper, the encoder outputs have variable length, so padded frames are masked in cross-attention
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
class CanaryForConditionalGeneration(MoonshineForConditionalGeneration):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=True)

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
        >>> transcription = processor.decode(generated_ids, skip_special_tokens=True)[0]
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
