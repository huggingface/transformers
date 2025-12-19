# coding=utf-8
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

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...activations import ACT2FN
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ..auto import AutoConfig
from ..voxtral.configuration_voxtral import VoxtralConfig, VoxtralEncoderConfig
from ..voxtral.modeling_voxtral import (
    VoxtralAttention,
    VoxtralEncoder,
    VoxtralEncoderLayer,
    VoxtralForConditionalGeneration,
    VoxtralPreTrainedModel,
    eager_attention_forward,
)
from ..glm4.modeling_glm4 import apply_rotary_pos_emb as apply_rotary_pos_emb_audio

logger = logging.get_logger(__name__)


class GlmasrAudioRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class GlmasrEncoderConfig(VoxtralEncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmasrEncoder`]. It is used to instantiate a
    glmasr audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the glmasr
    architecture.

    e.g. [zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51866):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(hidden_size) if True.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu",
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `GlmasrProcessor` class.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import GlmasrEncoderConfig, GlmasrEncoder

    >>> # Initializing a GlmasrEncoderConfig
    >>> configuration = GlmasrEncoderConfig()

    >>> # Initializing a GlmasrEncoder (with random weights)
    >>> model = GlmasrEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glmasr_encoder"

    attribute_map = {
        "hidden_size": "d_model",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    def __init__(
        self,
        vocab_size=51866,
        hidden_size=1280,
        intermediate_size=6144,
        num_hidden_layers=32,
        num_attention_heads=16,
        scale_embedding=False,
        activation_function="gelu",
        num_mel_bins=128,
        max_source_positions=1500,
        initializer_range=0.02,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(hidden_size) if True
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.initializer_range = initializer_range

        # TODO: @eustlb, we do not use dropout and layerdrop, yet we need to hardcode them
        # to be able to use Whisper with modular (here actually from Qwen2-Audio and copied from).
        # After a future Whisper refactor, we should remove this.
        self.dropout = 0.0
        self.layerdrop = 0.0
        self.activation_dropout = 0.0

        self.attention_dropout = attention_dropout


class GlmasrConfig(VoxtralConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmasrForConditionalGeneration`]. It is used to instantiate an
    glmasr model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the glmasr-Mini-3B.

    e.g. [zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        audio_token_id (`int`, *optional*):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function (function or string) in the multi-modal projector.

    ```python
    >>> from transformers import GlmasrForConditionalGeneration, GlmasrConfig

    >>> # Initializing a glmasr configuration
    >>> configuration = GlmasrConfig(audio_token_id=24, projector_hidden_act="gelu")

    >>> # Initializing a 3B model with random weights
    >>> model = GlmasrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glmasr"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 59264,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 10000.0,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=None,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__()


class GlmasrAttention(VoxtralAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        cos = position_embeddings[..., 0]
        sin = position_embeddings[..., 1]

        query_states, key_states = apply_rotary_pos_emb_audio(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class GlmasrEncoderLayer(VoxtralEncoderLayer):
    def __init__(self, config: GlmasrConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        position_embeddings: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


class GlmasrPreTrainedModel(VoxtralPreTrainedModel):
    pass


class GlmasrEncoder(VoxtralEncoder):
    def __init__(self, config: GlmasrEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList([GlmasrEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        self.rotary_pos_emb = GlmasrAudioRotaryEmbedding(config.hidden_size // config.encoder_attention_heads)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_embeddings: Optional[torch.Tensor] = None,
        position_ids=None,
        **kwargs,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        seqlen = inputs_embeds.shape[1]
        freqs = self.rotary_pos_emb(seqlen)
        rotary_embs = torch.stack([freqs.cos(), freqs.sin()], dim=-1).to(dtype=inputs_embeds.dtype)
        position_embeddings = rotary_embs[position_ids]

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class GlmasrMultiModalProjector(nn.Module):
    def __init__(self, config: GlmasrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.hidden_size * 4, config.text_config.hidden_size * 2)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GlmasrForConditionalGeneration(VoxtralForConditionalGeneration):
    pass


class GlmasrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {
            "max_source_positions": 1500,
        },
    }


class GlmasrProcessor(ProcessorMixin):
    r"""
    Constructs a Glmasr processor which wraps a Glmasr feature extractor and a Glmasr tokenizer into a single processor.

    [`GlmasrProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`PretrainTokenizerFast`]. See the
    [`~PretrainAudioProcessor.__call__`] and [`~PretrainAudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`PretrainTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
        audio_token (`str`, *optional*, defaults to `"<|pad|>"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|begin_of_audio|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|end_of_audio|>"`):
            The token to use for audio eos tokens.
    """

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|pad|>",
        audio_bos_token="<|begin_of_audio|>",
        audio_eos_token="<|end_of_audio|>",
    ):
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def _get_audio_token_length(self, audio_length: int, merge_factor: int = 4) -> int:
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_length = (audio_length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        num_tokens = (audio_length - merge_factor) // merge_factor + 1
        return min(num_tokens, 1500 // merge_factor)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio: Union[np.ndarray, list[np.ndarray]] = None,
        **kwargs: Unpack[GlmasrProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            GlmasrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if audio is not None:
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            expanded_text = []
            audio_lengths = audio_inputs.pop("attention_mask").sum(-1).tolist()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_audio_tokens = self._get_audio_token_length(audio_length)
                    expanded_audio_token = self.audio_token * num_audio_tokens
                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx - len(self.audio_bos_token) : audio_token_start_idx]
                        == self.audio_bos_token
                    )
                    has_eos = (
                        sample[audio_token_end_idx : audio_token_end_idx + len(self.audio_eos_token)]
                        == self.audio_eos_token
                    )

                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)

                expanded_text.append(sample)
            text = expanded_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, inputs, modalities=["audio"])

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)


__all__ = [
    "GlmasrEncoderConfig",
    "GlmasrConfig",
    "GlmasrPreTrainedModel",
    "GlmasrEncoder",
    "GlmasrForConditionalGeneration",
    "GlmasrProcessor",
]
