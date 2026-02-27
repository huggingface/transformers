# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

import re
from math import pi

import numpy as np
import torch
from torch import Tensor, broadcast_tensors, nn
from torch.amp import autocast

from ... import initialization as init
from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config, AudioFlamingo3EncoderConfig
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3Encoder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..llama.modeling_llama import LlamaRotaryEmbedding


logger = logging.get_logger(__name__)


class MusicFlamingoEncoderConfig(AudioFlamingo3EncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicFlamingoEncoder`].

    e.g. [nvidia/music-flamingo-hf](https://huggingface.co/nvidia/music-flamingo-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `MusicFlamingoProcessor` class.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556)
            for more details.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(hidden_size).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        head_dim (`int`, *optional*, defaults to 256):
            Rotary embedding dimension used per axis in [`MusicFlamingoRotaryEmbedding`]. Since the rotary embedding is
            applied on two axes (batch and time), the rotated hidden size is `2 * head_dim`, which must be less than
            or equal to `hidden_size`.
        max_position_embeddings (`int`, *optional*, defaults to 1200):
            Maximum cached positions used by the MusicFlamingo time rotary embedding. This should match the processor
            `max_audio_len`.
        rope_parameters (`dict`, *optional*):
            RoPE parameters for [`MusicFlamingoRotaryEmbedding`]. Supports the standard keys `"rope_type"` (defaults to
            `"default"`) and `"rope_theta"`. By default, `"rope_theta"` is derived from `max_position_embeddings / (2 * pi)`.

    Example:

    ```python
    >>> from transformers import MusicFlamingoEncoderConfig, MusicFlamingoEncoder

    >>> # Initializing an MusicFlamingoEncoderConfig
    >>> configuration = MusicFlamingoEncoderConfig()

    >>> # Initializing an MusicFlamingoEncoder (with random weights)
    >>> model = MusicFlamingoEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "musicflamingo_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=5120,
        layerdrop=0.0,
        activation_function="gelu",
        hidden_size=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_range=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        head_dim=256,
        max_position_embeddings=1200,
        rope_parameters=None,
        **kwargs,
    ):
        # Backward compatibility with older serialized configs before `rope_parameters`.
        legacy_rotary_dim = kwargs.pop("rotary_dim", None)
        legacy_rotary_max_time = kwargs.pop("rotary_max_time", None)

        super().__init__(
            num_mel_bins=num_mel_bins,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            layerdrop=layerdrop,
            activation_function=activation_function,
            hidden_size=hidden_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            initializer_range=initializer_range,
            scale_embedding=scale_embedding,
            max_source_positions=max_source_positions,
            **kwargs,
        )

        # Legacy names used before aligning with RoPE conventions.
        if legacy_rotary_dim is not None:
            head_dim = legacy_rotary_dim
        if legacy_rotary_max_time is not None:
            max_position_embeddings = legacy_rotary_max_time

        rope_parameters = {} if rope_parameters is None else dict(rope_parameters)
        if "dim" in rope_parameters:
            head_dim = rope_parameters.pop("dim")
        if "max_time" in rope_parameters:
            max_position_embeddings = rope_parameters.pop("max_time")

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault(
            "rope_theta",
            self.max_position_embeddings / (2 * pi) if self.max_position_embeddings is not None else 50000,
        )
        self.rope_parameters = rope_parameters


class MusicFlamingoConfig(AudioFlamingo3Config):
    r"""
    This is the configuration class to store the configuration of a [`MusicFlamingoForConditionalGeneration`].

    e.g. [nvidia/music-flamingo-hf](https://huggingface.co/nvidia/music-flamingo-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[MusicFlamingoEncoderConfig, dict]`, *optional*, defaults to `MusicFlamingoEncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151669):
            The audio token index to encode the audio prompt.
        audio_bos_token_id (`int`, *optional*, defaults to 151670):
            The beginning-of-audio token index used to mark the start of audio spans.
        audio_eos_token_id (`int`, *optional*, defaults to 151671):
            The end-of-audio token index used to mark the end of audio spans.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to include bias terms in the projector.

    Example:

    ```python
    >>> from transformers import MusicFlamingoForConditionalGeneration, MusicFlamingoConfig, MusicFlamingoEncoderConfig, Qwen2Config

    >>> # Initializing an MusicFlamingoEncoder config
    >>> audio_config = MusicFlamingoEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing an MusicFlamingo configuration
    >>> configuration = MusicFlamingoConfig(audio_config, text_config)

    >>> # Initializing a model from the musicflamingo style configuration
    >>> model = MusicFlamingoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "musicflamingo"

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151669,
        audio_bos_token_id=151670,
        audio_eos_token_id=151671,
        projector_hidden_act="gelu",
        projector_bias=True,
        **kwargs,
    ):
        # Backward compatibility with older serialized top-level configs; these are now owned by `audio_config`.
        legacy_rotary_dim = kwargs.pop("rotary_dim", None)
        legacy_rotary_max_time = kwargs.pop("rotary_max_time", None)

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "musicflamingo_encoder")
        elif audio_config is None:
            audio_config = {
                "model_type": "musicflamingo_encoder",
            }

        if isinstance(audio_config, dict) and (legacy_rotary_dim is not None or legacy_rotary_max_time is not None):
            if legacy_rotary_dim is not None:
                audio_config.setdefault("head_dim", legacy_rotary_dim)
            if legacy_rotary_max_time is not None:
                audio_config.setdefault("max_position_embeddings", legacy_rotary_max_time)

            rope_parameters = dict(audio_config.get("rope_parameters") or {})
            max_position_embeddings = audio_config.get("max_position_embeddings", legacy_rotary_max_time)
            if max_position_embeddings is not None:
                rope_parameters.setdefault("rope_theta", max_position_embeddings / (2 * pi))
            rope_parameters.setdefault("rope_type", "default")
            audio_config["rope_parameters"] = rope_parameters

        super().__init__(
            audio_config=audio_config,
            text_config=text_config,
            audio_token_id=audio_token_id,
            projector_hidden_act=projector_hidden_act,
            projector_bias=projector_bias,
            **kwargs,
        )

        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id


class MusicFlamingoProcessorKwargs(AudioFlamingo3ProcessorKwargs): ...


class MusicFlamingoProcessor(AudioFlamingo3Processor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
        audio_bos_token="<|sound_bos|>",
        audio_eos_token="<|sound_eos|>",
        max_audio_len=1200,
    ):
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            max_audio_len=max_audio_len,
        )
        del self.default_transcription_prompt
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(audio_bos_token)
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(audio_eos_token)

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[MusicFlamingoProcessorKwargs],
    ) -> BatchFeature:
        call_kwargs = self._merge_kwargs(
            MusicFlamingoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_inputs = {}
        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            window_size = int(audio_kwargs["sampling_rate"] * audio_kwargs["chunk_length"])
            max_windows = int(self.max_audio_len // audio_kwargs["chunk_length"])

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []
            num_audio_time_steps = int(
                self._get_audio_token_length(
                    torch.tensor([self.feature_extractor.nb_max_frames], dtype=torch.long)
                ).item()
            )
            audio_time_step = self.feature_extractor.chunk_length / num_audio_time_steps
            audio_time_offsets = torch.arange(num_audio_time_steps, dtype=torch.float32) * audio_time_step

            chunk_start_times: list[float] = []

            for audio_el in audio:
                n_samples = int(audio_el.shape[0])
                n_win = max(1, (n_samples + window_size - 1) // window_size)
                if n_win > max_windows:
                    logger.warning(
                        f"Audio duration ({n_samples / audio_kwargs['sampling_rate']:.1f}s) exceeds {self.max_audio_len}s; truncating to first {self.max_audio_len}s."
                    )
                    n_win = max_windows
                per_sample_windows.append(n_win)

                time_cap = min(n_samples, n_win * window_size)
                for i in range(n_win):
                    start = i * window_size
                    end = min((i + 1) * window_size, time_cap)
                    flat_chunks.append(audio_el[start:end])

                    start_sec = start / audio_kwargs["sampling_rate"]
                    chunk_start_times.append(start_sec)

            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask
            audio_inputs["audio_times"] = (
                torch.as_tensor(chunk_start_times, dtype=torch.float32).unsqueeze(1) + audio_time_offsets
            )

            audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
            audio_tokens_lengths = self._get_audio_token_length(audio_lengths)

            for i, audio_length in enumerate(audio_tokens_lengths):
                text[i] = re.sub(
                    re.escape(self.audio_token),
                    self.audio_bos_token + self.audio_token * audio_length + self.audio_eos_token,
                    text[i],
                )

        text_inputs = self.tokenizer(text, **text_kwargs)

        data = {**text_inputs, **audio_inputs}
        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        fea_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tok_names + fea_names + ["input_features_mask", "audio_times"]))

    def apply_transcription_request(self, *args, **kwargs):
        raise NotImplementedError("Not needed for MusicFlamingo")

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError("Not needed for MusicFlamingo")

    def _strip_assistant_prefix_and_quotes(self, *args, **kwargs):
        raise NotImplementedError("Not needed for MusicFlamingo")


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    ori_dtype = t.dtype
    embed_dtype = torch.float64
    t = t.to(embed_dtype)
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        if freqs.ndim == 2:
            freqs = freqs[-seq_len:].to(t)
        else:
            freqs = freqs.to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


class MusicFlamingoRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: MusicFlamingoEncoderConfig, device=None):
        super().__init__(config, device=device)
        position_angles = self._compute_position_angles(self.inv_freq, device=device)
        self.register_buffer("position_angles", position_angles, persistent=False)

    def _compute_position_angles(self, inv_freq, device=None, dtype=None):
        if self.max_seq_len_cached is None:
            return None

        positions = torch.arange(
            int(self.max_seq_len_cached), device=device, dtype=dtype if dtype is not None else inv_freq.dtype
        )
        positions = positions / self.max_seq_len_cached * (2 * pi)
        position_angles = positions.unsqueeze(-1) * inv_freq
        position_angles = torch.repeat_interleave(position_angles, 2, dim=-1)
        return position_angles.to(dtype=inv_freq.dtype)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor | tuple[int, ...], seq_len=None, offset=0):
        if isinstance(t, tuple):
            Colon = slice(None)
            all_freqs = []

            for ind, dim in enumerate(t):
                pos = torch.arange(dim, device=self.inv_freq.device)
                freqs = self.forward(pos, seq_len=dim)

                all_axis = [None] * len(t)
                all_axis[ind] = Colon

                new_axis_slice = (Ellipsis, *all_axis, Colon)
                all_freqs.append(freqs[new_axis_slice])

            all_freqs = broadcast_tensors(*all_freqs)
            return torch.cat(all_freqs, dim=-1)

        if (
            seq_len is not None
            and self.position_angles is not None
            and (offset + seq_len) <= self.position_angles.shape[0]
        ):
            return self.position_angles[offset : (offset + seq_len)].detach()

        inv_freq = self.inv_freq

        # Scale time to keep t * freq <= 2pi
        if self.max_seq_len_cached is not None:
            t = t / self.max_seq_len_cached * (2 * pi)

        freqs = t.type(inv_freq.dtype).unsqueeze(-1) * inv_freq
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        return freqs


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    The audio model from MusicFlamingo without any head or projection on top.
    """
)
class MusicFlamingoEncoder(AudioFlamingo3Encoder):
    """
    MusicFlamingo encoder: Whisper encoder with rotary embeddings for time information.
    """

    def __init__(self, config: MusicFlamingoEncoderConfig):
        super().__init__(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(config)

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MusicFlamingoRotaryEmbedding):
            buffer_value = module._compute_position_angles(
                module.inv_freq, device=module.inv_freq.device, dtype=module.inv_freq.dtype
            )
            init.copy_(module.position_angles, buffer_value)

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        **kwargs,
    ):
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Log-Mel features extracted from raw audio. Use the processor/feature extractor to compute and pad
                these features from waveform input.
            input_features_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            audio_times (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                The start time of the audio segments in seconds.
        """
        seq_len = (input_features.shape[-1] - 1) // 2 + 1  # After conv2 downsampling
        input_features_lengths = input_features_mask.sum(-1)
        input_features_lengths = (input_features_lengths - 1) // 2 + 1  # conv2 downsampling
        input_features_mask = torch.arange(seq_len, device=input_features.device) < input_features_lengths[:, None]

        # Conv front-end
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Add positions, dropout
        hidden_states = inputs_embeds + self.embed_positions.weight
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        # Transformer stack
        for layer in self.layers:
            drop = self.training and torch.rand([]) < self.layerdrop
            if not drop:
                hidden_states = layer(hidden_states, attention_mask)[0]

        # AvgPool (time/2) + LayerNorm
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

        if audio_times is not None:
            times = audio_times.to(hidden_states.device)
            freqs = self.pos_emb((times.shape[0], hidden_states.shape[-2])).to(self.conv1.weight.device)
            angle = (-times * 2 * pi).to(self.conv1.weight.device)
            angle_expanded = angle.unsqueeze(2).expand(times.shape[0], hidden_states.shape[-2], freqs.shape[-1])
            freqs = freqs * angle_expanded

            hidden_states = apply_rotary_emb(freqs, hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class MusicFlamingoMultiModalProjector(AudioFlamingo3MultiModalProjector):
    pass


@auto_docstring(
    custom_intro="""
    The MusicFlamingo model which consists of a fine-tuned Whisper encoder, a multi-modal projector and a Qwen2 language model.
    """
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        audio_times: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        # Encode audio with audio_times
        encoder_output = self.audio_tower(
            input_features, input_features_mask=input_features_mask, audio_times=audio_times
        )
        audio_embeds = self.multi_modal_projector(encoder_output.last_hidden_state)

        # Mask according to avg pooling
        post_lengths = (input_features_mask.sum(-1) - 2) // 2 + 1
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_embeds = audio_embeds[valid_mask.to(audio_embeds.device)]
        return audio_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_times (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            The start time of the audio segments in seconds.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/music-flamingo-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> conversations = [
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Transcribe the input speech."},
        >>>                 {
        >>>                     "type": "audio",
        >>>                     "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav",
        >>>                 },
        >>>             ],
        >>>         }
        >>>     ],
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {
        >>>                     "type": "text",
        >>>                     "text": "This track feels really peaceful and introspective. What elements make it feel so calming and meditative?",
        >>>                 },
        >>>                 {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/FPSbCAANfbJLVSwD.mp3"},
        >>>             ],
        >>>         }
        >>>     ],
        >>> ]

        >>> inputs = processor.apply_chat_template(
        >>>     conversations,
        >>>     tokenize=True,
        >>>     add_generation_prompt=True,
        >>>     return_dict=True,
        >>> ).to(model.device)

        >>> outputs = model.generate(**inputs, max_new_tokens=500)

        >>> decoded_outputs = processor.batch_decode(
        >>>     outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        >>> )
        >>> print(decoded_outputs)
        ["The spoken content of the audio is...", "The track's calming and meditative feel can be attributed to..."]
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, audio_times=audio_times)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs: CausalLMOutputWithPast = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        audio_times = kwargs.pop("audio_times", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask
            if audio_times is not None:
                model_inputs["audio_times"] = audio_times

        return model_inputs


__all__ = [
    "MusicFlamingoConfig",
    "MusicFlamingoEncoderConfig",
    "MusicFlamingoProcessor",
    "MusicFlamingoForConditionalGeneration",
    "MusicFlamingoPreTrainedModel",
    "MusicFlamingoEncoder",
]
