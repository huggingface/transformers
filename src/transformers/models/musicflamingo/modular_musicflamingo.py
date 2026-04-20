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

from huggingface_hub.dataclasses import strict
from torch import Tensor, broadcast_tensors

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_available
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from ..auto import CONFIG_MAPPING
from ..moonshine.modeling_moonshine import MoonshineRotaryEmbedding


if is_torch_available():
    import torch


@auto_docstring(checkpoint="nvidia/music-flamingo-2601-hf")
@strict
class MusicFlamingoConfig(AudioFlamingo3Config):
    r"""
    audio_bos_token_id (`int`, *optional*, defaults to 151670):
        The beginning-of-audio token index used to mark the start of audio spans.
    audio_eos_token_id (`int`, *optional*, defaults to 151671):
        The end-of-audio token index used to mark the end of audio spans.
    audio_frame_step (`float`, *optional*, defaults to 0.01):
        Duration in seconds of one input mel frame (trained with hop_length 160 at sampling_rate 16000).

    Example:

    ```python
    >>> from transformers import MusicFlamingoForConditionalGeneration, MusicFlamingoConfig, AudioFlamingo3EncoderConfig, Qwen2Config

    >>> # Initializing an MusicFlamingoEncoder config
    >>> audio_config = AudioFlamingo3EncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing an MusicFlamingo configuration
    >>> configuration = MusicFlamingoConfig(audio_config, text_config)

    >>> # Initializing a model from the musicflamingo style configuration
    >>> model = MusicFlamingoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    audio_bos_token_id: int = 151670
    audio_eos_token_id: int = 151671
    audio_frame_step: float = 0.01
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            if self.audio_config["model_type"] in [None, "musicflamingo_encoder"]:
                self.audio_config["model_type"] = "audioflamingo3_encoder"

            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 1200, "partial_rotary_factor": 0.2}
        self.max_position_embeddings = self.rope_parameters["rope_theta"]
        self.head_dim = self.audio_config.hidden_size
        PreTrainedConfig.__post_init__(**kwargs)


class MusicFlamingoProcessor(AudioFlamingo3Processor):
    r"""
    Constructs an MusicFlamingo processor which wraps an MusicFlamingo feature extractor and an MusicFlamingo
    tokenizer into a single processor.

    [`MusicFlamingoProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~MusicFlamingoProcessor.__call__`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the tokenizer's default chat
            template will be used.
        audio_token (`Optional[str]`, *optional*, defaults to `"<sound>"`):
            Special token used to represent audio inputs in the chat template.
        audio_bos_token (`Optional[str]`, *optional*, defaults to `"<|sound_bos|>"`):
            Special token used to represent the beginning of audio.
        audio_eos_token (`Optional[str]`, *optional*, defaults to `"<|sound_eos|>"`):
            Special token used to represent the end of audio.
        max_audio_len (`int`, *optional*, defaults to 1200):
            Maximum length of audio sequences in seconds. Audio longer than this will be truncated.
    """

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

    def _expand_audio_tokens(self, text, padding_mask, per_sample_windows):
        audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
        audio_tokens_lengths = self._get_audio_token_length(audio_lengths)
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for i, audio_length in enumerate(audio_tokens_lengths):
            text[i] = audio_token_pattern.sub(
                self.audio_bos_token + self.audio_token * audio_length + self.audio_eos_token,
                text[i],
            )
        return text

    def _get_audio_tokens_mask(self, input_ids):
        return (
            (input_ids == self.audio_token_id)
            | (input_ids == self.audio_bos_token_id)
            | (input_ids == self.audio_eos_token_id)
        )

    def apply_transcription_request(self, *args, **kwargs):
        raise NotImplementedError("This method is not supported for MusicFlamingo.")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("MusicFlamingo does not need to overwrite this method.")

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError("MusicFlamingo does not need to overwrite this method.")

    def _strip_assistant_prefix_and_quotes(self, *args, **kwargs):
        raise NotImplementedError("This method is not supported for MusicFlamingo.")


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_time_emb(hidden_states, cos, sin):
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float64)
    cos = cos.to(hidden_states)
    sin = sin.to(hidden_states)
    rot_dim = cos.shape[-1]

    passthrough = hidden_states[..., rot_dim:]
    rotated = hidden_states[..., :rot_dim]
    rotated = (rotated * cos) + (rotate_half(rotated) * sin)
    return torch.cat((rotated, passthrough), dim=-1).to(original_dtype)


class MusicFlamingoRotaryEmbedding(MoonshineRotaryEmbedding):
    """Rotary time embedding module used by MusicFlamingo checkpoints.

    This is a checkpoint-faithful integration, not a direct implementation of the RoTE formulation described in
    (Goel et al., 2024): https://arxiv.org/abs/2410.12109. It applies axial rotary embeddings over the window index
    within each audio sample and the encoder time index within each window, then modulates both axes with absolute
    timestamps in seconds.
    """

    def __init__(self, config: MusicFlamingoConfig, device=None):
        super().__init__(config, device=device)
        position_angles = self._compute_position_angles(self.inv_freq)
        self.register_buffer("position_angles", position_angles, persistent=False)

    def _compute_position_angles(self, inv_freq):
        positions = torch.arange(int(self.max_seq_len_cached), device=inv_freq.device, dtype=inv_freq.dtype)
        positions = positions / self.max_seq_len_cached * (2 * pi)
        position_angles = positions.unsqueeze(-1) * inv_freq
        position_angles = torch.repeat_interleave(position_angles, 2, dim=-1)
        return position_angles.to(dtype=inv_freq.dtype)

    @torch.no_grad()
    def forward(self, timestamps: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Compute 2D axial rotary embeddings for window and time dimensions."""

        # Compute frequencies for the window axis, accounting for x4 due to the downsampling in the audio encoder (conv2 and avg pooling)
        window_starts = timestamps[:, 0].to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        window_duration = self.config.audio_frame_step * 4 * seq_len
        window_positions = torch.round(window_starts / window_duration) / self.max_seq_len_cached
        window_freqs = window_positions.unsqueeze(-1) * self.inv_freq
        window_freqs = torch.repeat_interleave(window_freqs, 2, dim=-1)

        # Broadcasting and apply time-based angle modulation
        window_freqs = window_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        window_freqs, time_freqs = broadcast_tensors(window_freqs, time_freqs)
        freqs = torch.cat((window_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    _no_split_modules = None

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MusicFlamingoRotaryEmbedding):
            buffer_value = module._compute_position_angles(module.inv_freq)
            init.copy_(module.position_angles, buffer_value)


@auto_docstring(
    custom_intro="""
    The MusicFlamingo model which consists of a fine-tuned Whisper encoder, rotary time embedding, a multi-modal projector, and a Qwen2 language model.
    """
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def __init__(self, config: MusicFlamingoConfig):
        super().__init__(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(config)

    def _build_audio_timestamps(
        self,
        input_ids: torch.LongTensor,
        post_lengths: torch.LongTensor,
        max_post_length: int,
    ) -> torch.FloatTensor:
        audio_token_mask = input_ids == self.config.audio_token_id
        diff = torch.diff(torch.nn.functional.pad(audio_token_mask.int(), (1, 1), value=0), dim=1)
        _, starts = torch.where(diff == 1)
        _, ends = torch.where(diff == -1)
        sample_lengths = (ends - starts).to(torch.long)

        # Account for 4x downsampling in audio encoder (conv2 and avg pooling)
        audio_embed_frame_step = self.config.audio_frame_step * 4
        frame_offsets = (
            torch.arange(max_post_length, device=post_lengths.device, dtype=torch.float32) * audio_embed_frame_step
        )

        # Map each encoder output row to its audio sample using token counts
        cumsum_post = torch.cat([torch.zeros(1, device=post_lengths.device), torch.cumsum(post_lengths, dim=0)[:-1]])
        cumsum_samples = torch.cumsum(sample_lengths, dim=0)
        sample_indices = torch.searchsorted(cumsum_samples, cumsum_post, right=True)

        # Compute window index within each sample (0, 1, 2, ... then reset for next sample)
        sample_start_rows = torch.searchsorted(
            sample_indices, torch.arange(sample_lengths.shape[0], device=post_lengths.device)
        )
        window_indices = (
            torch.arange(post_lengths.shape[0], device=post_lengths.device) - sample_start_rows[sample_indices]
        )

        # Compute timestamps
        return window_indices.unsqueeze(1) * max_post_length * audio_embed_frame_step + frame_offsets

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Token ids containing the audio token ID placeholders, for reconstructing rotary time embedding timestamps.
        """
        audio_output = self.audio_tower(
            input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
            **kwargs,
        )
        hidden_states = audio_output.last_hidden_state
        _, post_lengths = self.audio_tower._get_feat_extract_output_lengths(input_features_mask.sum(-1).to(torch.long))
        audio_timestamps = self._build_audio_timestamps(input_ids, post_lengths, hidden_states.shape[-2])
        cos, sin = self.pos_emb(audio_timestamps.to(hidden_states.device), seq_len=hidden_states.shape[-2])
        hidden_states = apply_rotary_time_emb(hidden_states, cos, sin)
        audio_embeds = self.multi_modal_projector(hidden_states)

        # Mask according to the audio tower output lengths, accounting for both conv downsampling and final avg pooling
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_output.pooler_output = audio_embeds[valid_mask.to(audio_embeds.device)]

        return audio_output

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/music-flamingo-2601-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> conversation = [
        >>>     {
        >>>         "role": "user",
        >>>         "content": [
        >>>             {
        >>>                 "type": "text",
        >>>                 "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
        >>>             },
        >>>             {
        >>>                 "type": "audio",
        >>>                 "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3",
        >>>             },
        >>>         ],
        >>>     }
        >>> ]

        >>> inputs = processor.apply_chat_template(
        >>>     conversation,
        >>>     tokenize=True,
        >>>     add_generation_prompt=True,
        >>>     return_dict=True,
        >>> ).to(model.device, model.dtype)

        >>> outputs = model.generate(**inputs, max_new_tokens=100)

        >>> decoded_outputs = processor.batch_decode(
        >>>     outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        >>> )
        >>> print(decoded_outputs)
        ["This track is an uplifting Eurodance-style Trance-Pop anthem..."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(
                input_features, input_features_mask, input_ids=input_ids, return_dict=True
            ).pooler_output

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
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs


__all__ = [
    "MusicFlamingoConfig",
    "MusicFlamingoProcessor",
    "MusicFlamingoForConditionalGeneration",
    "MusicFlamingoPreTrainedModel",
]
