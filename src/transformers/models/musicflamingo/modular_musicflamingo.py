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
from torch import Tensor, broadcast_tensors
from torch.amp import autocast

from ... import initialization as init
from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..llama.modeling_llama import LlamaRotaryEmbedding


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/music-flamingo-2601-hf")
class MusicFlamingoConfig(AudioFlamingo3Config):
    r"""
    audio_bos_token_id (`int`, *optional*, defaults to 151670):
            The beginning-of-audio token index used to mark the start of audio spans.
    audio_eos_token_id (`int`, *optional*, defaults to 151671):
        The end-of-audio token index used to mark the end of audio spans.
    head_dim (`int`, *optional*, defaults to 256):
        Rotary embedding dimension used per axis in [`MusicFlamingoRotaryEmbedding`]. Since the rotary embedding is
        applied on two axes (batch and time), the rotated hidden size is `2 * head_dim`, which must be less than
        or equal to `hidden_size`.
    rope_parameters (`dict`, *optional*):
        RoPE parameters for [`MusicFlamingoRotaryEmbedding`]. Supports the standard keys `"rope_type"` (defaults to
        `"default"`) and `"rope_theta"`.

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

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151669,
        audio_bos_token_id=151670,
        audio_eos_token_id=151671,
        projector_hidden_act="gelu",
        projector_bias=True,
        head_dim=256,
        rope_parameters=None,
        **kwargs,
    ):
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
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 1200,
            }
        self.rope_parameters = rope_parameters

        # NOTE for modular with `LlamaRotaryEmbedding`
        self.head_dim = head_dim
        self.max_position_embeddings = rope_parameters["rope_theta"]


class MusicFlamingoProcessorKwargs(AudioFlamingo3ProcessorKwargs): ...


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
            frames_per_window = int(
                self._get_audio_token_length(torch.tensor([self.feature_extractor.nb_max_frames], dtype=torch.long))
            )
            time_step = audio_kwargs["chunk_length"] / frames_per_window

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []
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
                    chunk_start_times.append(start / audio_kwargs["sampling_rate"])

            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask
            frame_offsets = torch.arange(frames_per_window, dtype=torch.float32) * time_step
            audio_inputs["rote_timestamps"] = (
                torch.as_tensor(chunk_start_times, dtype=torch.float32).unsqueeze(1) + frame_offsets
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
        return list(dict.fromkeys(tok_names + fea_names + ["input_features_mask", "rote_timestamps"]))

    def apply_transcription_request(self, *args, **kwargs):
        raise NotImplementedError("This method is not supported for MusicFlamingo.")

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError("This method is not supported for MusicFlamingo.")

    def _strip_assistant_prefix_and_quotes(self, *args, **kwargs):
        raise NotImplementedError("This method is not supported for MusicFlamingo.")


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_time_emb(hidden_states, cos, sin):
    """Applies rotary time embeddings to the input tensor.

    See (Goel et al., 2024) for more details: https://arxiv.org/abs/2410.12109

    Args:
        hidden_states (`torch.Tensor`):
            The input tensor.
        cos (`torch.Tensor`):
            The cosine part of the rotary embedding.
        sin (`torch.Tensor`):
            The sine part of the rotary embedding.

    Returns:
        `torch.Tensor`: The tensor with rotary time embeddings applied.
    """
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float64)
    cos = cos.to(hidden_states)
    sin = sin.to(hidden_states)
    rot_dim = cos.shape[-1]
    if rot_dim > hidden_states.shape[-1]:
        raise ValueError(
            f"feature dimension {hidden_states.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

    rotated = hidden_states[..., :rot_dim]
    passthrough = hidden_states[..., rot_dim:]
    rotated = (rotated * cos) + (rotate_half(rotated) * sin)
    return torch.cat((rotated, passthrough), dim=-1).to(original_dtype)


class MusicFlamingoRotaryEmbedding(LlamaRotaryEmbedding):
    """Rotary time embedding module for computing 2D axial rotary embeddings for batch and time dimensions,
    with time-based angle modulation.

    See (Goel et al., 2024) for more details: https://arxiv.org/abs/2410.12109
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

    @autocast("cuda", enabled=False)
    def forward(self, timestamps: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Compute 2D axial rotary embeddings for batch and time dimensions.

        Args:
            timestamps: Tensor of shape (batch_size, seq_len) containing audio timestamps in seconds.
            seq_len: Sequence length after pooling.

        Returns:
            Tuple of (cos, sin) tensors, each of shape (batch_size, seq_len, 2 * head_dim),
            computed in float64 for numerical precision.
        """

        # Compute frequencies for batch axis
        batch_positions = torch.arange(timestamps.shape[0], device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        batch_positions = batch_positions / self.max_seq_len_cached
        batch_freqs = batch_positions.unsqueeze(-1) * self.inv_freq
        batch_freqs = torch.repeat_interleave(batch_freqs, 2, dim=-1)

        # Broadcasting: batch_freqs (batch, 1, D), time_freqs (1, seq, D)
        batch_freqs = batch_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        batch_freqs, time_freqs = broadcast_tensors(batch_freqs, time_freqs)
        freqs = torch.cat((batch_freqs, time_freqs), dim=-1)

        # Apply time-based angle modulation
        angle = (-timestamps * 2 * pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
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

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        rote_timestamps: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        rote_timestamps (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
            Timestamps in seconds for each encoder output position, used to compute rotary time embeddings.
        """
        encoder_output = self.audio_tower(input_features, input_features_mask=input_features_mask)
        hidden_states = encoder_output.last_hidden_state
        if rote_timestamps is not None:
            cos, sin = self.pos_emb(rote_timestamps.to(hidden_states.device), seq_len=hidden_states.shape[-2])
            hidden_states = apply_rotary_time_emb(hidden_states, cos, sin)
        audio_embeds = self.multi_modal_projector(hidden_states)

        # Mask according to the audio tower output lengths, accounting for both conv downsampling and final avg pooling
        _, post_lengths = self.audio_tower._get_feat_extract_output_lengths(input_features_mask.sum(-1).to(torch.long))
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
        rote_timestamps: torch.Tensor | None = None,
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
        rote_timestamps (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
            Timestamps in seconds for each encoder output position, used to compute rotary time embeddings.
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
            audio_embeds = self.get_audio_features(
                input_features, input_features_mask, rote_timestamps=rote_timestamps
            )

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

    def prepare_inputs_for_generation(self, *args, is_first_iteration=False, **kwargs):
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        rote_timestamps = kwargs.pop("rote_timestamps", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration:
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask
            if rote_timestamps is not None:
                model_inputs["rote_timestamps"] = rote_timestamps

        return model_inputs


__all__ = [
    "MusicFlamingoConfig",
    "MusicFlamingoProcessor",
    "MusicFlamingoForConditionalGeneration",
    "MusicFlamingoPreTrainedModel",
]
