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
from torch.nn import Module

from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3Encoder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from .configuration_musicflamingo import MusicFlamingoConfig


logger = logging.get_logger(__name__)


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
        default_transcription_prompt="Transcribe the input speech.",
        max_audio_len=1200,
    ):
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
            max_audio_len=max_audio_len,
        )
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
            audio_times_list: list[torch.Tensor] = []

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
                    chunk_times = torch.arange(750).float() * 0.04 + start_sec
                    audio_times_list.append(chunk_times)

            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask
            audio_inputs["audio_times"] = torch.stack(audio_times_list).to(dtype=torch.float32)

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


# rotary embedding helper functions
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

    assert rot_dim <= t.shape[-1], (
        f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    )

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


# classes
class MusicFlamingoRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_time=7200,
    ):
        super().__init__()

        self.dim = dim
        self.max_time = max_time

        theta = max_time / (2 * pi) if max_time is not None else 50000

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        cached_freqs = self._build_cached_freqs(freqs)
        self.register_buffer("cached_freqs", cached_freqs, persistent=False)

        # dummy for device

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def _build_cached_freqs(self, freqs, device=None, dtype=None):
        if self.max_time is None:
            return None

        positions = torch.arange(int(self.max_time), device=device, dtype=dtype if dtype is not None else freqs.dtype)
        positions = positions / self.max_time * (2 * pi)
        cached_freqs = positions.unsqueeze(-1) * freqs
        return torch.repeat_interleave(cached_freqs, 2, dim=-1)

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        if seq_len is not None and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        # Scale time to keep t * freq <= 2pi
        if self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = t.type(freqs.dtype).unsqueeze(-1) * freqs
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        return freqs


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights for MusicFlamingo-specific modules."""
        parent_init_weights = super()._init_weights
        parent_init_weights(module)

        if isinstance(module, MusicFlamingoRotaryEmbedding):
            # Reinitialize freqs parameter
            dim = module.dim
            max_time = module.max_time

            theta = max_time / (2 * pi) if max_time is not None else 50000

            # Generate freqs
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

            module.freqs.data = freqs

            module.cached_freqs = module._build_cached_freqs(
                module.freqs, device=module.freqs.device, dtype=module.freqs.dtype
            )

            # Reinitialize dummy buffer
            module.dummy.data = torch.tensor(0)


@auto_docstring(
    custom_intro="""
    The audio model from MusicFlamingo without any head or projection on top.
    """
)
class MusicFlamingoEncoder(AudioFlamingo3Encoder):
    """
    MusicFlamingo encoder: Whisper encoder with rotary embeddings for time information.
    """

    def __init__(self, config: MusicFlamingoConfig):
        super().__init__(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(dim=256, max_time=1200.0)

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
            freqs = self.pos_emb.get_axial_freqs(times.shape[0], hidden_states.shape[-2]).to(self.conv1.weight.device)
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
    "MusicFlamingoProcessor",
    "MusicFlamingoForConditionalGeneration",
    "MusicFlamingoPreTrainedModel",
    "MusicFlamingoEncoder",
]
