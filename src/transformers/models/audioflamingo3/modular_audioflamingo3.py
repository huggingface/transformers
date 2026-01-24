# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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


from math import pi

import torch
from torch import Tensor, broadcast_tensors, einsum, nn
from torch.amp import autocast
from torch.nn import Module

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioPreTrainedModel,
)
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration, VoxtralMultiModalProjector
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer
from .configuration_audioflamingo3 import AudioFlamingo3Config


logger = logging.get_logger(__name__)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


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


# learned rotation helpers
def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rotations.flatten(-2)

    rotations = torch.repeat_interleave(rotations, 2, dim=-1)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for="lang",
        theta=50000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
        max_time=7200,
    ):
        super().__init__()

        self.dim = dim
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.learned_freq = learned_freq
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        self.interpolate_factor = interpolate_factor
        self.theta_rescale_factor = theta_rescale_factor
        self.cache_if_possible = cache_if_possible
        self.max_time = max_time

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        if exists(max_time) and freqs_for == "lang":
            theta = max_time / (2 * pi)

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.theta = theta

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        self.tmp_store("dummy", torch.tensor(0))

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, (
            "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"
        )

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset
        )

        if seq_dim == -3:
            freqs = freqs.unsqueeze(1)

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = freqs.unsqueeze(1)
            scale = scale.unsqueeze(1)

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** power.unsqueeze(-1)
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
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
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        if hasattr(self, "max_time") and self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


class AudioFlamingo3Attention(WhisperAttention):
    pass


class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    pass


class AudioFlamingo3PreTrainedModel(Qwen2AudioPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights for AudioFlamingo3-specific modules."""
        if isinstance(module, RotaryEmbedding):
            # Reinitialize freqs parameter
            dim = module.dim
            freqs_for = module.freqs_for
            max_time = module.max_time
            theta_rescale_factor = module.theta_rescale_factor
            custom_freqs = None

            # Adjust theta
            if max_time is not None and freqs_for == "lang":
                theta = max_time / (2 * pi)
            else:
                theta = 50000  # default value

            theta *= theta_rescale_factor ** (dim / (dim - 2))

            # Generate freqs
            if custom_freqs is not None:
                freqs = custom_freqs
            elif freqs_for == "lang":
                freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, module.max_freq / 2, dim // 2) * pi
            elif freqs_for == "constant":
                freqs = torch.ones(module.num_freqs).float()

            module.freqs.data = freqs

            # Reinitialize dummy buffer
            module.dummy.data = torch.tensor(0)

            # Reinitialize scale if using xpos
            if module.use_xpos and module.scale is not None:
                scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
                module.scale.data = scale
        else:
            # Delegate to parent class for other modules
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    The audio model from AudioFlamingo3 without any head or projection on top.
    """
)
class AudioFlamingo3Encoder(Qwen2AudioEncoder):
    """
    AudioFlamingo3 encoder: Whisper encoder, average pool (time/2), then LayerNorm.
    """

    _can_record_outputs = {
        "hidden_states": AudioFlamingo3EncoderLayer,
        "attentions": AudioFlamingo3Attention,
    }

    def __init__(self, config: AudioFlamingo3Config):
        super().__init__(config)
        if getattr(config, "use_rotary_embedding", False):
            self.pos_emb = RotaryEmbedding(
                dim=config.rotary_dim,
                freqs_for=config.rotary_freqs_for,
                max_time=config.rotary_max_time,
            )

    @check_model_inputs
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
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
                The start time of the audio segments in seconds. Only used if rotary embeddings are enabled.
        """

        seq_len = (input_features.shape[-1] - 1) // 2 + 1  # After conv2 downsampling
        input_features_lengths = input_features_mask.sum(-1)
        input_features_lengths = (input_features_lengths - 1) // 2 + 1  # conv2 downsampling
        input_features_mask = torch.arange(seq_len, device=input_features.device) < input_features_lengths[:, None]

        # Cast to model dtype
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

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

        if (
            hasattr(self.config, "use_rotary_embedding")
            and self.config.use_rotary_embedding
            and audio_times is not None
        ):
            times = audio_times.to(hidden_states.device)
            freqs = self.pos_emb.get_axial_freqs(times.shape[0], hidden_states.shape[-2]).to(self.conv1.weight.device)
            angle = (-times * 2 * pi).to(self.conv1.weight.device)
            angle_expanded = angle.unsqueeze(2).expand(times.shape[0], hidden_states.shape[-2], freqs.shape[-1])
            freqs = freqs * angle_expanded

            hidden_states = apply_rotary_emb(freqs, hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
        )


class AudioFlamingo3MultiModalProjector(VoxtralMultiModalProjector):
    """
    Audio adaptor (small MLP) that projects AudioFlamingo3Encoder features
    to the LLM embedding space so they can replace `<sound>` tokens.
    """

    def __init__(self, config: AudioFlamingo3Config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )


@auto_docstring(
    custom_intro="""
    The AudioFlamingo3 model which consists of a fine-tuned Whisper encoder, a multi-modal projector and a Qwen2 language model.
    """
)
class AudioFlamingo3ForConditionalGeneration(VoxtralForConditionalGeneration):
    _tp_plan = None
    _pp_plan = None
    _keep_in_fp32_modules_strict = None

    def __init__(self, config):
        super().__init__(config)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        audio_times: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        audio_times (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            The start time of the audio segments in seconds.
        """

        # Encode audio
        audio_output = self.audio_tower(
            input_features,
            input_features_mask=input_features_mask,
            audio_times=audio_times,
            return_dict=True,
            **kwargs,
        )
        audio_embeds = self.multi_modal_projector(audio_output.last_hidden_state)

        # Mask according to avg pooling (which is after attention blocks)
        post_lengths = (input_features_mask.sum(-1) - 2) // 2 + 1
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_embeds = audio_embeds[valid_mask.to(audio_embeds.device)]
        audio_output.pooler_output = audio_embeds

        return audio_output

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
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_times (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            The start time of the audio segments in seconds.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/audio-flamingo-3-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

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
                input_features, input_features_mask, audio_times=audio_times, return_dict=True
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


__all__ = ["AudioFlamingo3ForConditionalGeneration", "AudioFlamingo3PreTrainedModel", "AudioFlamingo3Encoder"]
