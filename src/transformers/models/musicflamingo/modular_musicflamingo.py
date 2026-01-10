# coding=utf-8
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

from math import pi
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3Encoder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from .configuration_musicflamingo import MusicFlamingoConfig
from .rotary_embedding import RotaryEmbedding, apply_rotary_emb


logger = logging.get_logger(__name__)


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights for MusicFlamingo-specific modules."""
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
    The audio model from MusicFlamingo without any head or projection on top.
    """
)
class MusicFlamingoEncoder(AudioFlamingo3Encoder):
    """
    MusicFlamingo encoder: Whisper encoder with rotary embeddings for time information.
    """

    def __init__(self, config: MusicFlamingoConfig):
        super().__init__(config)
        self.pos_emb = RotaryEmbedding(dim=256, freqs_for="lang", max_time=1200.0)

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
        audio_times: Optional[torch.Tensor] = None,
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
            angle = (-times * 2 * np.pi).to(self.conv1.weight.device)
            # audio_times is [batch_size], need to expand to [batch_size, seq_len, freq_dim]
            angle_expanded = (
                angle.unsqueeze(1).unsqueeze(2).expand(times.shape[0], hidden_states.shape[-2], freqs.shape[-1])
            )
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
        audio_times: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # Encode audio with dtype conversion and audio_times
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype)
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
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        audio_times: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
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


__all__ = ["MusicFlamingoForConditionalGeneration", "MusicFlamingoPreTrainedModel", "MusicFlamingoEncoder"]
