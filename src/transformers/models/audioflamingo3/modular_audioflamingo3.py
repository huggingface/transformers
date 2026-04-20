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

from dataclasses import dataclass

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling, ModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioPreTrainedModel,
)
from ..voxtral.modeling_voxtral import (
    VoxtralForConditionalGeneration,
    VoxtralModel,
    VoxtralModelOutputWithPast,
    VoxtralMultiModalProjector,
)
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer
from .configuration_audioflamingo3 import AudioFlamingo3Config


logger = logging.get_logger(__name__)


class AudioFlamingo3Attention(WhisperAttention):
    pass


class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    pass


class AudioFlamingo3PreTrainedModel(Qwen2AudioPreTrainedModel):
    pass


@dataclass
class AudioFlamingo3ModelOutputWithPast(VoxtralModelOutputWithPast):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for AudioFlamingo3 causal language model (or autoregressive) outputs.
    """
)
class AudioFlamingo3CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states of the audio encoder after projection.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    audio_hidden_states: torch.FloatTensor | None = None


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

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
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
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        # Transformer stack
        for layer in self.layers:
            drop = self.training and torch.rand([]) < self.layerdrop
            if not drop:
                hidden_states = layer(hidden_states, attention_mask)

        # AvgPool (time/2) + LayerNorm
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

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
    The AudioFlamingo3 model (fine-tuned Whisper encoder, multi-modal projector, Qwen2 language model),
    without a language modeling head.
    """
)
class AudioFlamingo3Model(VoxtralModel):
    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        """

        audio_output = self.audio_tower(
            input_features, input_features_mask=input_features_mask, return_dict=True, **kwargs
        )
        audio_embeds = self.multi_modal_projector(audio_output.last_hidden_state)

        # Mask according to the audio tower output lengths, accounting for both conv downsampling and final avg pooling
        input_lengths = input_features_mask.sum(-1).to(torch.long)
        _, post_lengths = self.audio_tower._get_feat_extract_output_lengths(input_lengths)
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
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, return_dict=True).pooler_output

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return AudioFlamingo3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
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
        self.model = AudioFlamingo3Model(config)
        self.post_init()

    def get_audio_features(self, input_features, input_features_mask, **kwargs):
        return self.model.get_audio_features(input_features, input_features_mask, **kwargs)

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
    ) -> tuple | AudioFlamingo3CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/audio-flamingo-3-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return AudioFlamingo3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration: bool = False, **kwargs):
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration or not model_inputs.get("use_cache", False):
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


__all__ = [
    "AudioFlamingo3ForConditionalGeneration",
    "AudioFlamingo3PreTrainedModel",
    "AudioFlamingo3Encoder",
    "AudioFlamingo3Model",
]
