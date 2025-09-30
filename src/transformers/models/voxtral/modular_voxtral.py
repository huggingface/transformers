# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import warnings
from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel, AutoModelForCausalLM
from ..qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioAttention,
    Qwen2AudioEncoder,
    Qwen2AudioEncoderLayer,
    Qwen2AudioPreTrainedModel,
)
from .configuration_voxtral import VoxtralConfig


class VoxtralAttention(Qwen2AudioAttention):
    pass


class VoxtralEncoderLayer(Qwen2AudioEncoderLayer):
    pass


class VoxtralPreTrainedModel(Qwen2AudioPreTrainedModel):
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _no_split_modules = None


# TODO: @eustlb, I would really prefer to use WhisperEncoder but it's messing with modular
@auto_docstring(
    custom_intro="""
    The Voxtral encoder, which is a Whisper encoder.
    """
)
class VoxtralEncoder(Qwen2AudioEncoder):
    _can_record_outputs = {
        "attentions": VoxtralAttention,
        "hidden_states": VoxtralEncoderLayer,
    }

    @check_model_inputs
    def forward(
        self,
        input_features,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Voxtral does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
        """
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Qwen2Audio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        hidden_states = (inputs_embeds + embed_pos).to(inputs_embeds.dtype)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Voxtral model, which consists of Whisper encoder, a multi-modal projector and a LLama language model.
    """
)
class VoxtralForConditionalGeneration(VoxtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules_strict = ["embed_positions"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.multi_modal_projector = VoxtralMultiModalProjector(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_audio_features(self, input_features: torch.FloatTensor):
        """
        This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector.
        Args:
            input_features (`torch.FloatTensor`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

        Returns:
            `torch.FloatTensor`:
                The audio embeddings.
        """
        audio_outputs = self.audio_tower(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    def get_audio_embeds(self, input_features: torch.FloatTensor):
        warnings.warn(
            "The method `get_audio_embeds` is deprecated. Please use `get_audio_features` instead.", FutureWarning
        )
        return self.get_audio_features(input_features)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
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
        Example:

        ```python
        >>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> repo_id = "mistralai/Voxtral-Mini-3B-2507"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

        >>> conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(conversation)
        >>> inputs = inputs.to(device, dtype=torch.bfloat16)

        >>> outputs = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            model_inputs["input_features"] = input_features

        return model_inputs


__all__ = ["VoxtralPreTrainedModel", "VoxtralEncoder", "VoxtralForConditionalGeneration"]
