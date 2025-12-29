# coding=utf-8
# Copyright 2025 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoModel
from ..vibevoice.modeling_vibevoice import VibeVoicePreTrainedModel, VibeVoiceModel
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerModel
from .configuration_vibevoice_realtime import VibeVoiceRealTimeConfig, VibeVoiceRealTimeAcousticDecoderConfig
from .generation_vibevoice_realtime import VibeVoiceRealTimeGenerationMixin


@dataclass
class VibeVoiceRealTimeCausalLMOutputWithPast(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class VibeVoiceRealTimeBinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states = torch.relu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return hidden_states


@auto_docstring
class VibeVoiceRealTimePreTrainedModel(VibeVoicePreTrainedModel):
    config: VibeVoiceRealTimeConfig
    input_modalities = "text"


@auto_docstring(
    custom_intro="""
    Acoustic tokenizer which only decodes audio from latent representations.
    """
)
class VibeVoiceRealTimeAcousticDecoder(VibeVoiceAcousticTokenizerModel):
    config: VibeVoiceRealTimeAcousticDecoderConfig
    base_model_prefix = "vibevoice_realtime_acoustic_decoder"
    main_input_name = "latents"
    _no_split_modules = ["VibeVoiceRealTimeDecoder"]

    def __init__(self, config):
        super().__init__(config)
        del self.encoder
        del self.vae_std

    def encode(self, audio, padding_cache=None, use_cache=None):
        raise NotImplementedError("Encode method is not implemented for VibeVoiceRealTimeAcousticDecoder.")

    def forward(self, audio, padding_cache=None, use_cache=None, **kwargs: Unpack[TransformersKwargs]):
        raise NotImplementedError("Encode method is not implemented for VibeVoiceRealTimeAcousticDecoder.")


@auto_docstring(
    custom_intro="""
    The VibeVoice model which consists of an audio decoder and an LLM backbone, without a language modeling head.
    """
)
class VibeVoiceRealTimeModel(VibeVoiceModel):
    def __init__(self, config):
        super().__init__(config)
        del self.semantic_tokenizer
        del self.semantic_connector

        self.language_model.norm = nn.Identity()
        self.tts_language_model = AutoModel.from_config(config.tts_text_config)
        # NOTE: Marks the text that needs to be spoken by the TTS model.
        self.tts_input_types = nn.Embedding(num_embeddings=2, embedding_dim=config.text_config.hidden_size)

    def get_audio_features(self, input_features, input_features_mask, latent_scaling_factor, latent_bias_factor):
        raise NotImplementedError("get_audio_features method is not implemented for VibeVoiceRealTimeModel.")

    # TODO eventually remove / merge into forward? But needs it's own model_kwargs
    @can_return_tuple
    def forward_lm(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return self.language_model(inputs_embeds=inputs_embeds, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        tts_text_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            # TODO can input embeds be computed by self.language model if only input_ids is provided?
            raise ValueError("Input embeds should be computed by with `self.forward_lm`")

        inputs_embeds = inputs_embeds + self.tts_input_types(tts_text_masks)
        return self.tts_language_model(inputs_embeds=inputs_embeds, **kwargs)


@auto_docstring(
    custom_intro="""
    The VibeVoice model, which consists of a language model, speech tokenizers, connectors, and a diffusion head.
    """
)
class VibeVoiceRealTimeForConditionalGeneration(VibeVoiceRealTimePreTrainedModel, VibeVoiceRealTimeGenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceRealTimeModel(config)
        self.latent_scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.latent_bias_factor = nn.Parameter(torch.tensor(0.0))
        self.tts_eos_classifier = VibeVoiceRealTimeBinaryClassifier(config.text_config.hidden_size)
        self.post_init()

    @property
    def language_model(self):
        return self.model.language_model
    
    @property
    def tts_language_model(self):
        return self.model.tts_language_model
    
    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def diffusion_head(self):
        return self.model.diffusion_head
    
    # TODO eventually remove / merge into forward? But needs it's own model_kwargs
    def forward_lm(self, *args, **kwargs):
        return self.model.forward_lm(*args, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        tts_text_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceRealTimeCausalLMOutputWithPast]:
        """
        tts_text_masks (`torch.FloatTensor`, *optional*):
            Mask marking current position as text(1)/speech(0)
        """
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds, 
            tts_text_masks=tts_text_masks, 
            **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.tts_eos_classifier(last_hidden_state[:, -1, :])
                
        loss = None
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibeVoiceRealTimeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=last_hidden_state,
            attentions=outputs.attentions,
        )


__all__ = ["VibeVoiceRealTimeForConditionalGeneration", "VibeVoiceRealTimePreTrainedModel", "VibeVoiceRealTimeModel", "VibeVoiceRealTimeAcousticDecoder"]
