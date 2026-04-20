# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioPreTrainedModel


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict
class Qwen3ASRConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 151676):
        The audio token id to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRConfig

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151676
    pad_token_id: int = 151645
    eos_token_id: list[int] | tuple[int, ...] | int = (151643, 151645)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "qwen3_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen3_audio_encoder"](
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                d_model=1024,
                output_dim=2048,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=65536,
                tie_word_embeddings=True,
            )

        super().__post_init__(**kwargs)


class Qwen3ASRPreTrainedModel(Qwen2AudioPreTrainedModel):
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer", "Qwen3DecoderLayer"]
    _can_compile_fullgraph = False  # Audio encoder has data-dependent ops (same as Qwen3OmniMoe)
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3ASR model which consists of an audio encoder and a language model.
    """
)
class Qwen3ASRForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        del self.multi_modal_projector

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
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
        """
        # Flatten batched features for the Qwen3OmniMoe audio encoder
        audio_feature_lengths = input_features_mask.sum(dim=1)
        input_features = input_features.permute(0, 2, 1)[input_features_mask.bool()].permute(1, 0)

        audio_output = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            **kwargs,
        )
        audio_output.pooler_output = audio_output.last_hidden_state
        return audio_output

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
    ):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            input_features=input_features,
            input_features_mask=input_features_mask,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


@auto_docstring(checkpoint="bezzam/Qwen3-ForcedAligner-0.6B")
@strict
class Qwen3ForcedAlignerConfig(Qwen3ASRConfig):
    r"""
    classify_num (`int`, *optional*, defaults to 5000):
        Number of classification labels for forced alignment.
    timestamp_token_id (`int`, *optional*, defaults to 151705):
        Token ID for timestamp markers in the alignment output.
    timestamp_segment_time (`int`, *optional*, defaults to 80):
        Time segment (in milliseconds) that each timestamp token represents.

    Example:

    ```python
    >>> from transformers import Qwen3ForcedAlignerForTokenClassification, Qwen3ForcedAlignerConfig

    >>> # Initializing a Qwen3ForcedAligner style configuration
    >>> configuration = Qwen3ForcedAlignerConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ForcedAlignerForTokenClassification(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_forced_aligner"

    classify_num: int = 5000
    timestamp_token_id: int = 151705
    timestamp_segment_time: int = 80


class Qwen3ForcedAlignerPreTrainedModel(Qwen3ASRPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    The Qwen3 Forced Aligner model which consists of an audio encoder, a language model backbone,
    and a token classification head for forced alignment.
    """
)
class Qwen3ForcedAlignerForTokenClassification(Qwen3ForcedAlignerPreTrainedModel):
    def __init__(self, config: Qwen3ForcedAlignerConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.classify_num = config.classify_num
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.model = AutoModel.from_config(config.text_config)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        """
        # Flatten batched features for the Qwen3OmniMoe audio encoder
        audio_feature_lengths = input_features_mask.sum(dim=1)
        input_features = input_features.permute(0, 2, 1)[input_features_mask.bool()].permute(1, 0)

        audio_output = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            **kwargs,
        )
        audio_output.pooler_output = audio_output.last_hidden_state
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.classify_num - 1]`.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, return_dict=True).pooler_output

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.classify_num)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ForcedAlignerConfig",
    "Qwen3ForcedAlignerForTokenClassification",
    "Qwen3ForcedAlignerPreTrainedModel",
]
