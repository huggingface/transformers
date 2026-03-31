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

import re

import torch
from huggingface_hub.dataclasses import strict

from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration
from ..auto import CONFIG_MAPPING, AutoConfig
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioPreTrainedModel
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths


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


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class Qwen3ASRProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3ASRProcessor.__call__`] and [`~Qwen3ASRProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_token = self.tokenizer.audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

    # TODO (ebezzam) could use modular from VibeVoice ASR, if we define a method `_get_feat_extract_output_lengths` for it
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | list[TextInput],
        output_labels: bool | None = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.
        """
        call_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        audio = make_list_of_audio(audio)
        if not isinstance(text, list):
            text = [text]
        if len(text) != len(audio):
            raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

        # Prepare audio
        data = self.feature_extractor(audio, **audio_kwargs)
        data["input_features_mask"] = data.pop("attention_mask")

        # Replace audio tokens in text
        audio_lengths = _get_feat_extract_output_lengths(data["input_features_mask"].sum(-1)).cpu().numpy()
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for i, num_tokens in enumerate(audio_lengths):
            text[i] = audio_token_pattern.sub(self.audio_token * int(num_tokens), text[i])

        # Prepare text
        texts_inputs = self.tokenizer(text, **text_kwargs)
        data.update(texts_inputs)

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features_mask"]))


class Qwen3ASRPreTrainedModel(Qwen2AudioPreTrainedModel):
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer", "Qwen3DecoderLayer"]
    _can_compile_fullgraph = True
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

        # Flatten batch inputs for audio encoder (matches Qwen3OmniMoe approach) -> TODO in processor instead? see audio flamingo
        audio_feature_lengths = torch.sum(input_features_mask, dim=1)
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


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRProcessor",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
]
