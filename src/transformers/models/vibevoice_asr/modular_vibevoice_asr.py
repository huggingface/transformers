# Copyright 2026 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerPreTrainedModel,
)


logger = logging.get_logger(__name__)


class VibeVoiceAsrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceAsrForConditionalGeneration`]. It is used
    to instantiate a VibeVoice ASR model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of Microsoft's VibeVoice
    ASR architecture.

    e.g. [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        acoustic_tokenizer_encoder_config (`Union[VibeVoiceAcousticTokenizerConfig, dict]`, *optional*):
            The config object or dictionary of the acoustic tokenizer. This tokenizer extracts acoustic features from audio.
        semantic_tokenizer_encoder_config (`Union[VibeVoiceAcousticTokenizerConfig, dict]`, *optional*):
            The config object or dictionary of the semantic tokenizer. This tokenizer extracts semantic features from audio.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone (language model).
        audio_token_id (`int`, *optional*, defaults to 151648):
            The audio token index to encode the audio prompt.
        audio_bos_token_id (`int`, *optional*, defaults to 151646):
            The audio begin-of-sequence token index.
        audio_eos_token_id (`int`, *optional*, defaults to 151647):
            The audio end-of-sequence token index.
        acoustic_tokenizer_chunk_size (`int`, *optional*, defaults to 1440000):
            The chunk size (in number of samples) to use when tokenizer audio inputs. Default corresponds to 60 seconds at 24kHz.

    Example:

    ```python
    >>> from transformers import VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrConfig, VibeVoiceAcousticTokenizerEncoderConfig, Qwen2Config

    >>> # Initializing VibeVoice acoustic and semantic encoder configs
    >>> acoustic_config = VibeVoiceAcousticTokenizerEncoderConfig()
    >>> semantic_config = VibeVoiceAcousticTokenizerEncoderConfig(hidden_size=128)

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a VibeVoice ASR configuration
    >>> configuration = VibeVoiceAsrConfig(acoustic_config, semantic_config, text_config)

    >>> # Initializing a model from the vibevoice_asr style configuration
    >>> model = VibeVoiceAsrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_asr"
    sub_configs = {
        "acoustic_tokenizer_encoder_config": AutoConfig,
        "semantic_tokenizer_encoder_config": AutoConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        acoustic_tokenizer_encoder_config=None,
        semantic_tokenizer_encoder_config=None,
        text_config=None,
        audio_token_id=151648,
        audio_bos_token_id=151646,
        audio_eos_token_id=151647,
        acoustic_tokenizer_chunk_size=1440000,
        **kwargs,
    ):
        if isinstance(acoustic_tokenizer_encoder_config, dict):
            acoustic_tokenizer_encoder_config["model_type"] = acoustic_tokenizer_encoder_config.get(
                "model_type", "vibevoice_acoustic_tokenizer_encoder"
            )
            acoustic_tokenizer_encoder_config = CONFIG_MAPPING[acoustic_tokenizer_encoder_config["model_type"]](
                **acoustic_tokenizer_encoder_config
            )
        elif acoustic_tokenizer_encoder_config is None:
            acoustic_tokenizer_encoder_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer_encoder"]()
        self.acoustic_tokenizer_encoder_config = acoustic_tokenizer_encoder_config

        if isinstance(semantic_tokenizer_encoder_config, dict):
            semantic_tokenizer_encoder_config["model_type"] = semantic_tokenizer_encoder_config.get(
                "model_type", "vibevoice_acoustic_tokenizer_encoder"
            )
            semantic_tokenizer_encoder_config = CONFIG_MAPPING[semantic_tokenizer_encoder_config["model_type"]](
                **semantic_tokenizer_encoder_config
            )
        elif semantic_tokenizer_encoder_config is None:
            semantic_tokenizer_encoder_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer_encoder"](hidden_size=128)
        self.semantic_tokenizer_encoder_config = semantic_tokenizer_encoder_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.acoustic_tokenizer_chunk_size = acoustic_tokenizer_chunk_size

        super().__init__(**kwargs)


class VibeVoiceAsrRMSNorm(Qwen2RMSNorm):
    pass


class VibeVoiceAsrConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class VibeVoiceAsrMultiModalProjector(nn.Module):
    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__()
        # Acoustic path
        self.acoustic_linear_1 = nn.Linear(
            config.acoustic_tokenizer_encoder_config.hidden_size, config.text_config.hidden_size
        )
        self.acoustic_norm = VibeVoiceAsrRMSNorm(config.text_config.hidden_size, eps=1e-6)
        self.acoustic_linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

        # Semantic path
        self.semantic_linear_1 = nn.Linear(
            config.semantic_tokenizer_encoder_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_norm = VibeVoiceAsrRMSNorm(config.text_config.hidden_size, eps=1e-6)
        self.semantic_linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, acoustic_latents, semantic_latents):
        acoustic_features = self.acoustic_linear_1(acoustic_latents)
        acoustic_features = self.acoustic_norm(acoustic_features)
        acoustic_features = self.acoustic_linear_2(acoustic_features)

        semantic_features = self.semantic_linear_1(semantic_latents)
        semantic_features = self.semantic_norm(semantic_features)
        semantic_features = self.semantic_linear_2(semantic_features)

        return acoustic_features + semantic_features


@auto_docstring
class VibeVoiceAsrPreTrainedModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    config: VibeVoiceAsrConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = None
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True


@auto_docstring(
    custom_intro="""
    The VibeVoice ASR model with pre-trained acoustic tokenizers and a language model.
    """
)
class VibeVoiceAsrForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__(config)
        self.acoustic_tokenizer_encoder = AutoModel.from_config(config.acoustic_tokenizer_encoder_config)
        self.semantic_tokenizer_encoder = AutoModel.from_config(config.semantic_tokenizer_encoder_config)
        del self.audio_tower

    @can_return_tuple
    @auto_docstring(custom_intro="Encode audio into embeddings that can be used by the language model.")
    def get_audio_features(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.BoolTensor | None = None,
        acoustic_tokenizer_chunk_size: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Input audio tensor. Audio should be sampled at 24kHz.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.
        acoustic_tokenizer_chunk_size (`int`, *optional*):
            Size of audio chunks to process at once through the tokenizers. Defaults to `config.acoustic_tokenizer_chunk_size`,
            but can be modified to fit the available memory.
        """

        if acoustic_tokenizer_chunk_size is None:
            acoustic_tokenizer_chunk_size = self.config.acoustic_tokenizer_chunk_size
        else:
            if acoustic_tokenizer_chunk_size % self.config.acoustic_tokenizer_encoder_config.hop_length != 0:
                acoustic_tokenizer_chunk_size = int(
                    (acoustic_tokenizer_chunk_size // self.config.acoustic_tokenizer_encoder_config.hop_length)
                    * self.config.acoustic_tokenizer_encoder_config.hop_length
                )
                raise ValueError(
                    f"`acoustic_tokenizer_chunk_size` must be a multiple of hop length ({self.config.acoustic_tokenizer_encoder_config.hop_length}), {acoustic_tokenizer_chunk_size} is a valid option."
                )

        with torch.no_grad():
            acoustic_encoder_cache, semantic_encoder_cache = None, None
            acoustic_latents, semantic_latents = [], []

            for chunk in torch.split(input_values, acoustic_tokenizer_chunk_size, dim=-1):
                acoustic_encoder_output = self.acoustic_tokenizer_encoder(
                    chunk,
                    padding_cache=acoustic_encoder_cache,
                    use_cache=True,
                )
                acoustic_latents.append(acoustic_encoder_output.latents)
                acoustic_encoder_cache = acoustic_encoder_output.padding_cache

                semantic_encoder_output = self.semantic_tokenizer_encoder(
                    chunk,
                    padding_cache=semantic_encoder_cache,
                    use_cache=True,
                )
                semantic_latents.append(semantic_encoder_output.latents)
                semantic_encoder_cache = semantic_encoder_output.padding_cache

            acoustic_latents = torch.cat(acoustic_latents, dim=1)
            semantic_latents = torch.cat(semantic_latents, dim=1)

            # Sample acoustic tokens
            noise_std = self.config.acoustic_tokenizer_encoder_config.vae_std * torch.randn(
                acoustic_latents.shape[0], device=acoustic_latents.device, dtype=acoustic_latents.dtype
            )
            acoustic_latents = acoustic_latents + noise_std[:, None, None] * torch.randn_like(acoustic_latents)

        combined_features = self.multi_modal_projector(acoustic_latents, semantic_latents)
        if padding_mask is not None:
            # Adjust padding mask according to tokenizer compression
            num_audio_tokens = torch.ceil(
                padding_mask.sum(dim=-1) / self.config.acoustic_tokenizer_encoder_config.hop_length
            ).to(torch.int64)
            padding_mask = torch.arange(num_audio_tokens.max(), device=combined_features.device) < num_audio_tokens[
                :, None
            ].to(combined_features.device)
            combined_features = combined_features[padding_mask]

        return BaseModelOutputWithPooling(last_hidden_state=acoustic_latents, pooler_output=combined_features)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        acoustic_tokenizer_chunk_size: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.
        acoustic_tokenizer_chunk_size (`int`, *optional*):
            Size of audio chunks processed by the acoustic and semantic tokenizers. Defaults to
            `config.acoustic_tokenizer_chunk_size`, but can be modified to fit the available memory.

        Example:

        ```python
        >>> from transformers import VibeVoiceAsrForConditionalGeneration, AutoProcessor

        >>> model_id = "microsoft/VibeVoice-ASR-HF"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")

        >>> inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
        >>> inputs = inputs.to(model.device, dtype=model.dtype)
        >>> outputs = model.generate(**inputs)

        >>> decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        >>> print(decoded_outputs)
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_values is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(
                input_values=input_values,
                padding_mask=padding_mask,
                acoustic_tokenizer_chunk_size=acoustic_tokenizer_chunk_size,
            ).pooler_output

            # Replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        return self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration=False, **kwargs):
        input_values = kwargs.pop("input_values", None)
        padding_mask = kwargs.pop("padding_mask", None)
        acoustic_tokenizer_chunk_size = kwargs.pop("acoustic_tokenizer_chunk_size", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration:
            if input_values is not None:
                model_inputs["input_values"] = input_values
            if padding_mask is not None:
                model_inputs["padding_mask"] = padding_mask
            if acoustic_tokenizer_chunk_size is not None:
                model_inputs["acoustic_tokenizer_chunk_size"] = acoustic_tokenizer_chunk_size

        return model_inputs


__all__ = [
    "VibeVoiceAsrConfig",
    "VibeVoiceAsrForConditionalGeneration",
    "VibeVoiceAsrPreTrainedModel",
]
