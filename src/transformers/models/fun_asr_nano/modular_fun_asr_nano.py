# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable

import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...audio_utils import (
    AudioInput,
    make_audio_chat_template_content,
    make_list_of_audio_chat_template,
    prepare_language_inputs,
)
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack, prepare_prompt_input
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.generic import merge_with_config_defaults, no_inherit_decorator
from ...utils.output_capturing import capture_outputs
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3Model,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from ..qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    SinusoidsPositionEmbedding,
)
from ..whisper.modeling_whisper import eager_attention_forward
from .configuration_fun_asr_nano import FunAsrNanoAdaptorConfig, FunAsrNanoConfig, FunAsrNanoEncoderConfig


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


# The model was trained with these Chinese names in the transcription instruction ("语音转写成<NAME>：").
# The English names are aliases, so that both `"en"` and `"English"` resolve to the checkpoint's name.
LANGUAGE_CODE_TO_NAME = {
    "zh": "中文",
    "chinese": "中文",
    "en": "英文",
    "english": "英文",
    "ja": "日文",
    "japanese": "日文",
}


def _prepare_keyword_inputs(keywords, batch_size: int) -> list[list[str] | None]:
    """Broadcast / validate the hotword argument to match batch_size."""
    if isinstance(keywords, str):
        keywords = [keywords]
    if isinstance(keywords, list | tuple) and all(isinstance(item, str) for item in keywords):
        keywords = [list(keywords)] * batch_size
    return prepare_prompt_input(keywords, batch_size, input_name="keywords")


class FunAsrNanoProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "audio_kwargs": {"sampling_rate": 16000},
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class FunAsrNanoProcessor(AudioFlamingo3Processor):
    valid_processor_kwargs = FunAsrNanoProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|object_ref_start|>",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        """
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
        )
        del self.max_audio_len
        del self.default_transcription_prompt

    def _get_audio_token_length(self, audio_lengths):
        raise AttributeError("Not needed for Fun-ASR-Nano")

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        if "input_features_mask" not in audio_inputs:
            raise ValueError("FunAsrNanoProcessor requires an audio padding mask; set `return_attention_mask=True`.")
        audio_inputs["num_audio_tokens"] = audio_inputs["input_features_mask"].sum(-1)
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        language: str | list[str] | None = None,
        prompt: str | list[str] | None = None,
        keywords: str | list[str] | list[list[str]] | None = None,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        """Prepare inputs for ASR using the checkpoint's structured transcription chat template.

        Args:
            audio (`AudioInput` or `list[AudioInput]`):
                Audio to transcribe. Can be a URL, local path, NumPy array, PyTorch tensor, or a list of these.
            language (`str` or `list[str]`, *optional*):
                Target language. Accepts Chinese, English, or Japanese as full English names, ISO codes (`"zh"`,
                `"en"`, `"ja"`), or the checkpoint's Chinese language names (`"中文"`, `"英文"`, `"日文"`). A
                single value is broadcast across the batch.
            prompt (`str` or `list[str]`, *optional*):
                Contextual information that may improve transcription. A list must match the audio batch size.
            keywords (`str`, `list[str]`, or `list[list[str]]`, *optional*):
                Hotwords to bias recognition. A string or flat list is shared across the batch; a nested list
                supplies separate hotwords for each audio sample.
            **kwargs:
                Additional keyword arguments forwarded to [`~FunAsrNanoProcessor.apply_chat_template`]
                and the underlying processor call (for example `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`FunAsrNanoForConditionalGeneration.generate`].
        """
        audio_items = list(make_list_of_audio_chat_template(audio))

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        languages = prepare_language_inputs(language, batch_size, LANGUAGE_CODE_TO_NAME, return_code=False)
        prompts = prepare_prompt_input(prompt, batch_size, input_name="prompt")
        keyword_batches = _prepare_keyword_inputs(keywords, batch_size)

        conversations = []
        for audio_item, prompt_text, keyword_list, language_name in zip(
            audio_items, prompts, keyword_batches, languages
        ):
            content = [make_audio_chat_template_content(audio_item)]
            if prompt_text is not None:
                content.append({"type": "text", "text": prompt_text})
            if keyword_list:
                content.append({"type": "keywords", "keywords": keyword_list})
            if language_name is not None:
                content.append({"type": "language", "language": language_name})
            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def decode(self, *args, strip_prefix=False, **kwargs):
        """Decode token IDs and optionally remove common assistant framing from each transcription."""
        decoded = self.tokenizer.decode(*args, **kwargs)
        if not strip_prefix:
            return decoded
        if isinstance(decoded, str):
            return self._strip_assistant_prefix_and_quotes(decoded)
        return [self._strip_assistant_prefix_and_quotes(text) for text in decoded]

    def batch_decode(self, *args, **kwargs):
        raise AttributeError("Not needed")


@auto_docstring
class FunAsrNanoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FunAsrNanoPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.positional_embedding, position_embeddings)


@no_inherit_decorator
class FunAsrNanoAttention(LlamaAttention):
    """Multi-headed attention with the SAN-M FSMN value gate."""

    def __init__(
        self,
        config: FunAsrNanoEncoderConfig | FunAsrNanoAdaptorConfig,
        layer_idx: int | None = None,
        hidden_size: int | None = None,
        use_fsmn: bool = False,
    ):
        hidden_size = hidden_size or config.hidden_size
        super().__init__(config, layer_idx)
        self.num_key_value_groups = 1  # the model has no GQA
        self.is_causal = False
        self.q_proj = nn.Linear(hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.fsmn = FunAsrNanoFSMN(config) if use_fsmn else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # Keep the unsplit values for the FSMN convolution across neighboring frames.
        projected_values = self.v_proj(hidden_states)
        value_states = projected_values.view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.fsmn is not None:
            # SAN-M gate: the FSMN branch runs on the values before they are split into heads.
            attn_output = attn_output + self.fsmn(projected_values, input_features_mask)
        return attn_output, attn_weights


class FunAsrNanoFSMN(nn.Module):
    """Depthwise feedforward sequential memory network (FSMN) used alongside self-attention."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.fsmn_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        left_padding = (config.fsmn_kernel_size - 1) // 2
        right_padding = config.fsmn_kernel_size - 1 - left_padding
        self.pad = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = input_features_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states * expanded_mask

        residual = hidden_states
        hidden_states = self.conv(self.pad(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = hidden_states + residual
        return nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)


class FunAsrNanoMLP(CLIPMLP):
    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        return nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)


class FunAsrNanoPositionEmbedding(SinusoidsPositionEmbedding):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # FunASR's `SinusoidalPositionEncoder` counts positions from one.
        positions = self.positional_embedding[1 : hidden_states.shape[1] + 1]
        return positions.to(device=hidden_states.device, dtype=hidden_states.dtype)


class FunAsrNanoEncoderLayer(LlamaDecoderLayer):
    """Shared by the audio encoder (`use_fsmn=True`) and the projector's adaptor layers (`use_fsmn=False`)."""

    def __init__(
        self,
        config: FunAsrNanoEncoderConfig | FunAsrNanoAdaptorConfig,
        hidden_size: int | None = None,
        use_fsmn: bool = True,
        add_norm: bool = False,
    ):
        hidden_size = hidden_size or config.hidden_size
        super().__init__(config)
        self.hidden_dropout = config.hidden_dropout
        self.self_attn = FunAsrNanoAttention(config, hidden_size=hidden_size, use_fsmn=use_fsmn)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Only the last transcription block and the last timestamp prediction block normalize their output.
        self.final_layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if add_norm else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # No residual for the very first layer (low frame rate audio features have a different input dimension than the hidden size)
        residual = hidden_states if hidden_states.shape[-1] == self.hidden_size else None
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        if residual is not None:
            hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano audio encoder (SenseVoice SAN-M architecture), without any head on top.
    """
)
class FunAsrNanoEncoder(FunAsrNanoPreTrainedModel):
    config: FunAsrNanoEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["FunAsrNanoEncoderLayer"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": FunAsrNanoEncoderLayer,
        "attentions": FunAsrNanoAttention,
    }

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.position_embeddings = FunAsrNanoPositionEmbedding(config.max_position_embeddings, config.input_size)
        self.scaling = config.hidden_size**0.5

        # Only last layer for transcription, and last layer for timestamp prediction, are layer normalized.
        num_transcription_layers = config.num_hidden_layers - config.num_timestamp_prediction_layers
        layer_norm_indices = (num_transcription_layers - 1, config.num_hidden_layers - 1)

        self.layers = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    config,
                    # Only first layer has different input size (for the low frame rate audio features)
                    hidden_size=config.input_size if layer_idx == 0 else None,
                    add_norm=layer_idx in layer_norm_indices,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
            1 for valid mel frames and 0 for padding.
        """
        hidden_states = input_features.to(dtype=self.layers[0].input_layernorm.weight.dtype)

        # Every block attends over the same padded sequence, so the mask is built once here.
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        hidden_states = hidden_states * self.scaling + self.position_embeddings(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, input_features_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


class FunAsrNanoMultiModalProjector(nn.Module):
    """Projects audio features into the text space and applies the checkpoint's adaptor layers."""

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, config.adaptor_config.projector_hidden_size)
        self.act = ACT2FN[config.adaptor_config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.adaptor_config.projector_hidden_size, config.adaptor_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(config.adaptor_config, use_fsmn=False)
                for _ in range(config.adaptor_config.num_hidden_layers)
            ]
        )
        self.config = config.adaptor_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, **kwargs)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model (SenseVoice SAN-M audio encoder, a Transformer adaptor and a Qwen3 language model),
    without a language modeling head.
    """
)
class FunAsrNanoModel(AudioFlamingo3Model):
    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features, meaning inferring the audio encoder and the adaptor."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Audio features `(batch, time, feature_dim)` produced by the feature extractor (after LFR stacking).
        input_features_mask (`torch.Tensor`):
            Padding mask for the audio feature sequence, as returned by the processor.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPooling`]: `last_hidden_state` holds the audio encoder output,
            `pooler_output` holds the projected audio embeddings (flattened over valid positions), and
            `hidden_states`/`attentions` hold the per-layer encoder states and attention weights.
        """
        encoder_outputs = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds = self.multi_modal_projector(encoder_out, input_features_mask, **kwargs)
        pooler_output = audio_embeds[input_features_mask.to(device=audio_embeds.device, dtype=torch.bool)]

        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_out,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model for speech recognition: a SenseVoice SAN-M audio encoder, a Transformer adaptor and a
    Qwen3 language model with a language modeling head.
    """
)
class FunAsrNanoForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import FunAsrNanoForConditionalGeneration, AutoProcessor

        >>> model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "FunAsrNanoProcessor",
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoModel",
    "FunAsrNanoForConditionalGeneration",
]
