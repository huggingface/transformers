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
"""PyTorch Fun-ASR-Nano model."""

from dataclasses import dataclass

import torch.nn as nn

from ... import initialization as init
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
from ...utils import auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from ..qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioAttention, Qwen3ASRAudioEncoderLayer, Qwen3ASREncoder
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import SinusoidsPositionEmbedding
from ..whisper.modeling_whisper import WhisperEncoderLayer, eager_attention_forward
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
    if isinstance(keywords, (list, tuple)) and all(isinstance(item, str) for item in keywords):
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
            max_audio_len=None,
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


@auto_docstring(
    custom_intro="""
    Base class for Fun-ASR-Nano outputs, with hidden states and attentions.
    """
)
@dataclass
class FunAsrNanoModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    pass


@auto_docstring
class FunAsrNanoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    _no_split_modules = ["FunAsrNanoEncoderStem", "FunAsrNanoEncoderLayer", "FunAsrNanoAdaptorLayer"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, SinusoidsPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.positional_embedding, position_embeddings)


class FunAsrNanoAttention(Qwen3ASRAudioAttention):
    """Qwen3-ASR attention adapted for padded batch masks and checkpoint-compatible input projections."""

    def __init__(self, config: FunAsrNanoEncoderConfig, input_dim: int | None = None):
        input_dim = input_dim or config.d_model
        super().__init__(config)
        self.q_proj = nn.Linear(input_dim, config.d_model, bias=True)
        self.k_proj = nn.Linear(input_dim, config.d_model, bias=True)
        self.v_proj = nn.Linear(input_dim, config.d_model, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(target_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(target_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(target_shape).transpose(1, 2)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=self.dropout if self.training else 0.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, sequence_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class FunAsrNanoFSMN(nn.Module):
    """Depthwise feedforward sequential memory network (FSMN) used alongside self-attention."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            config.kernel_size,
            stride=1,
            padding=0,
            groups=config.d_model,
            bias=False,
        )
        left_padding = (config.kernel_size - 1) // 2
        right_padding = config.kernel_size - 1 - left_padding
        self.pad = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = config.attention_dropout

    def forward(self, value_states: torch.Tensor, input_features_mask: torch.Tensor | None = None) -> torch.Tensor:
        # The depthwise convolution mixes neighbouring frames, so padding has to be zeroed out on both sides of it.
        if input_features_mask is not None:
            expanded_mask = input_features_mask.unsqueeze(-1).to(dtype=value_states.dtype)
            value_states = value_states * expanded_mask
        else:
            expanded_mask = None

        hidden_states = self.conv(self.pad(value_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = hidden_states + value_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if expanded_mask is not None:
            hidden_states = hidden_states * expanded_mask
        return hidden_states


class FunAsrNanoEncoderLayer(Qwen3ASRAudioEncoderLayer):
    """SAN-M encoder layer combining standard self-attention with a separate feedforward sequential memory FSMN branch.

    The two branches need different masks: `attention_mask` is the backend-specific mask built by
    `create_bidirectional_mask`, while the FSMN convolution needs the plain 2D `input_features_mask`.
    """

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.self_attn = FunAsrNanoAttention(config)
        self.feedforward_sequential_memory = FunAsrNanoFSMN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        value_states = self.self_attn.v_proj(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        fsmn_output = self.feedforward_sequential_memory(value_states, input_features_mask)
        hidden_states = residual + nn.functional.dropout(
            attention_output + fsmn_output, p=self.dropout, training=self.training
        )

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class FunAsrNanoEncoderStem(Qwen3ASRAudioEncoderLayer):
    """Position encoding and the first heterogeneous SAN-M layer."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.position_embeddings = SinusoidsPositionEmbedding(config.max_position_embeddings, config.input_size)
        self.self_attn_layer_norm = nn.LayerNorm(config.input_size)
        self.self_attn = FunAsrNanoAttention(config, input_dim=config.input_size)
        self.feedforward_sequential_memory = FunAsrNanoFSMN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states * (self.self_attn.embed_dim**0.5)
        sequence_length = hidden_states.shape[1]
        positions = self.position_embeddings(sequence_length + 1)[1:].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states + positions.unsqueeze(0)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        value_states = self.self_attn.v_proj(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        fsmn_output = self.feedforward_sequential_memory(value_states, input_features_mask)
        hidden_states = nn.functional.dropout(attention_output + fsmn_output, p=self.dropout, training=self.training)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano audio encoder (SenseVoice SAN-M architecture), without any head on top.
    """
)
class FunAsrNanoEncoder(Qwen3ASREncoder):
    _no_split_modules = ["FunAsrNanoEncoderStem", "FunAsrNanoEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": [FunAsrNanoEncoderStem, FunAsrNanoEncoderLayer],
        "attentions": FunAsrNanoAttention,
    }

    def __init__(self, config: FunAsrNanoEncoderConfig):
        PreTrainedModel.__init__(self, config)
        self.stem = FunAsrNanoEncoderStem(config)
        self.layers = nn.ModuleList([FunAsrNanoEncoderLayer(config) for _ in range(config.encoder_layers - 1)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.timestamp_prediction_layers = nn.ModuleList(
            [FunAsrNanoEncoderLayer(config) for _ in range(config.num_timestamp_prediction_blocks)]
        )
        self.timestamp_prediction_layer_norm = nn.LayerNorm(config.d_model)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.stem

    def set_input_embeddings(self, value: nn.Module):
        self.stem = value

    def _freeze_parameters(self):
        raise AttributeError("Not needed for Fun-ASR-Nano")

    def _post_cnn_length(self):
        raise AttributeError("Not needed for Fun-ASR-Nano, which has no convolutional front-end")

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        hidden_states = input_features.to(dtype=self.layer_norm.weight.dtype)

        # Every block attends over the same padded sequence, so the mask is built once here.
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        hidden_states = self.stem(hidden_states, attention_mask, input_features_mask, **kwargs)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, input_features_mask, **kwargs)

        hidden_states = self.layer_norm(hidden_states)

        for layer in self.timestamp_prediction_layers:
            hidden_states = layer(hidden_states, attention_mask, input_features_mask, **kwargs)

        hidden_states = self.timestamp_prediction_layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class FunAsrNanoAdaptorLayer(WhisperEncoderLayer):
    """Bidirectional self-attention adaptor layer."""

    def __init__(self, config: FunAsrNanoAdaptorConfig):
        super().__init__(config)
        self.self_attn = FunAsrNanoAttention(config)


class FunAsrNanoMultiModalProjector(AudioFlamingo3MultiModalProjector):
    def __init__(self, config: FunAsrNanoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.encoder_config.d_model, config.projector_hidden_size)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.adaptor_config.d_model)


class FunAsrNanoAdaptor(nn.Module):
    """Bidirectional self-attention adaptor applied to the projected audio features."""

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__()
        adaptor_config = config.adaptor_config
        adaptor_config._attn_implementation = config.encoder_config._attn_implementation
        self.config = adaptor_config
        self.blocks = nn.ModuleList(
            [FunAsrNanoAdaptorLayer(adaptor_config) for _ in range(adaptor_config.encoder_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model (SenseVoice SAN-M audio encoder, a Transformer adaptor and a Qwen3 language model),
    without a language modeling head.
    """
)
class FunAsrNanoModel(AudioFlamingo3Model):
    def __init__(self, config):
        super().__init__(config)
        self.audio_adaptor = FunAsrNanoAdaptor(config)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features, meaning inferring the audio encoder and the adaptor."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Audio features `(batch, time, feature_dim)` produced by the feature extractor (after LFR stacking).
        input_features_mask (`torch.Tensor`, *optional*):
            Padding mask for the audio feature sequence. When not provided, every sequence is assumed to be full length.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPooling`]: `last_hidden_state` holds the audio encoder output,
            `pooler_output` holds the projected audio embeddings (flattened over valid positions), and
            `hidden_states`/`attentions` hold the per-layer encoder states and attention weights.
        """
        batch_size, max_len, _ = input_features.shape
        if input_features_mask is None:
            input_features_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=input_features.device)

        encoder_outputs = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
            **kwargs,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds = self.multi_modal_projector(encoder_out)
        audio_embeds = self.audio_adaptor(audio_embeds, input_features_mask)
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
class FunAsrNanoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}


__all__ = [
    "FunAsrNanoProcessor",
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoModel",
    "FunAsrNanoForConditionalGeneration",
]
