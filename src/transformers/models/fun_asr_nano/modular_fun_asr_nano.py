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
from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack, prepare_prompt_input
from ...utils import auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.generic import is_flash_attention_requested
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


LANGUAGE_ALIASES = {
    "zh": "中文",
    "chinese": "中文",
    "中文": "中文",
    "en": "英文",
    "english": "英文",
    "英文": "英文",
    "ja": "日文",
    "japanese": "日文",
    "日文": "日文",
}


# TODO: check other implementation
def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, 0]
    return (1.0 - mask[:, None, None, :].to(dtype=dtype)) * torch.finfo(dtype).min


class FunAsrNanoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class FunAsrNanoProcessor(AudioFlamingo3Processor):
    valid_processor_kwargs = FunAsrNanoProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|object_ref_start|>",
        **kwargs,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        """
        # Older processor configs stored this parent-class option; the chat template now owns the default prompt.
        kwargs.pop("default_transcription_prompt", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword argument {next(iter(kwargs))}.")
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            max_audio_len=None,
            **kwargs,
        )
        del self.max_audio_len
        del self.default_transcription_prompt

    def _get_audio_token_length(self, audio_lengths):
        return audio_lengths

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        if "attention_mask" not in audio_inputs:
            raise ValueError("FunAsrNanoProcessor requires an attention mask; set `return_attention_mask=True`.")
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_inputs["feature_lengths"])
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        num_audio_tokens = audio_inputs["num_audio_tokens"][audio_idx]
        return self.audio_token * num_audio_tokens

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
                Additional keyword arguments forwarded to [`~FunAsrNanoProcessor.apply_chat_template`].

        Returns:
            [`BatchFeature`]: Processor outputs ready for [`FunAsrNanoForConditionalGeneration.generate`].
        """
        audio_items = list(make_list_of_audio_chat_template(audio))
        audio_items = [
            item.detach().cpu().numpy() if is_torch_available() and isinstance(item, torch.Tensor) else item
            for item in audio_items
        ]
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        prompts = prepare_prompt_input(prompt, batch_size, input_name="prompt")
        if any(item is not None and not isinstance(item, str) for item in prompts):
            raise TypeError("Each prompt must be a string or `None`.")
        languages = prepare_prompt_input(language, batch_size, input_name="language")
        languages = [self._normalize_language(item) if item is not None else None for item in languages]
        keyword_batches = self._prepare_keyword_inputs(keywords, batch_size)

        conversations = []
        for audio_item, prompt_text, keyword_list, language_name in zip(
            audio_items, prompts, keyword_batches, languages
        ):
            content = [
                {"type": "audio", "path": audio_item}
                if isinstance(audio_item, str)
                else {"type": "audio", "audio": audio_item}
            ]
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

    @staticmethod
    def _normalize_language(language: str) -> str:
        if not isinstance(language, str):
            raise TypeError("Each language must be a string or `None`.")
        resolved = LANGUAGE_ALIASES.get(language.strip().lower())
        if resolved is None:
            raise ValueError(
                f"Unsupported language {language!r}. Use Chinese/zh, English/en, Japanese/ja, 中文, 英文, or 日文."
            )
        return resolved

    @staticmethod
    def _prepare_keyword_inputs(keywords, batch_size: int) -> list[list[str] | None]:
        if keywords is None:
            return [None] * batch_size
        if isinstance(keywords, str):
            return [[keywords]] * batch_size
        if not isinstance(keywords, (list, tuple)):
            raise TypeError("`keywords` must be a string, a sequence of strings, or a nested sequence of strings.")
        if all(isinstance(item, str) for item in keywords):
            return [list(keywords)] * batch_size
        if len(keywords) != batch_size:
            raise ValueError(
                f"Received keyword lists for {len(keywords)} samples, but the audio batch has {batch_size}."
            )

        prepared = []
        for items in keywords:
            if items is None:
                prepared.append(None)
            elif isinstance(items, (list, tuple)) and all(isinstance(item, str) for item in items):
                prepared.append(list(items))
            else:
                raise TypeError("Each per-sample keyword value must be a sequence of strings or `None`.")
        return prepared

    def decode(self, *args, strip_prefix=False, **kwargs):
        """Decode token IDs and optionally remove common assistant framing from each transcription."""
        decoded = self.tokenizer.decode(*args, **kwargs)
        if not strip_prefix:
            return decoded
        if isinstance(decoded, str):
            return self._strip_assistant_prefix_and_quotes(decoded)
        return [self._strip_assistant_prefix_and_quotes(text) for text in decoded]

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError("Not needed")

    @property
    def unused_input_names(self):
        return ["num_audio_tokens", "feature_lengths"]


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
        output_attentions: bool = False,
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
        if is_flash_attention_requested(self.config) and attention_mask is not None and attention_mask.ndim == 4:
            # Fun-ASR-Nano is bidirectional, so every query row has the same padding mask. Flash Attention expects
            # that mask as a 2D binary tensor rather than the additive 4D form used by eager and SDPA.
            attention_mask = attention_mask[:, 0, 0, :] == 0
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
        return attn_output, attn_weights if output_attentions else None


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

    def forward(self, value_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, 0]
            expanded_mask = attention_mask.unsqueeze(-1).to(dtype=value_states.dtype)
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
    """SAN-M encoder layer combining standard self-attention with a separate feedforward sequential memory FSMN branch."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.self_attn = FunAsrNanoAttention(config)
        self.feedforward_sequential_memory = FunAsrNanoFSMN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        value_states = self.self_attn.v_proj(hidden_states)
        additive_attention_mask = (
            _prepare_4d_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None
        )
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=additive_attention_mask,
            **kwargs,
        )
        fsmn_output = self.feedforward_sequential_memory(value_states, attention_mask)
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
        additive_attention_mask = (
            _prepare_4d_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None
        )
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=additive_attention_mask,
            **kwargs,
        )
        fsmn_output = self.feedforward_sequential_memory(value_states, attention_mask)
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

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        hidden_states = input_features.to(dtype=self.layer_norm.weight.dtype)

        hidden_states = self.stem(hidden_states, input_features_mask, **kwargs)

        for layer in self.layers:
            hidden_states = layer(hidden_states, input_features_mask, **kwargs)

        hidden_states = self.layer_norm(hidden_states)

        for layer in self.timestamp_prediction_layers:
            hidden_states = layer(hidden_states, input_features_mask, **kwargs)

        hidden_states = self.timestamp_prediction_layer_norm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


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
        self.blocks = nn.ModuleList(
            [FunAsrNanoAdaptorLayer(adaptor_config) for _ in range(adaptor_config.encoder_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = _prepare_4d_attention_mask(input_features_mask, hidden_states.dtype)
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
            `hidden_states` holds the per-layer encoder states.
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
