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
from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import LlamaDecoderLayer
from ..qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioAttention, Qwen3ASREncoder
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
    _no_split_modules = ["FunAsrNanoEncoderLayer"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FunAsrNanoPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.embedding, position_embeddings)


class FunAsrNanoAttention(Qwen3ASRAudioAttention):
    """Multi-headed attention with the SAN-M FSMN value gate."""

    def __init__(
        self,
        config: FunAsrNanoEncoderConfig,
        input_dim: int | None = None,
        use_fsmn: bool = False,
    ):
        input_dim = input_dim or config.hidden_size
        super().__init__(config)
        self.dropout = config.hidden_dropout
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.fsmn = FunAsrNanoFSMN(config) if use_fsmn else None
        del self.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(target_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(target_shape).transpose(1, 2)
        projected_values = self.v_proj(hidden_states)
        value_states = projected_values.view(target_shape).transpose(1, 2)
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
        # The depthwise convolution mixes neighbouring frames, so padding has to be zeroed out on both sides of it.
        expanded_mask = input_features_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states * expanded_mask

        residual = hidden_states
        hidden_states = self.conv(self.pad(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = hidden_states + residual
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states * expanded_mask


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


class FunAsrNanoPositionEmbedding(nn.Module):
    """Scale low frame rate (LFR) features and add the fixed sinusoidal positions before the first SAN-M layer."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__()
        self.scale = config.hidden_size**0.5
        self.length = config.max_position_embeddings
        self.channels = config.num_mel_bins * config.num_stacked_frames
        self.max_timescale = 10000
        if self.channels % 2 != 0:
            raise ValueError("FunAsrNanoPositionEmbedding needs even input channels")
        self.embedding = nn.Buffer(self.compute_default_singular_positional_embedding(), persistent=False)

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        log_timescale_increment = torch.log(torch.tensor(float(self.max_timescale))) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, None] * inv_timescales[None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states * self.scale
        sequence_length = hidden_states.shape[1]
        positions = self.embedding[1 : sequence_length + 1].to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + positions.unsqueeze(0)


class FunAsrNanoEncoderLayer(LlamaDecoderLayer):
    """Shared by the audio encoder (`use_fsmn=True`) and the projector's adaptor blocks (`use_fsmn=False`)."""

    def __init__(
        self,
        config: FunAsrNanoEncoderConfig | FunAsrNanoAdaptorConfig,
        input_dim: int | None = None,
        use_fsmn: bool = True,
    ):
        input_dim = input_dim or config.hidden_size
        super().__init__(config)
        self.hidden_dropout = config.hidden_dropout
        self.self_attn = FunAsrNanoAttention(config, input_dim=input_dim, use_fsmn=use_fsmn)
        self.input_layernorm = nn.LayerNorm(input_dim)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states if hidden_states.shape[-1] == self.hidden_size else None
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        if residual is not None:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano audio encoder (SenseVoice SAN-M architecture), without any head on top.
    """
)
class FunAsrNanoEncoder(Qwen3ASREncoder):
    _no_split_modules = ["FunAsrNanoEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": FunAsrNanoEncoderLayer,
        "attentions": FunAsrNanoAttention,
    }

    def __init__(self, config: FunAsrNanoEncoderConfig):
        PreTrainedModel.__init__(self, config)
        self.position_embeddings = FunAsrNanoPositionEmbedding(config)
        self.stem = FunAsrNanoEncoderLayer(config, input_dim=config.num_mel_bins * config.num_stacked_frames)
        self.layers = nn.ModuleList([FunAsrNanoEncoderLayer(config) for _ in range(config.num_hidden_layers - 1)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.timestamp_prediction_layers = nn.ModuleList(
            [FunAsrNanoEncoderLayer(config) for _ in range(config.num_timestamp_prediction_blocks)]
        )
        self.timestamp_prediction_layer_norm = nn.LayerNorm(config.hidden_size)

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
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutput:
        hidden_states = input_features.to(dtype=self.layer_norm.weight.dtype)

        # Every block attends over the same padded sequence, so the mask is built once here.
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        hidden_states = self.position_embeddings(hidden_states)
        hidden_states = self.stem(hidden_states, attention_mask, input_features_mask, **kwargs)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, input_features_mask, **kwargs)

        hidden_states = self.layer_norm(hidden_states)

        for layer in self.timestamp_prediction_layers:
            hidden_states = layer(hidden_states, attention_mask, input_features_mask, **kwargs)

        hidden_states = self.timestamp_prediction_layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class FunAsrNanoMultiModalProjector(AudioFlamingo3MultiModalProjector):
    """Projects audio features into the text space and applies the checkpoint's adaptor blocks."""

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, config.projector_hidden_size)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.adaptor_config.hidden_size)
        adaptor_config = config.adaptor_config
        self.config = adaptor_config
        self.blocks = nn.ModuleList(
            [FunAsrNanoEncoderLayer(adaptor_config, use_fsmn=False) for _ in range(adaptor_config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
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
        self.multi_modal_projector = FunAsrNanoMultiModalProjector(config)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features, meaning inferring the audio encoder and the adaptor."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        **kwargs,
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
            return_dict=True,
            **kwargs,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds = self.multi_modal_projector(encoder_out, input_features_mask)
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
