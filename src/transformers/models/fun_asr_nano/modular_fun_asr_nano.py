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
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...audio_utils import AudioInput
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.generic import no_inherit_decorator
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from ..parakeet.modeling_parakeet import ParakeetEncoderFeedForward
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import SinusoidsPositionEmbedding
from ..voxtral.modeling_voxtral import VoxtralMultiModalProjector


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - mask.unsqueeze(1).to(dtype=dtype)) * torch.finfo(dtype).min


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoEncoder`]. It is used to instantiate a
    Fun-ASR-Nano audio encoder (a SenseVoice SAN-M encoder) according to the specified arguments, defining the model
    architecture.

    input_size (`int`, *optional*, defaults to 560):
        Input feature dimension (after LFR: 80 mel bins * 7 frames = 560).
    output_dim (`int`, *optional*, defaults to 512):
        Hidden size of the encoder layers.
    num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in each SANM layer.
    intermediate_size (`int`, *optional*, defaults to 2048):
        Dimension of the feedforward layer.
    encoder_layers (`int`, *optional*, defaults to 50):
        Number of main encoder blocks.
    tp_blocks (`int`, *optional*, defaults to 20):
        Number of timestamp prediction encoder blocks.
    dropout (`float`, *optional*, defaults to 0.1):
        Dropout rate.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        Attention dropout rate.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the FSMN convolution.
    sanm_shift (`int`, *optional*, defaults to 0):
        Shift for asymmetric padding in FSMN.
    Example:

    ```python
    >>> from transformers import FunAsrNanoEncoderConfig, FunAsrNanoEncoder

    >>> configuration = FunAsrNanoEncoderConfig()
    >>> model = FunAsrNanoEncoder(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano_encoder"
    attribute_map = {
        "output_size": "output_dim",
        "attention_heads": "num_attention_heads",
        "linear_units": "intermediate_size",
        "num_blocks": "encoder_layers",
        "dropout_rate": "dropout",
        "attention_dropout_rate": "attention_dropout",
    }

    input_size: int = 560
    output_dim: int = 512
    num_attention_heads: int = 4
    intermediate_size: int = 2048
    encoder_layers: int = 50
    tp_blocks: int = 20
    dropout: float = 0.1
    attention_dropout: float = 0.0
    kernel_size: int = 11
    sanm_shift: int = 0


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoForConditionalGeneration`]. It is used
    to instantiate a Fun-ASR-Nano model according to the specified arguments, defining the model architecture.

    audio_encoder_config (`dict` or `FunAsrNanoEncoderConfig`, *optional*):
        Configuration for the audio encoder.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the language model (Qwen3).
    audio_token_index (`int`, *optional*, defaults to 151646):
        Token ID used as placeholder for audio features.
    adaptor_downsample_rate (`int`, *optional*, defaults to 1):
        Downsampling factor applied to the encoder sequence before projecting to the language model.
    adaptor_intermediate_size (`int`, *optional*, defaults to 2048):
        Hidden size of the adaptor feed-forward projection.
    adaptor_num_hidden_layers (`int`, *optional*, defaults to 2):
        Number of adaptor transformer layers.
    adaptor_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the adaptor transformer layers.
    adaptor_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability used in the adaptor.
    Example:

    ```python
    >>> from transformers import FunAsrNanoConfig, FunAsrNanoForConditionalGeneration

    >>> configuration = FunAsrNanoConfig()
    >>> model = FunAsrNanoForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano"
    attribute_map = {
        "audio_config": "audio_encoder_config",
        "audio_token_id": "audio_token_index",
        "adaptor_ffn_dim": "adaptor_intermediate_size",
        "adaptor_num_layers": "adaptor_num_hidden_layers",
        "adaptor_attention_heads": "adaptor_num_attention_heads",
        "adaptor_dropout_rate": "adaptor_dropout",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "audio_encoder_config": FunAsrNanoEncoderConfig,
    }

    audio_encoder_config: dict | FunAsrNanoEncoderConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646
    adaptor_downsample_rate: int = 1
    adaptor_intermediate_size: int = 2048
    adaptor_num_hidden_layers: int = 2
    adaptor_num_attention_heads: int = 8
    adaptor_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config["model_type"] = self.audio_encoder_config.get(
                "model_type", "fun_asr_nano_encoder"
            )
            self.audio_encoder_config = FunAsrNanoEncoderConfig(**self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = FunAsrNanoEncoderConfig()

        if isinstance(self.text_config, dict):
            text_config_model_type = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[text_config_model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


class FunAsrNanoProcessorKwargs(AudioFlamingo3ProcessorKwargs):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "return_attention_mask": True,
            "padding": True,
        },
        "text_kwargs": {
            "padding": True,
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
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
        audio_downsample_rate=1,
        default_transcription_prompt="Transcribe the audio:",
        max_audio_len=None,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        audio_downsample_rate (`int`, *optional*, defaults to 1):
            Downsampling ratio applied by the audio adaptor, used to expand the audio placeholder token to the right
            number of audio tokens.
        default_transcription_prompt (`str`, *optional*, defaults to `"Transcribe the audio:"`):
            Default prompt to use for transcription tasks when applying transcription requests.
        max_audio_len (`int`, *optional*):
            Maximum audio duration in seconds. `None` disables the inherited duration limit.
        """
        if tokenizer.convert_tokens_to_ids(audio_token) is None:
            raise ValueError(f"Audio token {audio_token!r} is not present in the tokenizer vocabulary.")

        self.audio_downsample_rate = audio_downsample_rate
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
            max_audio_len=max_audio_len,
        )

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (`bool`, *optional*, defaults to `False`):
            Whether to create causal-language-model labels from text token IDs. Padding and audio placeholder positions
            are masked with `-100`.
        """
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(text=text, audio=audio, **kwargs)

        if output_labels:
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids")
            labels = model_inputs["input_ids"].clone()
            labels.masked_fill_(mm_token_type_ids != 0, -100)
            if "attention_mask" in model_inputs:
                labels.masked_fill_(model_inputs["attention_mask"] == 0, -100)
            model_inputs["labels"] = labels
        return model_inputs

    def _get_audio_token_length(self, audio_lengths):
        return (audio_lengths - 1) // self.audio_downsample_rate + 1

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        if "attention_mask" not in audio_inputs:
            raise ValueError("FunAsrNanoProcessor requires an attention mask; set `return_attention_mask=True`.")
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_inputs["feature_lengths"])
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def decode(self, *args, strip_prefix=False, **kwargs):
        """Decode token IDs and optionally remove common assistant framing from each transcription."""
        decoded = self.tokenizer.decode(*args, **kwargs)
        if not strip_prefix:
            return decoded
        if isinstance(decoded, str):
            return self._strip_assistant_prefix_and_quotes(decoded)
        return [self._strip_assistant_prefix_and_quotes(text) for text in decoded]

    @property
    def unused_input_names(self):
        return ["num_audio_tokens", "feature_lengths"]

    def apply_transcription_request(self, **super_kwargs):
        r"""
        prompt (`str` or `list[str]`, *optional*):
            Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
            each sample uses `"Transcribe the audio:"`.
        """
        return super().apply_transcription_request(**super_kwargs)


@auto_docstring(
    custom_intro="""
    Base class for Fun-ASR-Nano outputs, with hidden states and attentions.
    """
)
@dataclass
class FunAsrNanoModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        Projected audio embeddings produced by the audio encoder and adaptor.
    """

    audio_hidden_states: torch.FloatTensor | None = None


class FunAsrNanoSinusoidalPositionEncoder(SinusoidsPositionEmbedding):
    """Fun-ASR-Nano sinusoidal positional encoding.

    The shared helper starts at position 0, while the original FunASR encoder starts at position 1.
    """

    def __init__(self, channels: int, length: int = 2049, max_timescale: int = 10000):
        super().__init__(length, channels, max_timescale=max_timescale)

    def _resize(self, length: int):
        expanded = SinusoidsPositionEmbedding(length, self.channels, self.max_timescale)
        self.length = length
        self.register_buffer(
            "positional_embedding",
            expanded.positional_embedding.to(
                device=self.positional_embedding.device, dtype=self.positional_embedding.dtype
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, timesteps, input_dim = x.size()
        if input_dim != self.channels:
            raise ValueError(f"Expected input dimension {self.channels}, but received {input_dim}.")
        if timesteps + 1 > self.length:
            self._resize(timesteps + 1)

        encoding = self.positional_embedding[: timesteps + 1][1:].to(device=x.device, dtype=x.dtype)
        return x + encoding.unsqueeze(0)


@no_inherit_decorator
class FunAsrNanoAttention(LlamaAttention):
    pass


class FunAsrNanoSANMAttention(FunAsrNanoAttention):
    """Self-Attention with FSMN Memory (SANM).

    State dict keys:
        self_attn.linear_q_k_v.{weight,bias}
        self_attn.linear_out.{weight,bias}
        self_attn.fsmn_block.weight  (Conv1d depthwise, no bias)
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        nn.Module.__init__(self)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_key_value_groups = 1
        self.scaling = self.d_k**-0.5

        self.linear_q_k_v = nn.Linear(in_features, hidden_size * 3)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        # FSMN depthwise conv (key: self_attn.fsmn_block.weight)
        self.fsmn_block = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, stride=1, padding=0, groups=hidden_size, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

        self.dropout = nn.Dropout(p=attention_dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.size()

        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, self.hidden_size, dim=-1)

        # FSMN memory path
        if mask is not None:
            mask_expanded = mask.view(b, -1, 1)
            v_masked = v * mask_expanded
        else:
            v_masked = v

        fsmn_out = v_masked.transpose(1, 2)
        fsmn_out = self.pad_fn(fsmn_out)
        fsmn_out = self.fsmn_block(fsmn_out)
        fsmn_out = fsmn_out.transpose(1, 2)
        fsmn_memory = fsmn_out + v_masked
        fsmn_memory = self.dropout(fsmn_memory)
        if mask is not None:
            fsmn_memory = fsmn_memory * mask_expanded

        # Multi-head attention path
        q = q.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        attention_mask = _prepare_4d_attention_mask(mask, q.dtype) if mask is not None else None
        attn_output, _ = eager_attention_forward(
            self,
            q,
            k,
            v_heads,
            attention_mask,
            scaling=self.scaling,
            dropout=self.dropout.p if self.training else 0.0,
        )
        attn_output = attn_output.view(b, t, self.hidden_size)
        attn_output = self.linear_out(attn_output)

        return attn_output + fsmn_memory


class FunAsrNanoFeedForward(ParakeetEncoderFeedForward):
    """Positionwise feed-forward using the Parakeet encoder implementation."""

    def __init__(self, hidden_size: int, linear_units: int, dropout_rate: float = 0.1):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(hidden_size, linear_units)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(linear_units, hidden_size)
        self.activation_dropout = dropout_rate


class FunAsrNanoEncoderLayer(GradientCheckpointingLayer):
    """SANM encoder layer. State dict keys: norm1, norm2, self_attn.*, feed_forward.*"""

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_heads: int,
        linear_units: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.self_attn = FunAsrNanoSANMAttention(
            in_features=in_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            kernel_size=kernel_size,
            sanm_shift=sanm_shift,
        )

        self.feed_forward = FunAsrNanoFeedForward(hidden_size, linear_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, mask)

        if self.in_size == self.hidden_size:
            x = residual + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano audio encoder (SenseVoice SAN-M architecture), without any head on top.
    """
)
class FunAsrNanoEncoder(PreTrainedModel):
    config_class = FunAsrNanoEncoderConfig
    main_input_name = "input_features"

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)

        self.embed = FunAsrNanoSinusoidalPositionEncoder(config.input_size)

        self.encoders0 = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.input_size,
                    hidden_size=config.output_dim,
                    num_heads=config.num_attention_heads,
                    linear_units=config.intermediate_size,
                    dropout_rate=config.dropout,
                    attention_dropout_rate=config.attention_dropout,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
            ]
        )

        self.encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_dim,
                    hidden_size=config.output_dim,
                    num_heads=config.num_attention_heads,
                    linear_units=config.intermediate_size,
                    dropout_rate=config.dropout,
                    attention_dropout_rate=config.attention_dropout,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.encoder_layers - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_dim,
                    hidden_size=config.output_dim,
                    num_heads=config.num_attention_heads,
                    linear_units=config.intermediate_size,
                    dropout_rate=config.dropout,
                    attention_dropout_rate=config.attention_dropout,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.tp_blocks)
            ]
        )

        self.after_norm = nn.LayerNorm(config.output_dim)
        self.tp_norm = nn.LayerNorm(config.output_dim)

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FunAsrNanoSinusoidalPositionEncoder):
            expanded = SinusoidsPositionEmbedding(module.length, module.channels, module.max_timescale)
            init.copy_(
                module.positional_embedding,
                expanded.positional_embedding.to(
                    device=module.positional_embedding.device, dtype=module.positional_embedding.dtype
                ),
            )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutput | tuple:
        hidden_states = input_features.to(dtype=self.after_norm.weight.dtype)
        batch_size, max_len, _ = hidden_states.shape

        if feature_lengths is not None:
            mask = torch.arange(max_len, device=hidden_states.device)[None, :] < feature_lengths[:, None]
            mask = mask[:, None, :].to(dtype=hidden_states.dtype)
        else:
            mask = None

        hidden_states = hidden_states * (self.config.output_dim**0.5)
        hidden_states = self.embed(hidden_states)

        all_hidden_states = () if output_hidden_states else None

        for layer in self.encoders0:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        for layer in self.encoders:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.after_norm(hidden_states)

        for layer in self.tp_encoders:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.tp_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + ((all_hidden_states,) if output_hidden_states else ())

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class FunAsrNanoAdaptorAttention(FunAsrNanoAttention):
    """Adaptor attention with separate Q/K/V projections matching checkpoint keys."""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.size()

        q = self.linear_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        attention_mask = _prepare_4d_attention_mask(mask, q.dtype) if mask is not None else None
        out, _ = eager_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            scaling=self.scaling,
            dropout=self.dropout.p if self.training else 0.0,
        )
        out = out.view(b, t, self.hidden_size)
        return self.linear_out(out)


class FunAsrNanoAdaptorLayer(GradientCheckpointingLayer):
    """Adaptor transformer layer matching checkpoint structure."""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.self_attn = FunAsrNanoAdaptorAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = FunAsrNanoFeedForward(hidden_size, hidden_size // 4, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


class FunAsrNanoMultiModalProjector(VoxtralMultiModalProjector):
    """Audio adaptor projecting encoder output to the language-model dimension."""

    def __init__(self, config: FunAsrNanoConfig):
        nn.Module.__init__(self)
        self.config = config
        self.downsample_rate = config.adaptor_downsample_rate

        encoder_dim = config.audio_encoder_config.output_dim
        llm_dim = config.text_config.hidden_size

        self.linear_1 = nn.Linear(encoder_dim * config.adaptor_downsample_rate, config.adaptor_intermediate_size)
        self.act = nn.Sequential(nn.ReLU(), nn.Dropout(config.adaptor_dropout))
        self.linear_2 = nn.Linear(config.adaptor_intermediate_size, llm_dim)

        if config.adaptor_num_hidden_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    FunAsrNanoAdaptorLayer(
                        hidden_size=llm_dim,
                        num_heads=config.adaptor_num_attention_heads,
                        dropout_rate=config.adaptor_dropout,
                    )
                    for _ in range(config.adaptor_num_hidden_layers)
                ]
            )
        else:
            self.blocks = None

    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = encoder_out.size()
        k = self.downsample_rate

        chunk_num = (seq_len - 1) // k + 1
        pad_num = chunk_num * k - seq_len
        if pad_num > 0:
            encoder_out = F.pad(encoder_out, (0, 0, 0, pad_num, 0, 0), value=0.0)

        encoder_out = encoder_out.contiguous().view(batch_size, chunk_num, dim * k)
        x = self.linear_1(encoder_out)
        x = self.act(x)
        x = self.linear_2(x)

        output_lens = (encoder_out_lens - 1) // k + 1

        if self.blocks is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < output_lens[:, None]
            mask = mask[:, None, :].float()  # (batch, 1, time)

            for block in self.blocks:
                x = block(x, mask)

        return x, output_lens


@auto_docstring
class FunAsrNanoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    config: FunAsrNanoConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["FunAsrNanoEncoderLayer", "FunAsrNanoAdaptorLayer"]
    _skip_keys_device_placement = ["past_key_values"]


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
        input_features_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size, max_len, _ = input_features.shape
        if input_features_mask is None:
            feature_lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=input_features.device)
        else:
            feature_lengths = input_features_mask.sum(-1).to(torch.long)

        encoder_outputs = self.audio_tower(
            input_features=input_features,
            feature_lengths=feature_lengths,
            output_hidden_states=output_hidden_states,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds, audio_embed_lens = self.multi_modal_projector(encoder_out, feature_lengths)

        # Flatten audio embeddings over valid positions so they can directly replace placeholder tokens.
        valid_mask = (
            torch.arange(audio_embeds.shape[1], device=audio_embeds.device)[None, :] < audio_embed_lens[:, None]
        )
        pooler_output = audio_embeds[valid_mask]

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
    _keep_in_fp32_modules_strict = AttributeError()

    def get_audio_features(self, input_features, input_features_mask=None, **kwargs):
        return self.model.get_audio_features(input_features, input_features_mask, **kwargs)

    def forward(self, **super_kwargs):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

        >>> model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
    "FunAsrNanoProcessor",
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoModel",
    "FunAsrNanoForConditionalGeneration",
]
