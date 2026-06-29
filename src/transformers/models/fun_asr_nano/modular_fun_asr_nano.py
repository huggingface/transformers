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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...audio_utils import AudioInput, make_list_of_audio
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, can_return_tuple, is_torch_available, logging
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoEncoder`]. It is used to instantiate a
    Fun-ASR-Nano audio encoder (a SenseVoice SAN-M encoder) according to the specified arguments, defining the model
    architecture. Like [`ParakeetEncoderConfig`], this is a standalone encoder configuration since the encoder is a
    standalone model registered in the auto mappings.

    input_size (`int`, *optional*, defaults to 560):
        Input feature dimension (after LFR: 80 mel bins * 7 frames = 560).
    output_size (`int`, *optional*, defaults to 512):
        Hidden size of the encoder layers.
    attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in each SANM layer.
    linear_units (`int`, *optional*, defaults to 2048):
        Dimension of the feedforward layer.
    num_blocks (`int`, *optional*, defaults to 50):
        Number of main encoder blocks.
    tp_blocks (`int`, *optional*, defaults to 20):
        Number of timestamp prediction encoder blocks.
    dropout_rate (`float`, *optional*, defaults to 0.1):
        Dropout rate.
    attention_dropout_rate (`float`, *optional*, defaults to 0.0):
        Attention dropout rate.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the FSMN convolution.
    sanm_shift (`int`, *optional*, defaults to 0):
        Shift for asymmetric padding in FSMN.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for weight initialization.

    Example:

    ```python
    >>> from transformers import FunAsrNanoEncoderConfig, FunAsrNanoEncoder

    >>> configuration = FunAsrNanoEncoderConfig()
    >>> model = FunAsrNanoEncoder(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano_encoder"

    input_size: int = 560
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    tp_blocks: int = 20
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    kernel_size: int = 11
    sanm_shift: int = 0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoForConditionalGeneration`]. It is used
    to instantiate a Fun-ASR-Nano model according to the specified arguments, defining the model architecture.

    The adaptor (audio projector) is *not* a standalone model, so following the [`VoxtralConfig`] pattern its
    parameters live directly on this config rather than in a nested sub-config.

    audio_encoder_config (`dict` or `FunAsrNanoEncoderConfig`, *optional*):
        Configuration for the audio encoder.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the language model (Qwen3).
    audio_token_index (`int`, *optional*, defaults to 151646):
        Token ID used as placeholder for audio features.
    adaptor_downsample_rate (`int`, *optional*, defaults to 1):
        Downsampling factor applied to the encoder sequence before projecting to the language model.
    adaptor_ffn_dim (`int`, *optional*, defaults to 2048):
        Hidden size of the adaptor feed-forward projection.
    adaptor_num_layers (`int`, *optional*, defaults to 2):
        Number of adaptor transformer layers.
    adaptor_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the adaptor transformer layers.
    adaptor_dropout_rate (`float`, *optional*, defaults to 0.0):
        Dropout probability used in the adaptor.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for weight initialization.

    Example:

    ```python
    >>> from transformers import FunAsrNanoConfig, FunAsrNanoForConditionalGeneration

    >>> configuration = FunAsrNanoConfig()
    >>> model = FunAsrNanoForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano"
    attribute_map = {"audio_token_id": "audio_token_index"}
    sub_configs = {
        "text_config": AutoConfig,
        "audio_encoder_config": FunAsrNanoEncoderConfig,
    }

    audio_encoder_config: dict | FunAsrNanoEncoderConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646
    adaptor_downsample_rate: int = 1
    adaptor_ffn_dim: int = 2048
    adaptor_num_layers: int = 2
    adaptor_attention_heads: int = 8
    adaptor_dropout_rate: float = 0.0
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


class FunAsrNanoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
        "text_kwargs": {
            "padding": True,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class FunAsrNanoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|object_ref_start|>",
        audio_downsample_rate=1,
        default_transcription_prompt="Transcribe the audio:",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        audio_downsample_rate (`int`, *optional*, defaults to 1):
            Downsampling ratio applied by the audio adaptor, used to expand the audio placeholder token to the right
            number of audio tokens.
        default_transcription_prompt (`str`, *optional*, defaults to `"Transcribe the audio:"`):
            Default prompt to use for transcription tasks when applying transcription requests.
        """
        if tokenizer is not None and tokenizer.convert_tokens_to_ids(audio_token) is None:
            raise ValueError(f"Audio token {audio_token!r} is not present in the tokenizer vocabulary.")

        self.audio_token = audio_token
        self.audio_downsample_rate = audio_downsample_rate
        self.default_transcription_prompt = default_transcription_prompt
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        sampling_rate: int | None = None,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        r"""
        sampling_rate (`int`, *optional*):
            Sampling rate of the input audio. Must be 16000 for Fun-ASR-Nano.
        """
        if text is None:
            raise ValueError("You need to specify `text` input to process.")

        output_kwargs = self._merge_kwargs(
            FunAsrNanoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.pop("return_tensors", None)

        text = list(text) if isinstance(text, list) else [text]

        audio_features = None
        if audio is not None:
            audio = make_list_of_audio(audio)
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or audio_kwargs.get("sampling_rate"),
                return_tensors=return_tensors,
            )
            input_features_mask = audio_features.pop("attention_mask", None)
            if input_features_mask is not None:
                audio_features["input_features_mask"] = input_features_mask

            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}."
                )

            # Expand each audio placeholder into as many tokens as the (downsampled) audio feature length.
            audio_lengths = audio_features["feature_lengths"].tolist()
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_tokens = (audio_length - 1) // self.audio_downsample_rate + 1
                    replace_str.append(self.audio_token * int(num_tokens))
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **text_kwargs)

        if audio_features is not None:
            return BatchFeature(data={**text_inputs, **audio_features})

        return BatchFeature(data=dict(text_inputs))

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses the processor's default transcription prompt.
            **kwargs:
                Additional keyword arguments forwarded to [`~FunAsrNanoProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`FunAsrNanoForConditionalGeneration.generate`].
        """

        if isinstance(audio, str):
            audio_items = [audio]
        elif isinstance(audio, (list, tuple)) and all(isinstance(item, str) for item in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                import torch as torch_module

                audio_items = [
                    item.detach().cpu().numpy() if isinstance(item, torch_module.Tensor) else item
                    for item in audio_items
                ]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = []
        for prompt_text, audio_item in zip(prompts, audio_items):
            content = [{"type": "text", "text": prompt_text}]
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})
            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    # `decode` and `batch_decode` are inherited from `ProcessorMixin` and forward to the tokenizer; the base `decode`
    # already handles batches, so no custom override is needed here.

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(feature_extractor_input_names + tokenizer_input_names + ["input_features_mask"]))


@auto_docstring(
    custom_intro="""
    Base class for Fun-ASR-Nano outputs, with hidden states and attentions.
    """
)
@dataclass
class FunAsrNanoModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        Projected audio embeddings produced by the audio encoder and adaptor.
    """

    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Fun-ASR-Nano causal language model (or autoregressive) outputs.
    """
)
@dataclass
class FunAsrNanoCausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head.
    past_key_values (`tuple`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Cached key/value states that can be used to speed up sequential decoding.
    hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Hidden states of the language model at the output of each layer.
    attentions (`tuple[torch.FloatTensor]`, *optional*):
        Attention weights of the language model.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        Projected audio embeddings produced by the audio encoder and adaptor.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: tuple | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    audio_hidden_states: torch.FloatTensor | None = None


class FunAsrNanoSinusoidalPositionEncoder(nn.Module):
    """Sinusoidal positional encoding generated on the fly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device, dtype=x.dtype).unsqueeze(0)

        log_timescale_increment = math.log(10000.0) / (input_dim / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(0, input_dim // 2, device=x.device, dtype=x.dtype) * (-log_timescale_increment)
        )
        scaled_time = positions.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)

        return x + encoding


class FunAsrNanoSANMAttention(nn.Module):
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
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size

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

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k**-0.5)

        if mask is not None:
            mask_for_attn = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask_for_attn, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(mask_for_attn, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v_heads)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        attn_output = self.linear_out(attn_output)

        return attn_output + fsmn_memory


class FunAsrNanoFeedForward(nn.Module):
    """Positionwise feedforward with keys: feed_forward.w_1, feed_forward.w_2."""

    def __init__(self, hidden_size: int, linear_units: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, linear_units)
        self.w_2 = nn.Linear(linear_units, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class FunAsrNanoLayerNorm(nn.LayerNorm):
    """LayerNorm that casts to float32 for numerical stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class FunAsrNanoEncoderLayer(nn.Module):
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

        self.norm1 = FunAsrNanoLayerNorm(in_size)
        self.norm2 = FunAsrNanoLayerNorm(hidden_size)

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

        self.embed = FunAsrNanoSinusoidalPositionEncoder()

        self.encoders0 = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.input_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
            ]
        )

        self.encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.tp_blocks)
            ]
        )

        self.after_norm = FunAsrNanoLayerNorm(config.output_size)
        self.tp_norm = FunAsrNanoLayerNorm(config.output_size)

        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutput | tuple:
        hidden_states = input_features.to(dtype=next(self.parameters()).dtype)
        batch_size, max_len, _ = hidden_states.shape

        if feature_lengths is not None:
            mask = torch.arange(max_len, device=hidden_states.device)[None, :] < feature_lengths[:, None]
            mask = mask[:, None, :].to(dtype=hidden_states.dtype)
        else:
            mask = None

        hidden_states = hidden_states * (self.config.output_size**0.5)
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


class FunAsrNanoAdaptorAttention(nn.Module):
    """Adaptor attention with separate Q/K/V projections matching checkpoint keys."""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

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

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)

        if mask is not None:
            # mask shape: (batch, 1, time)
            mask_for_attn = (~mask.bool()).unsqueeze(1)  # (batch, 1, 1, time)
            scores = scores.masked_fill(mask_for_attn, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        return self.linear_out(out)


class FunAsrNanoAdaptorLayer(nn.Module):
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


class FunAsrNanoAdaptor(nn.Module):
    """Audio adaptor projecting encoder output to the language-model dimension."""

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__()
        self.config = config
        self.downsample_rate = config.adaptor_downsample_rate

        encoder_dim = config.audio_encoder_config.output_size
        llm_dim = config.text_config.hidden_size

        self.linear1 = nn.Linear(encoder_dim * config.adaptor_downsample_rate, config.adaptor_ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.adaptor_ffn_dim, llm_dim)

        if config.adaptor_num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    FunAsrNanoAdaptorLayer(
                        hidden_size=llm_dim,
                        num_heads=config.adaptor_attention_heads,
                        dropout_rate=config.adaptor_dropout_rate,
                    )
                    for _ in range(config.adaptor_num_layers)
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
        x = self.linear1(encoder_out)
        x = self.relu(x)
        x = self.linear2(x)

        output_lens = (encoder_out_lens - 1) // k + 1

        if self.blocks is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < output_lens[:, None]
            mask = mask[:, None, :].float()  # (batch, 1, time)

            for block in self.blocks:
                x = block(x, mask)

        return x, output_lens


@auto_docstring
class FunAsrNanoPreTrainedModel(PreTrainedModel):
    config: FunAsrNanoConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["FunAsrNanoEncoderLayer", "FunAsrNanoAdaptorLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model (SenseVoice SAN-M audio encoder, a Transformer adaptor and a Qwen3 language model),
    without a language modeling head.
    """
)
class FunAsrNanoModel(FunAsrNanoPreTrainedModel):
    def __init__(self, config: FunAsrNanoConfig):
        super().__init__(config)

        self.audio_encoder = FunAsrNanoEncoder(config.audio_encoder_config)
        self.audio_adaptor = FunAsrNanoAdaptor(config)
        self.language_model = AutoModel.from_config(config.text_config)

        self.audio_token_index = config.audio_token_index

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features, meaning inferring the audio encoder and the adaptor."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Audio features `(batch, time, feature_dim)` produced by the feature extractor (after LFR stacking).
        feature_lengths (`torch.LongTensor`, *optional*):
            Length of each audio feature sequence. When not provided, every sequence is assumed to be full length.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPooling`]: `last_hidden_state` holds the audio encoder output,
            `pooler_output` holds the projected audio embeddings (flattened over valid positions), and
            `hidden_states` holds the per-layer encoder states.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size, max_len, _ = input_features.shape
        if feature_lengths is None:
            feature_lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=input_features.device)

        encoder_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lengths=feature_lengths,
            output_hidden_states=output_hidden_states,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds, audio_embed_lens = self.audio_adaptor(encoder_out, feature_lengths)

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

    def _get_audio_embeds(
        self, input_features: torch.FloatTensor, feature_lengths: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded audio embeddings `(batch, time, dim)` and their per-sample lengths (used by `forward`)."""
        encoder_out = self.audio_encoder(
            input_features=input_features, feature_lengths=feature_lengths
        ).last_hidden_state
        return self.audio_adaptor(encoder_out, feature_lengths)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_lengths: torch.LongTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> FunAsrNanoModelOutputWithPast | tuple:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, time, feature_dim)`, *optional*):
            Audio features after LFR stacking.
        feature_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Length of each audio feature sequence.
        input_features_mask (`torch.Tensor` of shape `(batch_size, time)`, *optional*):
            Padding mask for the audio feature sequence.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        audio_embeds = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            special_audio_mask = input_ids == self.audio_token_index
            if (
                input_features is not None
                and input_ids is not None
                and input_ids.shape[1] != 1
                and special_audio_mask.any()
            ):
                if feature_lengths is None and input_features_mask is not None:
                    feature_lengths = input_features_mask.sum(-1).to(torch.long)
                audio_embeds, audio_embed_lens = self._get_audio_embeds(input_features, feature_lengths)

                # Mask and scatter audio embeddings into token positions
                special_audio_mask_expanded = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

                max_audio_len = audio_embeds.shape[1]
                audio_len_mask = (
                    torch.arange(max_audio_len, device=audio_embeds.device)[None, :] < audio_embed_lens[:, None]
                )
                flat_audio = audio_embeds[audio_len_mask]
                if special_audio_mask.sum() != flat_audio.shape[0]:
                    raise ValueError(
                        f"Number of audio tokens ({special_audio_mask.sum().item()}) does not match "
                        f"number of audio features ({flat_audio.shape[0]})."
                    )

                inputs_embeds = inputs_embeds.masked_scatter(
                    special_audio_mask_expanded.to(inputs_embeds.device),
                    flat_audio.to(inputs_embeds.device, inputs_embeds.dtype),
                )

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        return FunAsrNanoModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model for speech recognition: a SenseVoice SAN-M audio encoder, a Transformer adaptor and a
    Qwen3 language model with a language modeling head.
    """
)
class FunAsrNanoForConditionalGeneration(FunAsrNanoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__(config)

        self.model = FunAsrNanoModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(self, *args, **kwargs):
        return self.model.get_audio_features(*args, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_lengths: torch.LongTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> FunAsrNanoCausalLMOutput | tuple:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, time, feature_dim)`, *optional*):
            Audio features after LFR stacking.
        feature_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Length of each audio feature sequence.
        input_features_mask (`torch.Tensor` of shape `(batch_size, time)`, *optional*):
            Padding mask for the audio feature sequence.

        Example:

        ```python
        >>> from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

        >>> model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            feature_lengths=feature_lengths,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return FunAsrNanoCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration: bool = False, **kwargs):
        input_features = kwargs.pop("input_features", None)
        feature_lengths = kwargs.pop("feature_lengths", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["input_features"] = input_features
            model_inputs["feature_lengths"] = feature_lengths
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
    "FunAsrNanoProcessor",
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoModel",
    "FunAsrNanoForConditionalGeneration",
]
