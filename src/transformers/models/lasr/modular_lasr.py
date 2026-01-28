# Copyright 2025 The HuggingFace Inc. team and Google LLC. All rights reserved.
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

import itertools
from collections.abc import Callable

import torch
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from ..parakeet.configuration_parakeet import ParakeetCTCConfig
from ..parakeet.modeling_parakeet import (
    ParakeetEncoderBlock,
    ParakeetEncoderConvolutionModule,
    ParakeetForCTC,
    ParakeetPreTrainedModel,
)
from ..parakeet.processing_parakeet import ParakeetProcessor
from ..t5.tokenization_t5 import T5Tokenizer


class LasrTokenizer(T5Tokenizer, TokenizersBackend):
    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        vocab=None,
        vocab_file=None,
        **kwargs,
    ):
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            vocab=vocab,
            vocab_file=vocab_file,
            **kwargs,
        )
        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=3,
                byte_fallback=False,
            )
        )

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        group_tokens: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if group_tokens:
            token_ids = [token_group[0] for token_group in itertools.groupby(token_ids)]

        # for CTC we filter out the blank token, which is the pad token
        token_ids = [token for token in token_ids if token != self.pad_token_id]

        return TokenizersBackend._decode(
            self,
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


class LasrProcessor(ParakeetProcessor):
    pass


# Cannot inhert because the mixin will not carry over otherwise
class LasrEncoderConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`LasrEncoder`]. It is used to instantiate a
    `LasrEncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            hidden_size (`int`, *optional*, defaults to 512):
                Dimension of the layers and the hidden states.
            num_hidden_layers (`int`, *optional*, defaults to 17):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (`int`, *optional*, defaults to 2048):
                Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the encoder and pooler.
            attention_bias (`bool`, *optional*, defaults to `False`):
                Whether to use bias in the attention layers.
            convolution_bias (`bool`, *optional*, defaults to `False`):
                Whether to use bias in convolutions of the conformer's convolution module.
            conv_kernel_size (`int`, *optional*, defaults to 32):
                The kernel size of the convolution layers in the Conformer block.
            subsampling_conv_channels (`int`, *optional*, defaults to 256):
                The number of channels in the subsampling convolution layers.
            subsampling_conv_kernel_size (`int`, *optional*, defaults to 5):
                The kernel size of the subsampling convolution layers.
            subsampling_conv_stride (`int`, *optional*, defaults to 2):
                The stride of the subsampling convolution layers.
            num_mel_bins (`int`, *optional*, defaults to 128):
                Number of mel features.
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout ratio for all fully connected layers in the embeddings, encoder, and pooler.
            dropout_positions (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the positions in the input sequence.
            layerdrop (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the layers in the encoder.
            activation_dropout (`float`, *optional*, defaults to 0.1):
                The dropout ratio for activations inside the fully connected layer.
            attention_dropout (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention layers.
            max_position_embeddings (`int`, *optional*, defaults to 10000):
                The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (`float`, *optional*, defaults to 1e-06):
                The epsilon used by the layer normalization layers.
            feed_forward_residual_weights (`tuple[float, float]`, *optional*, defaults to `[1.5, 0.5]`):
                The residual weights for the feed forward layers.
            conv_residual_weights (`tuple[float, float]`, *optional*, defaults to `[2.0, 1.0]`):
                The residual weights for the convolution layers.
            batch_norm_momentum (`float`, *optional*, defaults to 0.01):
                The momentum for the batch normalization layers.
            rope_parameters (`RopeParameters`, *optional*):
                Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
                a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
                with longer `max_position_embeddings`.

    Example:
        ```python
        >>> from transformers import LasrEncoderModel, LasrEncoderConfig

        >>> # Initializing a `LasrEncoder` configuration
        >>> configuration = LasrEncoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = LasrEncoderModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the LasrEncoder architecture from Google Health AI. You can find more details
    and pre-trained models at [TODO/TODO](https://huggingface.co/TODO/TODO).
    """

    model_type = "lasr_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size=512,
        num_hidden_layers=17,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="silu",
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=32,
        subsampling_conv_channels=256,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=2,
        num_mel_bins=128,
        dropout=0.1,
        dropout_positions=0.0,
        layerdrop=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=10000,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        feed_forward_residual_weights=[1.5, 0.5],
        conv_residual_weights=[2.0, 1.0],
        batch_norm_momentum=0.01,
        rope_parameters=None,
        **kwargs,
    ):
        self.rope_parameters = rope_parameters
        self.layer_norm_eps = layer_norm_eps
        self.feed_forward_residual_weights = feed_forward_residual_weights
        self.conv_residual_weights = conv_residual_weights
        self.batch_norm_momentum = batch_norm_momentum
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads  # LlamaAttention compatibility
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.convolution_bias = convolution_bias

        self.conv_kernel_size = conv_kernel_size
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins

        self.dropout = dropout
        self.dropout_positions = dropout_positions
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

        super().__init__(
            **kwargs,
        )


class LasrCTCConfig(ParakeetCTCConfig):
    r"""
    This is the configuration class to store the configuration of a [`LasrForCTC`]. It is used to instantiate a
    Lasr CTC model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
            vocab_size (`int`, *optional*, defaults to 512):
                Vocabulary size of the model.
            ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
                Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
                instance of [`LasrForCTC`].
            ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
                Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
                occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
                of [`LasrForCTC`].
            encoder_config (`Union[dict, LasrEncoderConfig]`, *optional*):
                The config object or dictionary of the encoder.
            pad_token_id (`int`, *optional*, defaults to 0):
                Padding token id. Also used as blank token id.
    Example:
        ```python
        >>> from transformers import LasrForCTC, LasrCTCConfig
        >>> # Initializing a Lasr configuration
        >>> configuration = LasrCTCConfig()
        >>> # Initializing a model from the configuration
        >>> model = LasrForCTC(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    This configuration class is based on the Lasr CTC architecture from Google Health AI. You can find more details
    and pre-trained models at [TODO/TODO](https://huggingface.co/TODO/TODO).
    """

    def __init__(
        self,
        vocab_size=512,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        encoder_config: dict | LasrEncoderConfig = None,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            ctc_loss_reduction=ctc_loss_reduction,
            ctc_zero_infinity=ctc_zero_infinity,
            encoder_config=encoder_config,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @property
    def inputs_to_logits_ratio(self):
        return self.encoder_config.subsampling_conv_stride**2


class LasrEncoderSubsampling(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.num_mel_bins, config.hidden_size)
        self.conv_0 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.subsampling_conv_channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.dense_1 = nn.Linear(config.subsampling_conv_channels, config.hidden_size)
        self.act_fn = nn.ReLU()

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.dense_0(input_features))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.act_fn(self.conv_0(hidden_states))
        hidden_states = self.act_fn(self.conv_1(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        return self.dense_1(hidden_states)


class LasrEncoderRotaryEmbedding(LlamaRotaryEmbedding): ...


class LasrEncoderAttention(LlamaAttention):
    def __init__(self, config: LasrEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
        return attn_output, attn_weights


class LasrEncoderConvolutionModule(ParakeetEncoderConvolutionModule):
    def __init__(self, config: LasrEncoderConfig, module_config=None):
        super().__init__(config, module_config)
        self.padding = "same"
        self.norm = nn.BatchNorm1d(config.hidden_size, momentum=config.batch_norm_momentum)


class LasrEncoderBlock(ParakeetEncoderBlock):
    def __init__(self, config: LasrEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.feed_forward_residual_weights = config.feed_forward_residual_weights
        self.conv_residual_weights = config.conv_residual_weights

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias=False)
        self.norm_self_att = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias=False)
        self.norm_conv = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias=False)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias=False)
        self.norm_out = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual + self.feed_forward_residual_weights[1] * hidden_states
        )

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
        hidden_states = self.conv_residual_weights[0] * hidden_states + self.conv_residual_weights[1] * conv_output

        residual = hidden_states
        hidden_states = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual + self.feed_forward_residual_weights[1] * hidden_states
        )

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


class LasrPreTrainedModel(ParakeetPreTrainedModel):
    # padding is incompatible with flex attention as the resulting mask cannot be used to apply padding
    _supports_flex_attn = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = self.config.encoder_config if isinstance(self.config, LasrCTCConfig) else self.config
        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride

        num_layers = 2
        for _ in range(num_layers):
            input_lengths = (input_lengths - kernel_size) // stride + 1

        return input_lengths


@auto_docstring(
    custom_intro="""
    The LasrEncoder model, based on the Conformer architecture](https://arxiv.org/abs/2005.08100).
    """
)
class LasrEncoder(LasrPreTrainedModel):
    config: LasrEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: LasrEncoderConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop

        self.subsampler = LasrEncoderSubsampling(config)
        self.rotary_emb = LasrEncoderRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [LasrEncoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.out_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)

        self.post_init()

    @auto_docstring
    @check_model_inputs()
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, LasrEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = TODO
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = ParakeetEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """

        hidden_states = self.subsampler(input_features)
        cos, sin = self.rotary_emb(
            hidden_states, torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        cos = nn.functional.dropout(cos, p=self.dropout_positions, training=self.training)
        sin = nn.functional.dropout(sin, p=self.dropout_positions, training=self.training)

        if attention_mask is not None:
            attention_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=(cos, sin),
                    **kwargs,
                )

        hidden_states = self.out_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class LasrForCTC(ParakeetForCTC):
    def generate(**super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, LasrForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = TODO
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = LasrForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        return super().generate(**super_kwargs)


__all__ = [
    "LasrForCTC",
    "LasrEncoder",
    "LasrPreTrainedModel",
    "LasrProcessor",
    "LasrEncoderConfig",
    "LasrCTCConfig",
    "LasrTokenizer",
]
