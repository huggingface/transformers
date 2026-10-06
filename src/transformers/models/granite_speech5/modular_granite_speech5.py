# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..llama.modeling_llama import eager_attention_forward
from ..moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoderAttention
from ..parakeet.configuration_parakeet import ParakeetCTCConfig
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderBlock,
    ParakeetEncoderModelOutput,
    ParakeetForCTC,
    ParakeetPreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-granite/granite-speech-5.0-470m-turboctc")
@strict
class GraniteSpeech5EncoderConfig(PreTrainedConfig):
    r"""
    max_position_embeddings (`int`, *optional*, defaults to 512):
        Maximum relative position index of Shaw's relative positional encoding; the embedding table holds
        `2 * max_position_embeddings + 1` entries.
    context_size (`int`, *optional*, defaults to 128):
        Context size for block-wise conformer attention.
    conv_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size of the depthwise convolution in the conformer convolution module.
    conv_expansion_factor (`int`, *optional*, defaults to 2):
        Expansion factor for the conformer convolution module.
    subsample_layers (`list[int]`, *optional*, defaults to `[0, 1]`):
        Indices of the conformer blocks that subsample time by 2 (stride-2 depthwise convolution with a
        mean-pooled residual).

    Example:

    ```python
    >>> from transformers import GraniteSpeech5EncoderConfig, GraniteSpeech5Encoder

    >>> # Initializing a GraniteSpeech5EncoderConfig
    >>> configuration = GraniteSpeech5EncoderConfig()

    >>> # Initializing a GraniteSpeech5Encoder (with random weights)
    >>> model = GraniteSpeech5Encoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech5_encoder"

    vocab_size: int = 16384
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    num_mel_bins: int = 80
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 512
    context_size: int = 128
    conv_kernel_size: int = 7
    conv_expansion_factor: int = 2
    subsample_layers: list[int] | None = None
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.subsample_layers is None:
            self.subsample_layers = [0, 1]
        if self.context_size <= 0 or self.context_size > self.max_position_embeddings:
            raise ValueError(
                f"`context_size` must be in (0, max_position_embeddings={self.max_position_embeddings}], "
                f"got {self.context_size}."
            )


@auto_docstring(checkpoint="ibm-granite/granite-speech-5.0-470m-turboctc")
@strict
class GraniteSpeech5CTCConfig(ParakeetCTCConfig):
    r"""
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`GraniteSpeech5ForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`GraniteSpeech5ForCTC`].
    encoder_config (`Union[dict, GraniteSpeech5EncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.

    Example:

    ```python
    >>> from transformers import GraniteSpeech5ForCTC, GraniteSpeech5CTCConfig
    >>> # Initializing a GraniteSpeech5 configuration
    >>> configuration = GraniteSpeech5CTCConfig()
    >>> # Initializing a model from the configuration
    >>> model = GraniteSpeech5ForCTC(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    vocab_size: int = 16384
    pad_token_id: int | None = 0
    # controls the tying of `ctc_head` to the encoder's (self-conditioning) CTC head `out`
    tie_word_embeddings: bool = True

    def validate_architecture(self):
        if self.encoder_config.vocab_size != self.vocab_size:
            raise ValueError(
                f"The encoder config vocabulary size ({self.encoder_config.vocab_size}) does not match the CTC "
                f"config vocabulary size ({self.vocab_size})."
            )


class GraniteSpeech5EncoderModelOutput(ParakeetEncoderModelOutput): ...


class GraniteSpeech5EncoderAttention(MoonshineStreamingEncoderAttention):
    """Block-wise self-attention with Shaw's relative positional embeddings."""

    def __init__(self, config: GraniteSpeech5EncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.context_size = config.context_size
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)
        self.rel_pos_emb = nn.Embedding(2 * config.max_position_embeddings + 1, config.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        batch_size, seq_length, _ = hidden_states.shape

        # right-pad the sequence to a whole number of blocks
        num_padded = -seq_length % self.context_size
        if num_padded > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, num_padded))
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=hidden_states.device)
            attention_mask = nn.functional.pad(attention_mask, (0, num_padded), value=False)

        # every block is folded into the batch dimension, so all blocks attend independently in one call
        flattened_batch_size = batch_size * (hidden_states.shape[1] // self.context_size)
        hidden_shape = (flattened_batch_size, self.context_size, self.config.num_attention_heads, -1)
        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

        relative_position_embeddings = self.rel_pos_emb(position_embeddings) * self.scaling
        queries = query_states.permute(2, 0, 1, 3).reshape(self.context_size, -1, self.head_dim)
        position_bias = queries @ relative_position_embeddings.transpose(1, 2)
        position_bias = position_bias.view(
            self.context_size, flattened_batch_size, self.config.num_attention_heads, self.context_size
        )
        position_bias = position_bias.permute(1, 2, 0, 3)
        position_bias = position_bias.contiguous()  # the fused attention kernels want a contiguous bias
        if attention_mask is not None:
            # mask padded key columns so no query attends to a padded frame
            key_mask = attention_mask.reshape(flattened_batch_size, self.context_size)
            position_bias = position_bias.masked_fill(
                ~key_mask[:, None, None, :], torch.finfo(position_bias.dtype).min
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=position_bias,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, -1, self.config.num_attention_heads * self.head_dim)

        return self.o_proj(attn_output[:, :seq_length]), None


class GraniteSpeech5EncoderConvolutionModule(nn.Module):
    def __init__(self, config: GraniteSpeech5EncoderConfig, stride: int = 1):
        super().__init__()
        inner_dim = config.hidden_size * config.conv_expansion_factor

        self.pointwise_lin1 = nn.Linear(config.hidden_size, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            config.conv_kernel_size,  # kernel_size should be an odd number for 'SAME' padding
            stride=stride,
            padding=(config.conv_kernel_size - 1) // 2,
            groups=inner_dim,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(inner_dim)
        self.pointwise_lin2 = nn.Linear(inner_dim, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.pointwise_lin1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=-1)

        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        hidden_states = self.depthwise_conv(hidden_states.transpose(1, 2))
        hidden_states = nn.functional.silu(self.norm(hidden_states))
        hidden_states = self.pointwise_lin2(hidden_states.transpose(1, 2))
        return hidden_states


def downsample_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Downsample a `(batch_size, seq_length)` padding mask by 2: a half-rate frame is valid iff both source
    frames are valid (a trailing odd frame is dropped, mirroring the residual pooling)."""
    half_length = attention_mask.shape[1] // 2
    return attention_mask[:, : 2 * half_length].reshape(attention_mask.shape[0], half_length, 2).all(dim=2)


class GraniteSpeech5EncoderBlock(GradientCheckpointingLayer, ParakeetEncoderBlock): ...


class GraniteSpeech5EncoderSubsamplingBlock(GraniteSpeech5EncoderBlock):
    """Conformer block that subsamples time by 2"""

    def __init__(self, config: GraniteSpeech5EncoderConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.conv = GraniteSpeech5EncoderConvolutionModule(config, stride=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states  # the conformer architecture uses a factor of 0.5

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
        pooled = hidden_states.unfold(1, 2, 2).mean(-1)  # pool by 2 along time dim, eventually dropping last frame
        hidden_states = pooled + conv_output[:, : pooled.shape[1]]

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


@auto_docstring
class GraniteSpeech5PreTrainedModel(ParakeetPreTrainedModel):
    config: GraniteSpeech5CTCConfig
    _no_split_modules = ["GraniteSpeech5EncoderBlock", "GraniteSpeech5EncoderSubsamplingBlock"]

    # float attention bias is not supported by flash attention
    _supports_flash_attn = False

    _keep_in_fp32_modules_strict = ["conv.norm"]

    _can_record_outputs = {
        "hidden_states": GraniteSpeech5EncoderBlock,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteSpeech5Encoder):
            init.copy_(module.attention_dists, module.compute_attention_dists())

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)
        output_lengths = input_lengths
        for _ in encoder_config.subsample_layers:
            output_lengths = output_lengths // 2
        return output_lengths


@auto_docstring(
    custom_intro="""
    The Granite Speech 5.0 conformer encoder, adapted from the [Granite Speech CTC encoder](https://huggingface.co/papers/2505.08699)
    with block-wise time subsampling and self-conditioned CTC from the middle layer.
    """
)
class GraniteSpeech5Encoder(ParakeetEncoder):
    config: GraniteSpeech5EncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: GraniteSpeech5EncoderConfig):
        GraniteSpeech5PreTrainedModel.__init__(self, config)
        self.gradient_checkpointing = False

        self.attention_dists = nn.Buffer(self.compute_attention_dists(), persistent=False)
        # see [`feature_extraction_granite_speech5.GraniteSpeech5FeatureExtractor`]
        # mel frames are delta-expanded (×2), then stacked in pairs (×2)
        self.input_linear = nn.Linear(config.num_mel_bins * 4, config.hidden_size, bias=True)
        self.layers = nn.ModuleList(
            [
                # CODEPATH: subsampling blocks
                GraniteSpeech5EncoderSubsamplingBlock(config, layer_idx)
                if layer_idx in config.subsample_layers
                else GraniteSpeech5EncoderBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.out = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.out_mid = nn.Linear(config.vocab_size, config.hidden_size, bias=True)

        self.post_init()

    def compute_attention_dists(self) -> torch.Tensor:
        """Clamped relative positional distances used by Shaw's relative positional embeddings."""
        context_size = self.config.context_size
        seq = torch.arange(context_size)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        return torch.clamp(relpos_dist, -context_size, context_size) + self.config.max_position_embeddings

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> GraniteSpeech5EncoderModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.

        Example:

        ```python
        >>> from transformers import AutoProcessor, GraniteSpeech5Encoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "ibm-granite/granite-speech-5.0-470m-turboctc"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = GraniteSpeech5Encoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """
        hidden_states = self.input_linear(input_features.to(self.input_linear.weight.dtype))
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=self.attention_dists,
                **kwargs,
            )

            # CODEPATH: the padding mask is halved after each subsampling block
            if layer_idx in self.config.subsample_layers and attention_mask is not None:
                attention_mask = downsample_attention_mask(attention_mask)

            # CODEPATH: self-conditioned CTC: feed the mid-layer CTC posteriors back into the hidden statess
            if layer_idx + 1 == self.config.num_hidden_layers // 2:
                mid_logits = self.out(hidden_states)
                mid_injection = self.out_mid(nn.functional.softmax(mid_logits, dim=-1))
                hidden_states = hidden_states + mid_injection.to(hidden_states.device)

        return GraniteSpeech5EncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask.int() if attention_mask is not None and output_attention_mask else None,
        )


@auto_docstring(
    custom_intro="""
    Granite Speech 5.0 encoder with a Connectionist Temporal Classification (CTC) head.
    """
)
class GraniteSpeech5ForCTC(ParakeetForCTC):
    config: GraniteSpeech5CTCConfig
    # the CTC head is the same projection as the encoder's mid-layer self-conditioning head
    _tied_weights_keys = {
        "ctc_head.weight": "encoder.out.weight",
        "ctc_head.bias": "encoder.out.bias",
    }

    def __init__(self, config: GraniteSpeech5CTCConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.ctc_head = nn.Linear(config.encoder_config.hidden_size, config.vocab_size, bias=True)

    def generate(**super_kwargs):
        r"""
        compile_config ([`~generation.CompileConfig`], *optional*):
            If provided, `torch.compile` will be applied to the forward calls in the decoding loop.

        Example:

        ```python
        >>> from transformers import AutoProcessor, GraniteSpeech5ForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "ibm-granite/granite-speech-5.0-470m-turboctc"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = GraniteSpeech5ForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        return super().generate(**super_kwargs)


__all__ = [
    "GraniteSpeech5CTCConfig",
    "GraniteSpeech5EncoderConfig",
    "GraniteSpeech5ForCTC",
    "GraniteSpeech5Encoder",
    "GraniteSpeech5PreTrainedModel",
]
