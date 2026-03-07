# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Parakeet model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from .configuration_parakeet import ParakeetCTCConfig, ParakeetEncoderConfig, ParakeetTDTConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Extends [~modeling_outputs.BaseModelOutput] to include the output attention mask since sequence length is not preserved in the model's forward.
    """
)
class ParakeetEncoderModelOutput(BaseModelOutput):
    attention_mask: torch.Tensor | None = None


class ParakeetEncoderRelPositionalEncoding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: ParakeetEncoderConfig, device=None):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        base = 10000.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, config.hidden_size, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / config.hidden_size
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor):
        seq_length = hidden_states.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {seq_length} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        position_ids = torch.arange(seq_length - 1, -seq_length, -1, device=hidden_states.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(hidden_states.shape[0], -1, 1).to(hidden_states.device)
        )
        position_ids_expanded = position_ids[None, None, :].float()

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            sin = freqs.sin()
            cos = freqs.cos()
            # interleave sin and cos
            pos_embed = torch.stack([sin, cos], dim=-1)
            pos_embed = pos_embed.reshape(*pos_embed.shape[:-2], -1)

        return pos_embed.to(dtype=hidden_states.dtype)


class ParakeetEncoderFeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states):
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ParakeetEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: ParakeetEncoderConfig, module_config=None):
        super().__init__(config, module_config)


class ParakeetEncoderAttention(LlamaAttention):
    """Multi-head attention with relative positional encoding. See section 3.3 of https://huggingface.co/papers/1901.02860."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False
        # W_{k,R} projection
        self.relative_k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # global content bias
        self.bias_u = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))
        # global positional bias
        self.bias_v = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        query_states_with_bias_u = query_states + self.bias_u.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )
        query_states_with_bias_v = query_states + self.bias_v.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(batch_size, -1, self.config.num_attention_heads, self.head_dim)

        # terms (b) and (d)
        matrix_bd = query_states_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[..., :seq_length]
        matrix_bd = matrix_bd * self.scaling

        if attention_mask is not None:
            # here the original codebase uses -10000.0 rather than float("-inf") and then manual masked fill with 0.0s
            # see: https://github.com/NVIDIA-NeMo/NeMo/blob/8cfedd7203462cb251a914e700e5605444277561/nemo/collections/asr/parts/submodules/multi_head_attention.py#L320-L340
            # we rather went for a straight-forward approach with float("-inf")
            matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))

        # will compute matrix_ac - terms (a) and (c) - and add matrix_bd
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states_with_bias_u,
            key=key_states,
            value=value_states,
            attention_mask=matrix_bd,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _rel_shift(self, attention_scores):
        """Relative position shift for Shaw et al. style attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = nn.functional.pad(attention_scores, pad=(1, 0))
        attention_scores = attention_scores.view(batch_size, num_heads, -1, query_length)
        attention_scores = attention_scores[:, :, 1:].view(batch_size, num_heads, query_length, position_length)
        return attention_scores


class ParakeetEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()

        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))

        # define layers
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        )
        self.layers.append(nn.ReLU())
        for i in range(self.num_layers - 1):
            # depthwise conv
            self.layers.append(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.channels,
                )
            )
            # pointwise conv
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            # activation
            self.layers.append(nn.ReLU())

        out_length = config.num_mel_bins // (self.stride**self.num_layers)
        self.linear = nn.Linear(config.subsampling_conv_channels * out_length, config.hidden_size, bias=True)

    def _get_output_length(self, input_lengths: torch.Tensor, conv_layer: nn.Conv2d):
        if hasattr(conv_layer, "stride") and conv_layer.stride != (1, 1):
            padding = conv_layer.padding
            kernel_size = conv_layer.kernel_size[0]
            stride = conv_layer.stride[0]

            output_lengths = (input_lengths + padding[0] + padding[1] - kernel_size) // stride + 1
            return output_lengths

        return input_lengths

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden_states = input_features.unsqueeze(1)
        current_lengths = attention_mask.sum(-1) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states)

            # mask the hidden states
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                current_lengths = self._get_output_length(current_lengths, layer)
                current_seq_length = hidden_states.shape[2]
                channel_mask = (
                    torch.arange(current_seq_length, device=attention_mask.device) < current_lengths[:, None]
                )
                hidden_states *= channel_mask[:, None, :, None]

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = self.linear(hidden_states)

        return hidden_states


class ParakeetEncoderBlock(GradientCheckpointingLayer):
    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int | None = None):
        super().__init__()
        self.gradient_checkpointing = False

        self.feed_forward1 = ParakeetEncoderFeedForward(config)
        self.self_attn = ParakeetEncoderAttention(config, layer_idx)
        self.conv = ParakeetEncoderConvolutionModule(config)
        self.feed_forward2 = ParakeetEncoderFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

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
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


@auto_docstring
class ParakeetPreTrainedModel(PreTrainedModel):
    config: ParakeetCTCConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ParakeetEncoderBlock"]
    _supports_flat_attention_mask = True
    _supports_sdpa = True
    _supports_flex_attn = True

    # TODO: @eustlb, add support when flash attention supports custom attention bias
    _supports_flash_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ParakeetEncoderBlock,
        "attentions": ParakeetEncoderAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, ParakeetEncoderAttention):
            init.normal_(module.bias_u, mean=0.0, std=std)
            init.normal_(module.bias_v, mean=0.0, std=std)
        elif isinstance(module, ParakeetEncoderRelPositionalEncoding):
            encoder_config = getattr(self.config, "encoder_config", self.config)
            inv_freq = 1.0 / (
                10000.0
                ** (torch.arange(0, encoder_config.hidden_size, 2, dtype=torch.int64) / encoder_config.hidden_size)
            )
            init.copy_(module.inv_freq, inv_freq)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)

        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        lengths = input_lengths

        for _ in range(num_layers):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0
            lengths = torch.floor(lengths)

        return lengths.to(dtype=torch.int)

    def _get_output_attention_mask(self, attention_mask: torch.Tensor, target_length: int | None = None):
        """
        Convert the input attention mask to its subsampled form. `target_length` sets the desired output length, useful
        when the attention mask length differs from `sum(-1).max()` (i.e., when the longest sequence in the batch is padded)
        """
        output_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        # Use target_length if provided, otherwise use max length in batch
        max_length = target_length if target_length is not None else output_lengths.max()
        attention_mask = torch.arange(max_length, device=attention_mask.device) < output_lengths[:, None]
        return attention_mask


@auto_docstring(
    custom_intro="""
    The Parakeet Encoder model, based on the [Fast Conformer architecture](https://huggingface.co/papers/2305.05084).
    """
)
class ParakeetEncoder(ParakeetPreTrainedModel):
    config: ParakeetEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop

        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = ParakeetEncoderSubsamplingConv2D(config)
        self.encode_positions = ParakeetEncoderRelPositionalEncoding(config)

        self.layers = nn.ModuleList(
            [ParakeetEncoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*):
            Whether to return the output attention mask.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = ParakeetEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """

        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
            attention_mask = output_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            attention_mask = attention_mask & attention_mask.transpose(1, 2)
            attention_mask = attention_mask.unsqueeze(1)

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
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        return ParakeetEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_attention_mask and output_mask is not None else None,
        )


@dataclass
class ParakeetCTCGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet CTC model generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor
    logits: tuple[torch.FloatTensor] | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


@dataclass
class ParakeetTDTGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet TDT model generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        token_timestamps (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token-level timestamps in seconds indicating when each token was emitted. Only returned when
            `return_timestamps=True` is passed to `generate()`.
        token_durations (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token-level durations in frames indicating how many frames each token spans. Only returned when
            `return_timestamps=True` is passed to `generate()`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple of tuples (one element for each layer of the encoder) of `torch.FloatTensor` of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions from the encoder.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple of tuples (one element for each layer of the encoder) of `torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden states from the encoder.
    """

    sequences: torch.LongTensor
    token_timestamps: torch.FloatTensor | None = None
    token_durations: torch.LongTensor | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


@auto_docstring(
    custom_intro="""
    Parakeet Encoder with a Connectionist Temporal Classification (CTC) head.
    """
)
class ParakeetForCTC(ParakeetPreTrainedModel):
    config: ParakeetCTCConfig

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        # Conv rather than linear to be consistent with NeMO decoding layer
        self.ctc_head = nn.Conv1d(config.encoder_config.hidden_size, config.vocab_size, kernel_size=1)

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```"""

        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states.transpose(1, 2)).transpose(1, 2)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_features, dtype=torch.long)
            )
            input_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))

            # assuming that padded tokens are filled with pad_token_id when not being attended to
            labels_mask = labels != self.config.pad_token_id
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetCTCGenerateOutput | torch.LongTensor:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        kwargs["return_dict"] = True
        outputs: CausalLMOutput = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        # greedy decoding
        sequences = outputs.logits.argmax(dim=-1)

        # mask out padded tokens
        if attention_mask is not None:
            attention_mask = self._get_output_attention_mask(attention_mask, target_length=sequences.shape[1])
            sequences[~attention_mask] = self.config.pad_token_id

        if return_dict_in_generate:
            return ParakeetCTCGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


class ParakeetTDTDecoder(nn.Module):
    """LSTM-based prediction network for TDT."""

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_size)
        self.lstm = nn.LSTM(
            input_size=config.decoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_state: torch.Tensor | None = None,
        cell_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.decoder_projector.weight.device)
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(
                self.config.num_decoder_layers,
                input_ids.shape[0],
                self.config.decoder_hidden_size,
                device=self.decoder_projector.weight.device,
                dtype=self.decoder_projector.weight.dtype,
            )
            cell_state = torch.zeros_like(hidden_state)
        hidden_state = hidden_state.to(self.decoder_projector.weight.device)
        cell_state = cell_state.to(self.decoder_projector.weight.device)

        embeddings = self.embedding(input_ids)
        lstm_output, (hidden_state, cell_state) = self.lstm(embeddings, (hidden_state, cell_state))
        decoder_output = self.decoder_projector(lstm_output)
        return decoder_output, hidden_state, cell_state


# TODO (ebezzam) eventually move to audio_utils or loss_utils for common usage?
def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    durations: list[int],
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute TDT (Token-and-Duration Transducer) loss (https://arxiv.org/abs/2304.06795).

    Ported from NeMo's `TDTLossPytorch`. Unlike standard RNNT loss, this loss trains both
    the token prediction head and the duration prediction head. Uses vectorized anti-diagonal
    processing for efficiency: all (t, u) pairs on each anti-diagonal t+u=n are computed in
    parallel as batched tensor operations.

    Args:
        token_logits: Token logits of shape `(batch, T, U+1, vocab_size+1)`.
        duration_logits: Duration logits of shape `(batch, T, U+1, num_durations)`.
        targets: Target labels of shape `(batch, U)`.
        logit_lengths: Encoder output lengths of shape `(batch,)`.
        target_lengths: Target lengths of shape `(batch,)`.
        blank: Blank token id.
        durations: List of duration values (e.g., `[0, 1, 2, 3, 4]`).
        sigma: Logit undernormalization constant (see TDT paper). Defaults to `0.0`.
        reduction: Loss reduction method. One of `"mean"`, `"sum"`, or `"none"`. Defaults to `"mean"`.

    Returns:
        Scalar loss tensor (or per-example losses if `reduction="none"`).

    """
    device = token_logits.device
    batch_size, max_t, max_u, _ = token_logits.shape

    # Apply log-softmax to get log probabilities
    token_log_probs = torch.log_softmax(token_logits, dim=-1) - sigma
    duration_log_probs = torch.log_softmax(duration_logits, dim=-1)

    log_alpha = torch.full((batch_size, max_t, max_u), float("-inf"), device=device)
    log_alpha[:, 0, 0] = 0.0

    # Precompute blank and label log-probs for vectorized access
    blank_log_probs = token_log_probs[:, :, :, blank]

    if max_u > 1:
        targets_expanded = targets.unsqueeze(1).expand(-1, max_t, -1)  # (batch, T, U_labels)
        label_log_probs = torch.gather(
            token_log_probs[:, :, : max_u - 1, :],  # (batch, T, U-1, vocab)
            dim=3,
            index=targets_expanded.unsqueeze(-1),
        ).squeeze(-1)  # (batch, T, U-1)

    # Process anti-diagonals: all (t, u) with t + u = n have no mutual dependencies
    for n in range(1, max_t + max_u - 1):
        u_start = max(0, n - max_t + 1)
        u_end = min(n + 1, max_u)
        u_indices = torch.arange(u_start, u_end, device=device)
        t_indices = n - u_indices

        all_candidates = []

        for i, dur in enumerate(durations):
            t_prev = t_indices - dur
            valid_t = t_prev >= 0

            if not valid_t.any():
                continue

            t_src = t_prev.clamp(min=0)

            # Blank arcs (dur > 0): from (t-dur, u) to (t, u)
            if dur > 0:
                contrib = (
                    log_alpha[:, t_src, u_indices]
                    + blank_log_probs[:, t_src, u_indices]
                    + duration_log_probs[:, t_src, u_indices, i]
                )
                contrib = torch.where(valid_t.unsqueeze(0), contrib, torch.tensor(float("-inf"), device=device))
                all_candidates.append(contrib)

            # Label arcs: from (t-dur, u-1) to (t, u), only if u > 0
            valid_u = u_indices > 0
            valid_both = valid_t & valid_u
            if valid_both.any():
                u_src = (u_indices - 1).clamp(min=0)
                u_src_label = u_src.clamp(max=max_u - 2) if max_u > 1 else u_src

                contrib = (
                    log_alpha[:, t_src, u_src]
                    + label_log_probs[:, t_src, u_src_label]
                    + duration_log_probs[:, t_src, u_src, i]
                )
                contrib = torch.where(valid_both.unsqueeze(0), contrib, torch.tensor(float("-inf"), device=device))
                all_candidates.append(contrib)

        if all_candidates:
            stacked = torch.stack(all_candidates, dim=0)
            log_alpha[:, t_indices, u_indices] = torch.logsumexp(stacked, dim=0)

    # Terminal probability: sum over blank arcs that reach (T, U) from (T-dur, U)
    batch_idx = torch.arange(batch_size, device=device)
    log_probs = torch.full((batch_size,), float("-inf"), device=device)
    for i, dur in enumerate(durations):
        if dur == 0:
            continue
        t_final = logit_lengths - dur
        valid = t_final >= 0
        if not valid.any():
            continue

        t_clamped = t_final.clamp(min=0)
        terminal = (
            log_alpha[batch_idx, t_clamped, target_lengths]
            + token_log_probs[batch_idx, t_clamped, target_lengths, blank]
            + duration_log_probs[batch_idx, t_clamped, target_lengths, i]
        )
        combined = torch.stack([log_probs, terminal], dim=0)
        log_probs = torch.where(valid, torch.logsumexp(combined, dim=0), log_probs)

    losses = -log_probs

    if reduction == "mean":
        return (losses / target_lengths.float()).mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        return (losses / target_lengths.float()).mean()


class ParakeetTDTJointNetwork(nn.Module):
    """Joint network that combines encoder and decoder outputs to predict tokens and durations."""

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__()
        self.encoder_projector = nn.Linear(config.encoder_config.hidden_size, config.decoder_hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.token_head = nn.Linear(config.decoder_hidden_size, config.vocab_size)
        self.duration_head = nn.Linear(config.decoder_hidden_size, len(config.durations))

    def forward(
        self,
        decoder_output: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        projected_encoder_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected_encoder_output is None:
            if encoder_output is None:
                raise ValueError("Either encoder_output or projected_encoder_output must be provided.")
            projected_encoder_output = self.encoder_projector(encoder_output)
        joint_output = self.activation(projected_encoder_output + decoder_output)
        return self.token_head(joint_output), self.duration_head(joint_output)


@auto_docstring(
    custom_intro="""
    Parakeet Encoder with a TDT (Token Duration Transducer) head.
    """
)
class ParakeetForTDT(ParakeetPreTrainedModel):
    config: ParakeetTDTConfig

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.decoder = ParakeetTDTDecoder(config)
        self.joint = ParakeetTDTJointNetwork(config)

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForTDT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-tdt-0.6b-v3"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForTDT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> outputs = model(**inputs)
        ```
        """
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        loss = None
        if labels is not None:
            # Compute encoder output lengths
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(input_features.shape[:-1], dtype=torch.long, device=input_features.device)
            )
            encoder_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))

            # Prepare labels for TDT loss
            target_lengths = (labels != self.config.pad_token_id).sum(-1)

            # Get joint decoder outputs
            blank_tokens = torch.full(
                (labels.shape[0], 1), self.config.blank_token_id, dtype=labels.dtype, device=labels.device
            )
            decoder_input = torch.cat([blank_tokens, labels], dim=1)
            decoder_output, _, _ = self.decoder(decoder_input)
            token_logits, duration_logits = self.joint(
                decoder_output=decoder_output.unsqueeze(1),
                encoder_output=encoder_hidden_states.unsqueeze(2),
            )

            loss = tdt_loss(
                token_logits=token_logits.float(),
                duration_logits=duration_logits.float(),
                targets=labels.to(token_logits.device).int(),
                logit_lengths=encoder_lengths.to(token_logits.device).int(),
                target_lengths=target_lengths.to(token_logits.device).int(),
                blank=self.config.blank_token_id,
                durations=self.config.durations,
                reduction="mean",
            )

        return CausalLMOutput(
            loss=loss,
            logits=encoder_hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_timestamps: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetTDTGenerateOutput | torch.LongTensor:
        r"""
        Perform TDT greedy decoding to generate token sequences.

        Args:
            return_timestamps (`bool`, *optional*, defaults to `False`):
                Whether to return per-token timestamps and durations. When `True`, forces
                `return_dict_in_generate=True` and includes `token_timestamps` and `token_durations` in the output.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForTDT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-tdt-0.6b-v3"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForTDT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=processor.feature_extractor.sampling_rate)
        >>> inputs = inputs.to(model.device, dtype=model.dtype)
        >>> output = model.generate(**inputs, return_dict_in_generate=True, return_timestamps=True)

        >>> decoded_output, decoded_timestamps = processor.decode(
        ...     output.sequences,
        ...     token_timestamps=output.token_timestamps,
        ...     token_durations=output.token_durations,
        ...     skip_special_tokens=True
        ... )
        >>> print("Transcription:", decoded_output)
        >>> print("Timestamped tokens:", decoded_timestamps)
        ```
        """
        kwargs["return_dict"] = True
        if return_timestamps:
            return_dict_in_generate = True
        outputs: CausalLMOutput = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        # greedy TDT decoding, `GreedyBatchedTDTLabelLoopingComputer.torch_impl` in NeMo
        encoder_hidden_states = outputs.logits
        batch_size, sequence_length = encoder_hidden_states.shape[:2]
        device = encoder_hidden_states.device
        if attention_mask is not None:
            encoder_attention_mask = self._get_output_attention_mask(attention_mask, target_length=sequence_length)
            valid_lengths = encoder_attention_mask.sum(dim=1).int()
        else:
            valid_lengths = torch.full((batch_size,), sequence_length, dtype=torch.int, device=device)

        # Initialization
        hidden_state, cell_state = None, None
        prev_tokens = torch.full((batch_size, 1), self.config.blank_token_id, dtype=torch.long, device=device)
        decoder_output, hidden_state, cell_state = self.decoder(prev_tokens, hidden_state, cell_state)
        decoder_output = decoder_output.to(device)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)

        batch_indices = torch.arange(batch_size, device=device)
        time_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        time_indices_current_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        active_mask = time_indices < valid_lengths
        active_mask_prev = torch.zeros_like(active_mask)

        zeros_symbols = torch.zeros(batch_size, dtype=torch.long, device=device)
        symbols_per_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        last_label_time = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        max_output_len = sequence_length * self.config.max_symbols_per_step
        all_tokens_tensor = torch.full(
            (batch_size, max_output_len), self.config.pad_token_id, dtype=torch.long, device=device
        )
        token_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        if return_timestamps:
            all_frame_indices = torch.zeros((batch_size, max_output_len), dtype=torch.long, device=device)
            all_durations_tensor = torch.zeros((batch_size, max_output_len), dtype=torch.long, device=device)

        # separately call encoder projection to avoid redundant computation inside loop
        projected_encoder_output = self.joint.encoder_projector(encoder_hidden_states).to(device)

        while active_mask.any():
            active_mask_prev.copy_(active_mask)
            safe_time_indices = torch.clamp(time_indices, max=sequence_length - 1)
            projected_encoder_frames = projected_encoder_output[batch_indices, safe_time_indices].unsqueeze(1)

            token_logits, duration_logits = self.joint(
                decoder_output,
                projected_encoder_output=projected_encoder_frames,
            )
            token_logits = token_logits.squeeze(1).to(device)
            duration_logits = duration_logits.squeeze(1).to(device)

            tokens = token_logits.argmax(dim=-1)
            durations = duration_logits.argmax(dim=-1)

            # Force blank duration >= 1 to guarantee forward progress
            blank_mask = active_mask_prev & (tokens == self.config.blank_token_id)
            durations = durations.masked_fill(blank_mask & (durations == 0), 1)

            # Save pre-advance position for timestamp recording
            time_indices_current_labels.copy_(time_indices)

            # Advance time for all active elements
            time_indices = time_indices + durations.masked_fill(~active_mask, 0)
            safe_time_indices = torch.clamp(time_indices, max=sequence_length - 1)
            active_mask = time_indices < valid_lengths
            advance_mask = active_mask & blank_mask

            # Inner loop: skip past consecutive blanks to find non-blank
            while advance_mask.any():
                time_indices_current_labels = torch.where(advance_mask, time_indices, time_indices_current_labels)
                projected_encoder_frames = projected_encoder_output[batch_indices, safe_time_indices].unsqueeze(1)

                token_logits, duration_logits = self.joint(
                    decoder_output, projected_encoder_output=projected_encoder_frames
                )
                token_logits = token_logits.squeeze(1).to(device)
                duration_logits = duration_logits.squeeze(1).to(device)

                more_tokens = token_logits.argmax(dim=-1)
                more_durations = duration_logits.argmax(dim=-1)
                tokens = torch.where(advance_mask, more_tokens, tokens)
                durations = torch.where(advance_mask, more_durations, durations)

                blank_mask = tokens == self.config.blank_token_id
                durations = durations.masked_fill(blank_mask & (durations == 0), 1)

                time_indices = torch.where(advance_mask, time_indices + durations, time_indices)
                safe_time_indices = torch.clamp(time_indices, max=sequence_length - 1)
                active_mask = time_indices < valid_lengths
                advance_mask = active_mask & blank_mask

            # Record results for non-blank tokens found
            emit_mask = active_mask_prev & (tokens != self.config.blank_token_id)
            emit_indices = token_counts[emit_mask]
            all_tokens_tensor[emit_mask, emit_indices] = tokens[emit_mask]
            if return_timestamps:
                all_frame_indices[emit_mask, emit_indices] = time_indices_current_labels[emit_mask]
                all_durations_tensor[emit_mask, emit_indices] = durations[emit_mask]
            token_counts += emit_mask.long()

            new_decoder_output, new_hidden_state, new_cell_state = self.decoder(
                tokens.unsqueeze(1), hidden_state, cell_state
            )
            new_decoder_output = new_decoder_output.to(device)
            new_hidden_state = new_hidden_state.to(device)
            new_cell_state = new_cell_state.to(device)

            emit_mask_expanded = emit_mask.view(batch_size, 1, 1)
            decoder_output = torch.where(emit_mask_expanded, new_decoder_output, decoder_output)
            emit_mask_state = emit_mask.view(1, batch_size, 1)
            hidden_state = torch.where(emit_mask_state, new_hidden_state, hidden_state)
            cell_state = torch.where(emit_mask_state, new_cell_state, cell_state)

            # Track symbols emitted per time step; force advance when max_symbols reached
            time_changed = time_indices_current_labels != last_label_time
            symbols_per_step = torch.where(time_changed, zeros_symbols, symbols_per_step)
            symbols_per_step = torch.where(emit_mask, symbols_per_step + 1, symbols_per_step)
            last_label_time = torch.where(emit_mask, time_indices_current_labels, last_label_time)
            force_advance = active_mask & (symbols_per_step >= self.config.max_symbols_per_step)
            time_indices = time_indices + force_advance.long()
            symbols_per_step = symbols_per_step.masked_fill(force_advance, 0)
            active_mask = time_indices < valid_lengths

        # Guard against edge case where no tokens were decoded (e.g. silent audio)
        max_len = max(token_counts.max().item(), 1)
        sequences = all_tokens_tensor[:, :max_len]
        token_timestamps, token_durations = None, None
        if return_timestamps:
            token_timestamps = all_frame_indices[:, :max_len]
            token_durations = all_durations_tensor[:, :max_len]

        if return_dict_in_generate:
            return ParakeetTDTGenerateOutput(
                sequences=sequences,
                token_timestamps=token_timestamps,
                token_durations=token_durations,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )
        return sequences


__all__ = ["ParakeetForCTC", "ParakeetForTDT", "ParakeetEncoder", "ParakeetPreTrainedModel"]
