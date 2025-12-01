# coding=utf-8
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
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from .configuration_parakeet import ParakeetCTCConfig, ParakeetEncoderConfig


class ParakeetEncoderRelPositionalEncoding(nn.Module):
    """Relative positional encoding for Parakeet."""

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
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
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
        position_embeddings: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
    def __init__(self, config: ParakeetEncoderConfig, layer_idx: Optional[int] = None):
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
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
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

    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value accross the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, ParakeetEncoderAttention):
            # Initialize positional bias parameters
            module.bias_u.data.normal_(mean=0.0, std=std)
            module.bias_v.data.normal_(mean=0.0, std=std)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = self.config.encoder_config if isinstance(self.config, ParakeetCTCConfig) else self.config

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

    def _get_output_attention_mask(self, attention_mask: torch.Tensor, target_length: Optional[int] = None):
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
    @check_model_inputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
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

        if attention_mask is not None:
            attention_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
            attention_mask = attention_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
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

        return BaseModelOutput(last_hidden_state=hidden_states)


@dataclass
class ParakeetGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet models.

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
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None


@auto_docstring(
    custom_intro="""
    Parakeet Encoder with a Connectionist Temporal Classification (CTC) head.
    """
)
class ParakeetForCTC(ParakeetPreTrainedModel):
    config: ParakeetCTCConfig

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__(config)
        self.encoder = ParakeetEncoder(config.encoder_config)
        # Conv rather than linear to be consistent with NeMO decoding layer
        self.ctc_head = nn.Conv1d(config.encoder_config.hidden_size, config.vocab_size, kernel_size=1)

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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

            # assuming that padded tokens are filled with -100
            # when not being attended to
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
        attention_mask: Optional[torch.Tensor] = None,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[ParakeetGenerateOutput, torch.LongTensor]:
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
            return ParakeetGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


__all__ = ["ParakeetForCTC", "ParakeetEncoder", "ParakeetPreTrainedModel"]
