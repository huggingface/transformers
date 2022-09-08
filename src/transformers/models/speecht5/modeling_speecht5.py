# coding=utf-8
# Copyright 2022 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SpeechT5 model."""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import torch_int_div
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_speecht5 import SpeechT5Config


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "SpeechT5Config"
_PROCESSOR_FOR_DOC = "SpeechT5Processor"

# Base docstring
_CHECKPOINT_FOR_DOC = "TODO"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 53.48


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->SpeechT5
class SpeechT5SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SpeechT5
class SpeechT5PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->SpeechT5
class SpeechT5SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->SpeechT5
class SpeechT5FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
                SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->SpeechT5
class SpeechT5FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class SpeechT5SpeechEncoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_encoder = SpeechT5FeatureEncoder(config)
        self.feature_projection = SpeechT5FeatureProjection(config)
        self.pos_conv_embed = SpeechT5PositionalConvEmbedding(config)
        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
            config.max_source_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        extract_features = self.feature_encoder(input_values)
        extract_features = extract_features.transpose(1, 2)

        # TODO: original implementation does the following which is only
        # used when the argument `require_feat_pen=True`
        # x = x.transpose(1, 2) # [batch, hidden_size, length]
        # if target_list is not None:
        #     x, target_list = self.forward_targets(x, target_list)
        # features_pen = x.float().pow(2).mean()
        # x = x.transpose(1, 2) # [batch, length, hidden_size]

        # TODO: note that Wav2Vec2Model calls `_get_feature_vector_attention_mask` at this point

        hidden_states, extract_features = self.feature_projection(extract_features)

        # TODO: original implementation does the following when `mask=True`
        # note that `x` here is the same as `extract_features`, i.e. after the
        # layer norm from `feature_projection` but before the conv + dropout!
        # encoder_padding_mask = padding_mask
        # encoder_padding_mask = self.forward_padding_mask(x, encoder_padding_mask)

        # TODO: original implementation does the following when `mask=True`
        # if mask:
        #     x, mask_indices = self.apply_hubert_mask(
        #         x, encoder_padding_mask
        #     )
        # else:
        #     x = x
        #     mask_indices = None

        # TODO: note that Wav2Vec2Model calls `_mask_hidden_states` at this point

        # TODO: Wav2Vec2Encoder does the following at this point
        # if attention_mask is not None:
        #     # make sure padded tokens output 0
        #     expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        #     hidden_states[~expand_attention_mask] = 0

        #     # extend attention_mask
        #     attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        #     attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        #     attention_mask = attention_mask.expand(
        #         attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        #     )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings

        # TODO: this is how Speech2TextEncoder does it
        # if attention_mask is not None:
        #     attention_mask = self._get_feature_vector_attention_mask(inputs_embeds.shape[1], attention_mask)
        #     padding_mask = attention_mask.ne(1).long()
        # else:
        #     padding_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        #embed_pos = self.embed_positions(padding_mask)

        # TODO: for now I'm just faking the size of the attention mask
        attention_mask = torch.ones(hidden_states.shape[:2]).long()

        # TODO: the sinusoidal embeddings are incorrect currently, not sure why yet
        positions = self.embed_positions(attention_mask)
        #print("positions", positions.shape, positions)
        hidden_states = hidden_states + positions

        return hidden_states, extract_features


class SpeechT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpeechT5Config
    base_model_prefix = "speecht5"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # # gumbel softmax requires special init
        # if isinstance(module, SpeechT5GumbelVectorQuantizer):
        #     module.weight_proj.weight.data.normal_(mean=0.0, std=1)
        #     module.weight_proj.bias.data.zero_()
        #     nn.init.uniform_(module.codevectors)
        # elif isinstance(module, SpeechT5PositionalConvEmbedding):
        #     nn.init.normal_(
        #         module.conv.weight,
        #         mean=0,
        #         std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
        #     )
        #     nn.init.constant_(module.conv.bias, 0)
        # elif isinstance(module, SpeechT5FeatureProjection):
        #     k = math.sqrt(1 / module.projection.in_features)
        #     nn.init.uniform_(module.projection.weight, a=-k, b=k)
        #     nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # elif isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # elif isinstance(module, nn.Conv1d):
        #     nn.init.kaiming_normal_(module.weight)

        #     if module.bias is not None:
        #         k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
        #         nn.init.uniform_(module.bias, a=-k, b=k)
        pass

    # def _get_feat_extract_output_lengths(
    #     self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    # ):
    #     """
    #     Computes the output length of the convolutional layers
    #     """

    #     add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

    #     def _conv_out_length(input_length, kernel_size, stride):
    #         # 1D convolutional layer output length formula taken
    #         # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    #         return torch_int_div(input_length - kernel_size, stride) + 1

    #     for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
    #         input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    #     if add_adapter:
    #         for _ in range(self.config.num_adapter_layers):
    #             input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

    #     return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, (SpeechT5Encoder, SpeechT5EncoderStableLayerNorm, SpeechT5FeatureEncoder)):
    #         module.gradient_checkpointing = value


SPEECHT5_START_DOCSTRING = r"""
    TODO
"""


SPEECHT5_INPUTS_DOCSTRING = r"""
    TODO

    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`SpeechT5Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [speecht5-base](https://huggingface.co/TODO), `attention_mask` should **not** be passed to avoid degraded
            performance when doing batched inference. For such models `input_values` should simply be padded with 0 and
            passed without `attention_mask`. Be aware that these models also yield slightly different results depending
            on whether `input_values` is padded or not.

            </Tip>

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare SpeechT5 Model transformer outputting raw hidden-states without any specific head on top.",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5Model(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.config = config

        self.speech_encoder_prenet = SpeechT5SpeechEncoderPrenet(config)

        # # model only needs masking vector if mask prob is > 0.0
        # if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
        #     self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # if config.do_stable_layer_norm:
        #     self.encoder = SpeechT5EncoderStableLayerNorm(config)
        # else:
        #     self.encoder = SpeechT5Encoder(config)

        # self.adapter = SpeechT5Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    # def freeze_feature_encoder(self):
    #     """
    #     Calling this function will disable the gradient computation for the feature encoder so that its parameter will
    #     not be updated during training.
    #     """
    #     self.feature_extractor._freeze_parameters()

    # def _mask_hidden_states(
    #     self,
    #     hidden_states: torch.FloatTensor,
    #     mask_time_indices: Optional[torch.FloatTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    # ):
    #     """
    #     Masks extracted features along time axis and/or along feature axis according to
    #     [SpecAugment](https://arxiv.org/abs/1904.08779).
    #     """

    #     # `config.apply_spec_augment` can set masking to False
    #     if not getattr(self.config, "apply_spec_augment", True):
    #         return hidden_states

    #     # generate indices & apply SpecAugment along time axis
    #     batch_size, sequence_length, hidden_size = hidden_states.size()

    #     if mask_time_indices is not None:
    #         # apply SpecAugment along time axis with given mask_time_indices
    #         hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    #     elif self.config.mask_time_prob > 0 and self.training:
    #         mask_time_indices = _compute_mask_indices(
    #             (batch_size, sequence_length),
    #             mask_prob=self.config.mask_time_prob,
    #             mask_length=self.config.mask_time_length,
    #             attention_mask=attention_mask,
    #             min_masks=self.config.mask_time_min_masks,
    #         )
    #         mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
    #         hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

    #     if self.config.mask_feature_prob > 0 and self.training:
    #         # generate indices & apply SpecAugment along feature axis
    #         mask_feature_indices = _compute_mask_indices(
    #             (batch_size, hidden_size),
    #             mask_prob=self.config.mask_feature_prob,
    #             mask_length=self.config.mask_feature_length,
    #             min_masks=self.config.mask_feature_min_masks,
    #         )
    #         mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
    #         mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
    #         hidden_states[mask_feature_indices] = 0

    #     return hidden_states

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,  # TODO: probably different output type
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, extract_features = self.speech_encoder_prenet(input_values, attention_mask)

        # TODO: not sure if this should go into SpeechEncoderPrenet too
        #       is this equivalent to `forward_padding_mask` from the original SpeechEncoderPrenet?
        # if attention_mask is not None:
        #     # compute reduced attention_mask corresponding to feature vectors
        #     attention_mask = self._get_feature_vector_attention_mask(
        #         extract_features.shape[1], attention_mask, add_adapter=False
        #     )

        # TODO: not sure if this should go into SpeechEncoderPrenet too
        #       is this equivalent to `apply_hubert_mask` from the original SpeechEncoderPrenet?
        # hidden_states = self._mask_hidden_states(
        #     hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # )

        # encoder_outputs = self.encoder(
        #     hidden_states,
        #     attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # hidden_states = encoder_outputs[0]

        # if self.adapter is not None:
        #     hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=None, #encoder_outputs.hidden_states,
            attentions=None, #encoder_outputs.attentions,
        )


@add_start_docstrings(
    """SpeechT5 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForCTC(SpeechT5PreTrainedModel):
    base_model_prefix = "speecht5"
    _keys_to_ignore_on_load_missing = [
        r"speecht5.speech_encoder_prenet.embed_positions.weights",
    ]
    _keys_to_ignore_on_save = [
        r"speecht5.speech_encoder_prenet.embed_positions.weights",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.speecht5 = SpeechT5Model(config)
        # self.dropout = nn.Dropout(config.final_dropout)

        # if config.vocab_size is None:
        #     raise ValueError(
        #         f"You are trying to instantiate {self.__class__} with a configuration that "
        #         "does not define the vocabulary size of the language model head. Please "
        #         "instantiate the model as follows: `SpeechT5ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
        #         "or define `vocab_size` of your model's configuration."
        #     )
        # output_hidden_size = (
        #     config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        # )
        # self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.speecht5.feature_extractor._freeze_parameters()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.speecht5(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # hidden_states = self.dropout(hidden_states)

        logits = None
        # logits = self.lm_head(hidden_states)

        loss = None
        # if labels is not None:

        #     if labels.max() >= self.config.vocab_size:
        #         raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        #     # retrieve loss input_lengths from attention_mask
        #     attention_mask = (
        #         attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        #     )
        #     input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        #     # assuming that padded tokens are filled with -100
        #     # when not being attended to
        #     labels_mask = labels >= 0
        #     target_lengths = labels_mask.sum(-1)
        #     flattened_targets = labels.masked_select(labels_mask)

        #     # ctc_loss doesn't support fp16
        #     log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        #     with torch.backends.cudnn.flags(enabled=False):
        #         loss = nn.functional.ctc_loss(
        #             log_probs,
        #             flattened_targets,
        #             input_lengths,
        #             target_lengths,
        #             blank=self.config.pad_token_id,
        #             reduction=self.config.ctc_loss_reduction,
        #             zero_infinity=self.config.ctc_zero_infinity,
        #         )

        # if not return_dict:
        #     output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        #     return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
