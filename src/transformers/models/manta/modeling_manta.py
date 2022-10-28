# coding=utf-8
# Copyright 2022 Mesh TensorFlow authors, Manta Authors and HuggingFace Inc. team.
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
""" PyTorch Manta model."""


import math
from multiprocessing import pool
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_manta import MantaConfig
from ...models.t5.configuration_t5 import T5Config
from ...models.t5.modeling_t5 import (
    T5LayerNorm,
    T5Model,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5Attention,
    T5Stack,
    __HEAD_MASK_WARNING_MSG,
)
from ...models.longformer import (
    LongformerConfig,
    LongformerModel
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MantaConfig"
_TOKENIZER_FOR_DOC = "ByT5Tokenizer"

MANTA_PRETRAINED_MODEL_ARCHIVE_LIST = []


def gaussian_pdf(x):
    return torch.exp(-x * x / 2.0)

def pad_block_embeddings(block_embeddings, pad_length):
    if not pad_length:
        return block_embeddings

    padding_tensor_len = max(pad_length - block_embeddings.size(1), 0)

    padding_tensor = torch.zeros(
        (block_embeddings.size(0), padding_tensor_len, block_embeddings.size(2)),
        device=block_embeddings.device,
        dtype=block_embeddings.dtype,
    )
    return torch.cat([block_embeddings[:, :pad_length, :], padding_tensor], dim=1)

class MantaHighway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.
    # Parameters
    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim,
        num_layers=1,
        activation="relu",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self._activation = getattr(torch.nn.functional, activation)
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

class MantaFrontierPredictor(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_attention_heads,
        dropout_rate,
        attention_window,
        max_length,
    ):
        super().__init__()

        # First, find out what the maximum position will be after tensors are padded to a multiple of local_transformer_attention_window.
        # Then, add 1 because LongFormer position embeddings are bugged when passed inputs_embeds.
        max_position_embeddings = (max_length // attention_window + 1) * attention_window + 1
        self.hidden_size = hidden_size

        self.config = LongformerConfig(
            attention_probs_dropout_prob=dropout_rate,
            attention_window=attention_window,
            hidden_act="gelu",
            hidden_dropout_prob=dropout_rate,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_layers,
            position_embedding_type="absolute",  # Actually cannot be changed
            vocab_size=1,  # Remove almost entirely the embeddings
            pad_token_id=0,
        )
        self.local_transformer = LongformerModel(self.config)

        self.output_projection = nn.Linear(hidden_size, 1)

    def forward(self, embeddings, attention_mask):
        longformer_output = self.local_transformer(inputs_embeds=embeddings, attention_mask=attention_mask)

        projection_outputs = self.output_projection(longformer_output.last_hidden_state)

        frontier_predictions = torch.sigmoid(projection_outputs.squeeze(-1))

        return frontier_predictions

class MantaConvFeatures(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        padding,
    ):
        """
        This nn.Module "decomposes" the convolution in order to extract and cache feature maps. This amounts to
        computing an element-wise multiplication between weights of size (hidden_dim, kernel_size) and the input.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding

        if groups == in_channels:
            assert (
                in_channels == out_channels
            ), "When using `groups = in_channels`, make sure to have `in_channels == out_channels`"
            self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, out_channels))
        elif self.groups == 1:
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        else:
            raise ValueError("MantaConvFeatures only supports `groups = 1` or `groups = in_channels`")

        left_pad = (kernel_size - 1) // 2
        self.pad = (left_pad, kernel_size - 1 - left_pad)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        See https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d, in the `_ConvNd` class :
            > Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
            > uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
            > For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573"

        The reason we permute the weights before init is because `kaiming_uniform_` uses the number of in and out
        features for initialization, which are computed as tensor.size(0) and tensor.size(1). However, these
        dimensions do not correspond for my weights.
        """
        if self.groups == self.out_channels:
            nn.init.kaiming_uniform_(self.weight.permute(3, 0, 1, 2), a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        if self.groups == 1:
            return self.forward_matmul(x)
        else:
            return self.forward_elementwise(x)

    def forward_matmul(self, x: torch.Tensor):

        if self.padding == "same":
            padded_x = self._pad_pre_conv(x)
        else:
            padded_x = x

        bs, _, seq_len = padded_x.size()

        padded_x = padded_x.transpose(-1, -2)
        # Size: (bs, seq_len+pad, hidden)

        out = padded_x.matmul(self.weight.view(self.weight.size(0), -1)).view(bs, seq_len, self.out_channels, -1)
        # Size: (bs, seq_len+pad, hidden, kernel_size)

        return out.permute(0, 2, 3, 1)

    def forward_elementwise(self, x: torch.Tensor):
        assert len(x.size()) == 3
        assert x.size(1) == self.out_channels
        # Size: (bs, hidden, seq_len)

        if self.padding == "same":
            padded_x = self._pad_pre_conv(x)
        else:
            padded_x = x

        # Unsqueeze for broadcasting with the kernel_size dim of the filters
        padded_x = padded_x.transpose(-1, -2).unsqueeze(2)
        # Size: (bs, seq_len, 1, hidden)

        out = padded_x * self.weight
        # Size: (bs, seq_len, kernel_size, hidden)

        return out.transpose(1, 3)

    def _pad_pre_conv(self, inp: torch.Tensor):
        """
        Pad with zeros at the beginning and end just like `nn.Conv1d`.
        """
        return nn.functional.pad(inp, self.pad, "constant", 0.0)

    def extra_repr(self):
        return "in_features={}, out_features={}, kernel_size={}, groups={}".format(
            self.in_channels, self.out_channels, self.kernel_size, self.groups
        )

class MantaCachedConvolutionPooling(nn.Module):
    def __init__(
        self,
        padding_length,
        output_dim,
        kernel_size,
        hidden_dim,
        depthwise_convolution,
        variance_regularization,
        n_highway_layers,
        highway_activation,
        mean_pool,
    ):
        super().__init__()
        self.padding_length = padding_length

        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.conv_output_dim = hidden_dim if isinstance(kernel_size, int) else sum([k_dim[1] for k_dim in kernel_size])

        self.eps = None
        self.variance_regularization = variance_regularization

        self.conv_layer = None
        self.highway = MantaHighway(self.conv_output_dim, n_highway_layers, highway_activation)
        # Since the sum of the hidden dimensions of all the filters might not match the language model hidden size, we
        # specify it here
        self.out_projection = nn.Linear(self.conv_output_dim, output_dim, bias=True)

        self.mean_pool = mean_pool

        self.depthwise_convolution = depthwise_convolution

        if isinstance(kernel_size, int):
            kernel_size = [[kernel_size, hidden_dim]]

        self.conv_layers = nn.Sequential(
            *[
                MantaConvFeatures(hidden_dim, h, k, groups=h if depthwise_convolution else 1, padding="same")
                for (k, h) in kernel_size
            ]
        )
        self.mean_pool = mean_pool

    def forward(self, unconstrained_separation_probs: torch.Tensor, byte_embeddings: torch.Tensor):
        device = unconstrained_separation_probs.device
        if self.eps is None:
            self.eps = 5 * torch.finfo(unconstrained_separation_probs.dtype).resolution
            self.variance_regularization = max(self.eps, self.variance_regularization)

        if self.conv_layer is not None:
            self.conv_layer = self.conv_layer.to(device)
        batch_size, seq_len, emb_dims = byte_embeddings.shape

        # We set the probability of the first token to be 0 therwise the cumsum will not work
        separation_probs = unconstrained_separation_probs.clone()
        separation_probs[:, 0] = 0

        assert separation_probs.shape == (batch_size, seq_len)

        # Compute the moments of the block_id random variable
        block_id_expectation = separation_probs.cumsum(axis=-1)
        block_id_std = torch.sqrt(
            (separation_probs * (1.0 - separation_probs)).cumsum(axis=-1) + self.variance_regularization
        )

        # Get the maximum number of blocks
        max_nb_blocks = min(seq_len, (block_id_expectation + 3 * block_id_std).max().int().item() + 1)
        possible_blocks_id = torch.arange(max_nb_blocks).to(device)

        # Get the block/byte proba using the Gaussian PDF
        log_scale = block_id_std[:, None, :].log()
        log_proba = (
            -((block_id_expectation[:, None, :] - possible_blocks_id[None, :, None]) ** 2)
            / (2 * block_id_std[:, None, :])
            - log_scale
            - math.log((2 * math.pi) ** 0.5)
        )
        block_byte_proba = log_proba.softmax(-2)

        token_size = block_byte_proba.sum(-1, keepdim=True)
        regularized_token_size = torch.maximum(token_size, torch.ones_like(token_size))

        if self.mean_pool:
            block_byte_proba_normalized = block_byte_proba / regularized_token_size
        else:
            # Makes no sense to regularize using sequence length in the max_pooling case.
            block_byte_proba_normalized = block_byte_proba

        block_embeddings = self.pooling(byte_embeddings, block_byte_proba_normalized)

        pad_length = min(self.padding_length, max_nb_blocks)

        block_embeddings = pad_block_embeddings(block_embeddings, pad_length)
        block_embeddings = self.highway(block_embeddings)
        block_embeddings = self.out_projection(block_embeddings)

        attention_mask = torch.ones_like(block_byte_proba)

        return block_embeddings, attention_mask, regularized_token_size, block_byte_proba

    def pooling(self, embeddings: torch.Tensor, block_byte_proba: torch.Tensor):
        block_embeddings = []

        for conv_layer in self.conv_layers:
            # First, compute the convolution maps SEPARATELY, i.e. without summing them together, only the element wise multiplication
            # This is similar to a cache that we'll reuse for each block probabilities.
            features = conv_layer(embeddings.transpose(1, 2)).permute(0, 3, 1, 2)
            # Size : (batch_size, seq_len + padding, hidden_dim, kernel_size)

            pad = conv_layer.pad

            for i in range(0, conv_layer.kernel_size):
                # We shift like that to match the padding done inside `conv_layer`
                features[..., i] = features[..., i].roll(pad[0] - i, 1)
            # Cut out the padded vector to obtain the right sequence length at the end
            features = features[:, pad[1] : features.size(1) - pad[0]]
            # Size : (batch_size, seq_len, hidden_dim, kernel_size)

            # Then, artificially sum the convolution features by shifting the input bytes
            padded_block_byte_proba = nn.functional.pad(block_byte_proba, pad, "constant", 0.0)
            expanded_block_byte_proba = []
            for i in range(0, conv_layer.kernel_size):
                rolled_proba = padded_block_byte_proba.clone().roll(pad[0] - i, -1)
                expanded_block_byte_proba.append(rolled_proba)
            expanded_block_byte_proba = torch.stack(expanded_block_byte_proba, -1)
            # We use :tensor.size(2) - pad instead of just :-pad because if pad = 0, we have an undesired behaviour where the whole sequence is removed
            expanded_block_byte_proba = expanded_block_byte_proba[
                :, :, pad[1] : expanded_block_byte_proba.size(2) - pad[0], :
            ]
            # Size : (batch_size, block_size, seq_len, kernel_size)

            if self.mean_pool:
                convolved = torch.einsum("b s h k, b B s k -> b B h", features, expanded_block_byte_proba)
            else:
                convolved = torch.einsum("b s h k, b B s k -> b B s h", features, expanded_block_byte_proba)
                convolved = convolved.max(dim=-2).values

            block_embeddings.append(convolved)

        block_embeddings = torch.cat(block_embeddings, dim=-1)

        return block_embeddings


class MantaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MantaConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (MantaModel, MantaForConditionalGeneration, MantaEncoderModel)):
            module.byte_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, MantaConvFeatures):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            if hasattr(module, "lm_head"):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class MantaModel(MantaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder_decoder.encoder.embed_tokens.weight",
        r"encoder_decoder.decoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"encoder_decoder.decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: MantaConfig):
        super().__init__(config)
        self.byte_embeddings = nn.Embedding(config.vocab_size, config.byte_embedding_dim)

        self.frontier_predictor = MantaFrontierPredictor(
            num_layers=config.frontier_predictor_num_layers,
            num_attention_heads=config.frontier_predictor_num_attention_heads,
            attention_window=config.frontier_predictor_attention_window,
            hidden_size=config.byte_embedding_dim,
            max_length=config.max_length_encoder_decoder,
            dropout_rate=config.dropout_rate,
        )

        self.pooler = MantaCachedConvolutionPooling(
            variance_regularization=config.pooling_variance_regularization,
            kernel_size=config.pooling_kernel_size,
            n_highway_layers=config.pooling_n_highway_layers,
            highway_activation=config.pooling_highway_activation,
            depthwise_convolution=config.pooling_depthwise_convolution,
            mean_pool=config.pooling_mean_pool,
            output_dim=config.d_model,
            padding_length=config.max_length_encoder_decoder,
            hidden_dim=config.byte_embedding_dim,
        )

        self.encoder_decoder = T5Model(
            T5Config(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_decoder_layers=config.num_decoder_layers,
                num_heads=config.num_heads,
                relative_attention_num_buckets=config.relative_attention_num_buckets,
                relative_attention_max_distance=config.relative_attention_max_distance,
                dropout_rate=config.dropout_rate,
                layer_norm_epsilon=config.layer_norm_epsilon,
                initializer_factor=config.initializer_factor,
                feed_forward_proj=config.feed_forward_proj,
                is_encoder_decoder=config.is_encoder_decoder,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.byte_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.byte_embeddings = new_embeddings

    def get_encoder(self):
        return self.encoder_decoder.encoder

    def get_decoder(self):
        return self.encoder_decoder.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder_decoder.encoder.layer[layer].attention.prune_heads(heads)
    
    def _compute_pooled_representations(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None and input_ids is None:
            return None
        
        byte_embeddings = inputs_embeds if inputs_embeds is not None else self.byte_embeddings(input_ids)
        
        frontier_predictions = self.frontier_predictor(byte_embeddings, attention_mask)
        
        pooled_representations, _, _, _ = self.pooler(frontier_predictions, byte_embeddings)

        return pooled_representations

    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import ByT5Tokenizer, MantaModel

        >>> tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        >>> model = MantaModel.from_pretrained("nthngdy/manta-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for MantaModel.
        >>> # This is not needed for torch's MantaForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_representations = self._compute_pooled_representations(
            input_ids,
            attention_mask,
            inputs_embeds
        )

        decoder_pooled_representations = self._compute_pooled_representations(
            decoder_input_ids,
            decoder_attention_mask,
            decoder_inputs_embeds
        )

        return self.encoder_decoder(
            inputs_embeds=pooled_representations,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_pooled_representations,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_outputs=encoder_outputs,
        )

@add_start_docstrings("""Manta Model with a `language modeling` head on top.""")
class MantaForConditionalGeneration(MantaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: MantaConfig):
        super().__init__(config)
        self.byte_embeddings = nn.Embedding(config.vocab_size, config.byte_embedding_dim)

        self.frontier_predictor = MantaFrontierPredictor(
            num_layers=config.frontier_predictor_num_layers,
            num_attention_heads=config.frontier_predictor_num_attention_heads,
            attention_window=config.frontier_predictor_attention_window,
            hidden_size=config.byte_embedding_dim,
            max_length=config.max_length_encoder_decoder,
            dropout_rate=config.dropout_rate,
        )

        self.pooler = MantaCachedConvolutionPooling(
            variance_regularization=config.pooling_variance_regularization,
            kernel_size=config.pooling_kernel_size,
            n_highway_layers=config.pooling_n_highway_layers,
            highway_activation=config.pooling_highway_activation,
            depthwise_convolution=config.pooling_depthwise_convolution,
            mean_pool=config.pooling_mean_pool,
            output_dim=config.d_model,
            padding_length=config.max_length_encoder_decoder,
            hidden_dim=config.byte_embedding_dim,
        )

        self.encoder_decoder = T5Model(
            T5Config(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_decoder_layers=config.num_decoder_layers,
                num_heads=config.num_heads,
                relative_attention_num_buckets=config.relative_attention_num_buckets,
                relative_attention_max_distance=config.relative_attention_max_distance,
                dropout_rate=config.dropout_rate,
                layer_norm_epsilon=config.layer_norm_epsilon,
                initializer_factor=config.initializer_factor,
                feed_forward_proj=config.feed_forward_proj,
                is_encoder_decoder=config.is_encoder_decoder,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.byte_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.byte_embeddings = new_embeddings

    def get_encoder(self):
        return self.encoder_decoder.encoder

    def get_decoder(self):
        return self.encoder_decoder.decoder
    
    def _compute_pooled_representations(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None and input_ids is None:
            return None
        
        byte_embeddings = inputs_embeds if inputs_embeds is not None else self.byte_embeddings(input_ids)
        
        frontier_predictions = self.frontier_predictor(byte_embeddings, attention_mask)
        
        pooled_representations, _, _, _ = self.pooler(frontier_predictions, byte_embeddings)

        return pooled_representations
    
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import ByT5Tokenizer, MantaForConditionalGeneration

        >>> tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        >>> model = MantaForConditionalGeneration.from_pretrained("nthngdy/manta-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:

            pooled_representations = self._compute_pooled_representations(
                input_ids,
                attention_mask,
                inputs_embeds
            )
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder_decoder.encoder(
                attention_mask=attention_mask,
                inputs_embeds=pooled_representations,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        

        # Decode
        decoder_outputs = self.encoder_decoder.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare Manta Model transformer outputting encoder's raw hidden-states without any specific head on top."
)
class MantaEncoderModel(MantaPreTrainedModel):
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.byte_embeddings = nn.Embedding(config.vocab_size, config.byte_embedding_dim)

        self.frontier_predictor = MantaFrontierPredictor(
            num_layers=config.frontier_predictor_num_layers,
            num_attention_heads=config.frontier_predictor_num_attention_heads,
            attention_window=config.frontier_predictor_attention_window,
            hidden_size=config.byte_embedding_dim,
            max_length=config.max_length_encoder_decoder,
            dropout_rate=config.dropout_rate,
        )

        self.pooler = MantaCachedConvolutionPooling(
            variance_regularization=config.pooling_variance_regularization,
            kernel_size=config.pooling_kernel_size,
            n_highway_layers=config.pooling_n_highway_layers,
            highway_activation=config.pooling_highway_activation,
            depthwise_convolution=config.pooling_depthwise_convolution,
            mean_pool=config.pooling_mean_pool,
            output_dim=config.d_model,
            padding_length=config.max_length_encoder_decoder,
            hidden_dim=config.byte_embedding_dim,
        )

        self.encoder = T5EncoderModel(
            T5Config(
                vocab_size=1,
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_decoder_layers=config.num_decoder_layers,
                num_heads=config.num_heads,
                relative_attention_num_buckets=config.relative_attention_num_buckets,
                relative_attention_max_distance=config.relative_attention_max_distance,
                dropout_rate=config.dropout_rate,
                layer_norm_epsilon=config.layer_norm_epsilon,
                initializer_factor=config.initializer_factor,
                feed_forward_proj=config.feed_forward_proj,
                is_encoder_decoder=config.is_encoder_decoder,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.byte_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.byte_embeddings = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder_decoder.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
    
    def _compute_pooled_representations(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None and input_ids is None:
            return None

        byte_embeddings = inputs_embeds if inputs_embeds is not None else self.byte_embeddings(input_ids)
        
        frontier_predictions = self.frontier_predictor(byte_embeddings, attention_mask)
        
        pooled_representations, _, _, _ = self.pooler(frontier_predictions, byte_embeddings)

        return pooled_representations

    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import ByT5Tokenizer, MantaEncoderModel

        >>> tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        >>> model = MantaEncoderModel.from_pretrained("nthngdy/manta-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_representations = self._compute_pooled_representations(
            input_ids,
            attention_mask,
            inputs_embeds
        )

        encoder_outputs = self.encoder(
            inputs_embeds=pooled_representations,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
