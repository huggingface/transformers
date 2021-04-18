# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch PiT model. """


import collections.abc
import math
from collections import namedtuple

import torch
import torch.utils.checkpoint
from numpy.core.shape_base import block
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_pit import PiTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PiTConfig"

PIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "navar-ai/pit-tiny-path16-224",
    # See all PiT models at https://huggingface.co/models?filter=pit
]


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


PiTEncoderBlockOutput = namedtuple(
    "PiTEncoderBlockOutput", field_names=["last_hidden_state", "cls_tokens", "hidden_states", "attentions"]
)
PiTModelOutput = namedtuple(
    "PiTModelOutput",
    field_names=[
        "last_cls_token_states",
        "last_hidden_state",
        "hidden_states",
        "pooled_states",
        "cls_token_states",
        "attentions",
    ],
)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/pit.py
class PiTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        width = math.floor((config.image_size + 2 - config.patch_size) / config.stride + 1)
        self.embed_dim = config.base_dims[0] * config.heads[0]
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=self.embed_dim,
            stride=config.stride,
        )
        self.position_embeddings = nn.Parameter(torch.randn(1, self.embed_dim, width, width), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768, stride=8):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        image_size = to_2tuple(image_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.size()
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values)
        return x


class PiTSelfAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    # Copied from transformers.models.vit.modeling_vit.ViTSelfAttention.forward
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->PiT
class PiTSelfOutput(nn.Module):
    """
    The residual connection is defined in PiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states



class PiTAttention(nn.Module):
    def __init__(self, config, hidden_size, num_heads):
        super().__init__()
        self.attention = PiTSelfAttention(config, hidden_size, num_heads)
        self.output = PiTSelfOutput(config, hidden_size)
        self.pruned_heads = set()

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.forward
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PiTIntermediate(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # Copied from transformers.models.vit.modeling_vit.ViTIntermediate.forward
    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states



class PiTOutput(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # Copied from transformers.models.vit.modeling_vit.ViTOutput.forward
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states



class PiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, block_id=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        hidden_size = config.base_dims[block_id] * config.heads[block_id]
        num_heads = config.heads[block_id]
        feed_forward_dim = hidden_size * 4

        self.attention = PiTAttention(config, hidden_size, num_heads)
        self.intermediate = PiTIntermediate(config, hidden_size, feed_forward_dim)
        self.output = PiTOutput(config, hidden_size, feed_forward_dim)

        self.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    # Copied from transformers.models.vit.modeling_vit.ViTLayer.forward with ViT->PiT
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in PiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in PiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        # TODO feedforward chunking not working for now
        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layer_output
        # )

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->PiT
class PiTEncoderBlock(nn.Module):
    def __init__(self, config, pool=None, block_id=0):
        super().__init__()
        self.config = config
        num_hidden_layers = config.depths[block_id]
        self.layer = nn.ModuleList([PiTLayer(config, block_id) for _ in range(num_hidden_layers)])
        self.pool = pool

    def forward(
        self,
        hidden_states,
        cls_tokens,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size, channels, height, width = hidden_states.shape
        tokens_length = cls_tokens.shape[1]

        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (batch, height*width, channel)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        cls_tokens = hidden_states[:, :tokens_length]
        hidden_states = hidden_states[:, tokens_length:]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)

        # pool the outputs
        if self.pool is not None:
            hidden_states, cls_tokens = self.pool(hidden_states, cls_tokens)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, cls_tokens, all_hidden_states, all_self_attentions] if v is not None
            )

        return PiTEncoderBlockOutput(
            last_hidden_state=hidden_states,
            cls_tokens=cls_tokens,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PiTConvHeadPooling(nn.Module):
    def __init__(self, config, block_id=0):
        super(PiTConvHeadPooling, self).__init__()
        current_hidden_dim = config.base_dims[block_id] * config.heads[block_id]
        next_hidden_size = config.base_dims[block_id + 1] * config.heads[block_id + 1]

        stride = config.conv_pooling_stride
        self.conv = nn.Conv2d(
            current_hidden_dim,
            next_hidden_size,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            groups=current_hidden_dim,
        )
        self.fc = nn.Linear(current_hidden_dim, next_hidden_size)

    def forward(self, hidden_states, cls_token):

        hidden_states = self.conv(hidden_states)
        cls_token = self.fc(cls_token)

        return hidden_states, cls_token


class PiTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depths = config.depths
        self.heads = config.heads
        self.base_dims = config.base_dims

        self.heads = config.heads
        self.base_dims = config.base_dims

        initia_embed_dim = self.base_dims[0] * self.heads[0]
        self.cls_token = nn.Parameter(torch.randn(1, 1, initia_embed_dim), requires_grad=True)

        self.stages = nn.ModuleList()
        for stage in range(len(self.depths)):
            pool = None
            if stage < len(self.depths) - 1:
                pool = PiTConvHeadPooling(config, block_id=stage)

            self.stages.append(PiTEncoderBlock(config, pool, block_id=stage))

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_pooled_states = () if output_hidden_states else None
        all_cls_token_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        cls_tokens = self.cls_token.expand(hidden_states.shape[0], -1, -1)

        for stage in self.stages:
            outputs = stage(
                hidden_states,
                cls_tokens,
                head_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

            hidden_states, cls_tokens = outputs[:2]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (outputs[2],)
                all_pooled_states = all_pooled_states + (hidden_states,)
                all_cls_token_states = all_cls_token_states + (cls_tokens,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[3],)

        if not return_dict:
            return (cls_tokens,)
        return PiTModelOutput(
            last_cls_token_states=cls_tokens,
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            pooled_states=all_pooled_states,
            cls_token_states=all_cls_token_states,
            attentions=all_self_attentions,
        )


class PiTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PiTConfig
    base_model_prefix = "pit"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PIT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.PiTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

PIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.PiTTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare PiT Model transformer outputting raw hidden-states without any specific head on top.",
    PIT_START_DOCSTRING,
)
class PiTModel(PiTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.heads = config.heads
        self.base_dims = config.base_dims

        self.embeddings = PiTEmbeddings(config)
        self.encoder = PiTEncoder(config)

        final_embed_dim = self.base_dims[-1] * self.heads[-1]
        self.layernorm = nn.LayerNorm(final_embed_dim, eps=config.layer_norm_eps)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PIT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import PiTFeatureExtractor, PiTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = PiTFeatureExtractor.from_pretrained('google/PiT-base-patch16-224-in21k')
            >>> model = PiTModel.from_pretrained('google/PiT-base-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = None

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    PIT_START_DOCSTRING,
)
class PiTForImageClassification(PiTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pit = PiTModel(config)

        self.final_hidden_size = config.base_dims[-1] * config.heads[-1]
        # Classifier head
        self.classifier = (
            nn.Linear(self.final_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        self.init_weights()

    @add_start_docstrings_to_model_forward(PIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            >>> model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.pit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
