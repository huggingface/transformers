# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch UniTE model."""

from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional, Tuple, Union

from ...utils import add_start_docstrings, logging
from ...modeling_utils import PreTrainedModel
from ...tokenization_utils_base import BatchEncoding
from ..xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaModel,
)
from .configuration_unite import UniTEConfig

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import Parameter, ParameterList
from torch.nn.utils.rnn import pad_sequence

from math import ceil

logger = logging.get_logger(__name__)

UNITE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unite-up",
    "unite-mup",
]


UNITE_START_DOCSTRING = r"""
    UniTE model is build upon XLMRoBERTa, with additional pooling layer and feedforward netowrk to predict a scalar
    to describe the agreement of translation output and given segments.

    Given segments can be source input or target reference, or even both. You can do such three translation
    evaluation tasks with one single model.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UnITEConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class UniTEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UniTEConfig
    base_model_prefix = "unite"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
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

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class FeedForward(nn.Module):
    """
    Feed Forward Neural Network.

    :param in_dim: Number input features.
    :param out_dim: Number of output features. Default is just a score.
    :param hidden_sizes: List with hidden layer sizes.
    :param activations: Name of the activation function to be used in the hidden layers.
    :param final_activation: Name of the final activation function if any.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = [3072, 768],
        activations: str = "Sigmoid",
        final_activation: Optional[str] = None
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation):
            return getattr(nn, activation)()

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)


@add_start_docstrings(
    """
    UniTE Model transformer with a sequence regression head on top.
    """,
    UNITE_START_DOCSTRING,
)
class UniTEForSequenceClassification(UniTEPreTrainedModel):
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = UniTEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = XLMRobertaModel(self.config)
        self.layerwise_attention = LayerwiseAttention(
            num_layers=self.config.num_hidden_layers + 1,
            layer_norm=True
        )
        self.estimator = FeedForward(
            in_dim=self.config.hidden_size,
            out_dim=1,
            hidden_sizes=self.config.estimator_hidden_sizes,
            activations=self.config.estimator_activations,
            final_activation=self.config.estimator_final_activation,
        )

        self.pad_idx = self.config.pad_token_id
        self.bos_idx = self.config.bos_token_id
        self.eos_idx = self.config.eos_token_id

        return
    
    def forward(
        self,
        hyp: BatchEncoding,
        src: Union[None, BatchEncoding] = None,
        ref: Union[None, BatchEncoding] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if src is None and ref is None:
            raise ValueError('At least one segment among source and reference should be provided.')

        if src is not None and ref is not None:
            src['input_ids'][:, 0] = self.eos_idx
            ref['input_ids'][:, 0] = self.eos_idx
            input_ids = cut_long_sequences3([hyp['input_ids'].unbind(dim=0), src['input_ids'].unbind(dim=0), ref['input_ids'].unbind(dim=0)], self.config.max_len, pad_idx=self.pad_idx)
        elif src is not None:
            src['input_ids'][:, 0] = self.eos_idx
            input_ids = cut_long_sequences2([hyp['input_ids'].unbind(dim=0), src['input_ids'].unbind(dim=0)], self.config.max_len, pad_idx=self.pad_idx)
        else:
            ref['input_ids'][:, 0] = self.eos_idx
            input_ids = cut_long_sequences2([hyp['input_ids'].unbind(dim=0), ref['input_ids'].unbind(dim=0)], self.config.max_len, pad_idx=self.pad_idx)

        attention_mask = input_ids.ne(self.pad_idx).long()
        embedded_sequences = self.get_sentence_embedding(
            input_ids,
            attention_mask
        )

        return self.estimator(embedded_sequences).squeeze(dim=-1)


    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
            a single sentence.

        :param tokens: sequences [batch_size x seq_len]
        :param lengths: lengths [batch_size]
        :param 

        :return: torch.Tensor [batch_size x hidden_size]
        """
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if self.layerwise_attention: 
            # HACK: LayerNorm is applied at the MiniBatch. This means that for big batch sizes the variance
            # and norm within the batch will create small differences in the final score
            # If we are predicting we split the data into equal size batches to minimize this variance.
            if not self.training:
                n_splits = len(torch.split(encoder_out["hidden_states"][-1], 8))
                embeddings = []
                for split in range(n_splits):
                    all_layers = []
                    for layer in range(len(encoder_out["hidden_states"])):
                        layer_embs = torch.split(encoder_out["hidden_states"][layer], 8)
                        all_layers.append(layer_embs[split])
                    split_attn = torch.split(attention_mask, 8)[split]
                    embeddings.append(self.layerwise_attention(all_layers, split_attn))
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = self.layerwise_attention(
                    encoder_out["hidden_states"], attention_mask
                )

        sentemb = embeddings[:, 0, :]

        return sentemb


class LayerwiseAttention(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: Optional[List[int]] = None
    ) -> None:
        super(LayerwiseAttention, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm

        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        elif len(layer_weights) != num_layers:
            raise Exception(
                "Length of layer_weights {} differs \
                from num_layers {}".format(
                    layer_weights, num_layers
                )
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([layer_weights[i]]),
                    requires_grad=True,
                )
                for i in range(num_layers)
            ]
        )

        self.scale = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if len(tensors) != self.num_layers:
            raise Exception(
                "{} tensors were passed, but the module was initialized to \
                mix {} tensors.".format(
                    len(tensors), self.num_layers
                )
            )

        def _layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        weights = torch.cat([parameter for parameter in self.scalar_parameters])
        gamma = self.scale

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight
                    * _layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return gamma * sum(pieces)


def cut_long_sequences2(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(int(x.ne(pad_idx).sum(dim=-1)) for x in tensor_tuple)
        tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, all_lens))

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 2)

            if min(all_lens) > (maximum_length // 2) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                lengths[lengths_sorted_idxes[0]] = maximum_length - lengths[lengths_sorted_idxes[1]]

            # new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, list(v for k, v in lengths.items())))
            new_lens = list(lengths[k] for k in range(0, len(tensor_tuple)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            print('Data length: %d -> %d (%s -> %s).' % (sum(all_lens), sum(lengths.values()), list(all_lens), new_lens))
        else:
            collected_tuples.append(tensor_tuple)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)

    return all_input_concat_padded


def cut_long_sequences3(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(int(x.ne(pad_idx).sum(dim=-1)) for x in tensor_tuple)
        tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, all_lens))

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 3)

            if min(all_lens) > (maximum_length // 3) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                while sum(lengths.values()) > maximum_length:
                    if lengths[lengths_sorted_idxes[0]] > lengths[lengths_sorted_idxes[1]]:
                        offset = maximum_length - lengths[lengths_sorted_idxes[1]] - lengths[lengths_sorted_idxes[2]]
                        if offset > lengths[lengths_sorted_idxes[1]]:
                            lengths[lengths_sorted_idxes[0]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]]
                    elif lengths[lengths_sorted_idxes[0]] == lengths[lengths_sorted_idxes[1]] > lengths[lengths_sorted_idxes[2]]:
                        offset = (maximum_length - lengths[lengths_sorted_idxes[2]]) // 2
                        if offset > lengths[lengths_sorted_idxes[2]]:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]]
                    else:
                        lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]] = maximum_length // 3

            # new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, list(v for k, v in lengths.items())))
            new_lens = list(lengths[k] for k in range(0, len(lengths)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            print('Data length: %d -> %d (%s -> %s).' % (sum(all_lens), sum(lengths.values()), list(all_lens), list(new_lens)))
        else:
            collected_tuples.append(tensor_tuple)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)

    return all_input_concat_padded
