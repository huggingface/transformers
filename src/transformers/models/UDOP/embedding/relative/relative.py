# Code taken from Due Benchmark https://github.com/due-benchmark/baselines

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch import nn as nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention

from core.models.udop_config import UdopConfig

# get function for bucket computation
# protected member access seems to be lesser evil than copy paste whole function
get_relative_position_bucket = T5Attention._relative_position_bucket
AUGMENTATION_RANGE = (0.80, 1.25)


class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases
    :param num_heads: number of heads in lm model, it will create embeddings of size `num_heads`,
        which will be added to scores per each token pair
    :param relative_attention_num_buckets: pair token metric
        (distance in the sequence, distance in pixels etc.) will be bucketed,
        parameter is defining number of such buckets
    :param bidirectional: defining if for pair of tokens distance should be bidirecional,
        if bidirectional=False, then distance(tok1, tok2) == distance(tok2, tok1)
    :param scaling_factor: defining factor which will be used to scale relative distance
    :param max_distance: all distances above this value will end up in the one/same bucket
    :param augmentation: whether to multiple relative distances by random scalar
    :param expand: used for re-using pretrained model with subsequent addition of prefix_bucket
    """

    def __init__(self, num_heads=None, relative_attention_num_buckets=32,
                 bidirectional=True, scaling_factor=1, max_distance=128,
                 level="tokens", augmentation=False, prefix_bucket=False, expand=False):

        super(RelativePositionBiasBase, self).__init__()
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and not self.expand else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    def get_required_segment_levels(self) -> Sequence[str]:
        return [self.level]

    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Optional[Tensor] = None,
        seg_data: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        pass

    def get_bucket(self, attention_mask: Optional[Tensor] = None,
                   seg_data: Optional[Dict[str, Any]] = None) -> Tensor:
        relative_position = self.prepare_input(attention_mask, seg_data)
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)

    def forward(self, attention_mask: Optional[Tensor] = None,
                seg_data: Optional[Dict[str, Any]] = None) -> Tensor:

        # re-using pretrained model with subsequent addition of prefix_bucket
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
            new_bias.weight.data[:self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets:] = 0.1
            self.relative_attention_bias = new_bias
            self.expand = False
 
        rp_bucket = self.get_bucket(attention_mask, seg_data)

        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
                rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
            # based on assumption that prefix bboxes are negative
            is_prefix = seg_data[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

        values: Tensor = self.relative_attention_bias(rp_bucket)
        assert values.dim() == 4, "Wrong dimension of values tensor"
        values = values.permute([0, 3, 1, 2])

        return values


class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is
        their distance in the sequence. Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None,
                      seg_data: Optional[Dict[str, Any]] = None) -> Tensor:
        assert self.scaling_factor == 1, "No need to scale 1d features"
        relative_position = self.get_relative_position(torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)[None, :])

        return relative_position


def expand_feature(token_map, feature, special_tokens_value=0):
    token_map = token_map.clone()
    # add values for special tokens
    feature_all = torch.cat([feature, torch.full_like(feature[:, 0:1],
                                                      fill_value=special_tokens_value)],
                            dim=1)
    if feature.dim() == 3:
        bs, seg_len, features_dim = feature.shape
        token_map[token_map == -1] = seg_len
        expand_index = token_map[:, :, None].expand(-1, -1, features_dim).to(torch.long)

    elif feature.dim() == 2:
        bs, seg_len = feature.shape
        token_map[token_map == -1] = seg_len
        expand_index = token_map.to(torch.long)
    else:
        raise AttributeError("Wrong dimension of input feature tensor")

    expanded_feature = torch.gather(feature_all, 1, expand_index)

    return expanded_feature


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None,
                      seg_data: Optional[Dict[str, Any]] = None) -> Tensor:
        assert self.scaling_factor > 1.0, \
            "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get x positions of left point of bbox
        assert seg_data is not None
        horizontal_position: Tensor = seg_data[:, :, [0, 2]].mean(dim=-1)

        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None,
                      seg_data: Optional[Dict[str, Any]] = None) -> Tensor:
        assert self.scaling_factor > 1.0, \
            "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get y positions of middle of bbox
        assert seg_data is not None
        vertical_position: Tensor = seg_data[:, :, [1, 3]].mean(dim=-1)

        return self.get_relative_position(vertical_position)


class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class will sums up computed biases
        :param modules: list of relative bias modules
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(
        self,
        attention_mask: Optional[Tensor] = None,
        seg_data: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Tensor]:
        x = 0.0
        for bias in self.biases:  # type: ignore
            x = bias(attention_mask, seg_data) + x

        return x


BIAS_CLASSES = {"1d": RelativePositionBias1D,
                "horizontal": RelativePositionBiasHorizontal,
                "vertical": RelativePositionBiasVertical,
                }


def create_relative_bias(config: Union[UdopConfig, T5Config]) -> Sequence[RelativePositionBiasBase]:
    """
    Creates empty list or one/multiple relative biases.

    :param config: Model's configuration
    :return: Sequence with created bias modules.
    """
    bias_list = []
    if hasattr(config, "relative_bias_args"):
        assert isinstance(config.relative_bias_args, list)
        for bias_kwargs_org in config.relative_bias_args:
            bias_kwargs = deepcopy(bias_kwargs_org)
            bias_type = bias_kwargs.pop("type")
            model_num_heads = config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
            if "num_heads" in bias_kwargs:
                assert (
                    bias_kwargs["num_heads"] == model_num_heads
                ), "Number of heads must match num of heads in the model"
            else:
                bias_kwargs["num_heads"] = model_num_heads
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))  # type: ignore

    return bias_list
