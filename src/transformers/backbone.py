# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch.nn as nn


@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor. It is often used as the auxiliary
    inputs/outputs of models, to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a specific integer. This is typically true
        for encoder / decoder type networks with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific input size divisibility is required.
        """
        return 0

    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        This property is a generalization of size_divisibility. Some backbones and training recipes require specific
        padding constraints, such as enforcing divisibility by a specific integer (e.g., FPN) or padding to a square
        (e.g., ViTDet with large-scale jitter in :paper:vitdet). `padding_constraints` contains these optional items
        like: {
            "size_divisibility": int, "square_size": int, # Future options are possible
        } `size_divisibility` will read from here if presented and `square_size` indicates the square padding size if
        `square_size` > 0.

        TODO: use type of Dict[str, int] to avoid torchscipt issues. The type of padding_constraints could be
        generalized as TypedDict (Python 3.8+) to support more types in the future.
        """
        return {}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }
