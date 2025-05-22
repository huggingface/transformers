# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
"""PyTorch EoMT model."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    replace_return_docstrings,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_eomt import EoMTConfig


if is_timm_available():
    from timm import create_model
    from timm.layers import LayerNorm2d


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "EoMTConfig"
_CHECKPOINT_FOR_DOC = "tue-mps/coco_panoptic_eomt_large_640"


@dataclass
# Adapted from transformers.models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput with Mask2Former->EoMT,Mask2FormerImageProcessor->Mask2FormerImageProcessor
class EoMTOutput(ModelOutput):
    """
    Class for outputs of [`EoMTModel`, `EoMTForUniversalSegmentationOutput`].

    This output can be directly passed to [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~Mask2FormerImageProcessor] for details regarding usage.

    Args:
        last_hidden_state (`torch.FloatTensor`):
            Last hidden state of the tokens of the last stage of the encoder model (backbone) of shape
            `(batch_size, num_tokens, embed_dim).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one before the blocks that process queries + one for the output of each block
            that processes queries) of shape `(batch_size, num_tokens, embed_dim)`. Returned when
            `output_hidden_states=True`.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    class_queries_logits: Optional[torch.FloatTensor] = None
    masks_queries_logits: Optional[torch.FloatTensor] = None


class EoMTScaleBlock(nn.Module):
    """
    A scale block for upsampling feature maps.

    Uses a transposed convolution, GELU activation, a depthwise convolution, and layer normalization.

    Args:
        embed_dim (int): Dimension of the embedding space.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for EoMTScaleBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim, height, width).

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x


class EoMTEncoder(nn.Module):
    """
    Encoder module that wraps a timm backbone.

    Args:
        config (EoMTConfig): Configuration for the EoMT model.
    """

    def __init__(self, config: EoMTConfig):
        super().__init__()
        self.backbone = create_model(
            config.backbone,
            pretrained=config.use_pretrained_backbone,
            **config.backbone_kwargs,
        )


class EoMTNetwork(nn.Module):
    config_class = EoMTConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EoMTConfig):
        super().__init__()
        self.config = config
        self.encoder = EoMTEncoder(config)
        self.num_queries = config.num_queries
        self.num_blocks = config.num_blocks

        self.q = nn.Embedding(config.num_queries, self.encoder.backbone.embed_dim)
        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, config.num_labels + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        num_upscale = max(1, int(math.log2(config.patch_size)) - 2)
        self.upscale = nn.Sequential(*[EoMTScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)])

    def _predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Predict the mask and class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_queries, embed_dim).
        Returns:
            Tensor: Mask logits of shape (batch_size, num_queries, height, width).
            Tensor: Class logits of shape (batch_size, num_queries, num_labels + 1).
        """
        q = x[:, : self.num_queries, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_queries + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))

        return class_logits, mask_logits

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> EoMTOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.encoder.backbone.patch_embed(pixel_values)
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.blocks[: -self.num_blocks](x)
        x = torch.cat((self.q.weight[None].expand(x.size(0), -1, -1), x), dim=1)

        hidden_states = () if output_hidden_states else None
        class_queries_logits = ()
        masks_queries_logits = ()

        for block in self.encoder.backbone.blocks[-self.num_blocks :]:
            if output_hidden_states:
                hidden_states += (x,)

            x = block(x)

        class_queries_logits, masks_queries_logits = self._predict(self.encoder.backbone.norm(x))
        output = EoMTOutput(
            last_hidden_state=x,
            hidden_states=hidden_states,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
        )

        if not return_dict:
            return tuple(v for v in output.values() if v is not None)
        return output


EOMT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EoMTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EOMT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.preprocess`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~EoMTModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The EoMT Model for instance, semantic and panoptic image segmentation.",
    EOMT_START_DOCSTRING,
)
class EoMTForUniversalSegmentation(PreTrainedModel):
    config_class = EoMTConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.network = EoMTNetwork(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(EOMT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=EoMTOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> EoMTOutput:
        r"""
        Returns:
            `EoMTOutput`

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoImageProcessor, EoMTModel

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load image preprocessor and EoMTModel trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
        >>> model = EoMTModel.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # model outputs last hidden state of shape (batch_size, num_queries, embed_dim)
        >>> print(outputs.last_hidden_state.shape)
        torch.Size([1, 200, 1024])
        ```
        """
        return self.network(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


__all__ = ["EoMTForUniversalSegmentation"]
