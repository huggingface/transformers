# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Provides utility to combine a vision backbone with a language backbone."""

from copy import copy
from typing import List, Optional

import torch
import torch.nn as nn

from torch.nn.attention import sdpa_kernel, SDPBackend

from .act_ckpt_utils import activation_ckpt_wrapper
from .model_misc import NestedTensor


class NonFusionVLBackbone(nn.Module):
    """A vision-language backbone that does not fuse the vision and language"""

    def __init__(
        self,
        visual,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        """Initialize the backbone combiner.

        :param visual: The vision backbone to use
        :param text: The text encoder to use
        """
        super().__init__()
        self.vision_backbone = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        # allow running activation checkpointing on the entire vision and language backbones
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: NestedTensor,
        captions: List[str],
        input_boxes: Optional[torch.Tensor] = None,
        additional_text: Optional[List[str]] = None,
    ):
        """Forward pass of the backbone combiner.

        :param samples: The input images
        :param captions: The input captions
        :param input_boxes: If the text contains place-holders for boxes, this
            parameter contains the tensor containing their spatial features
        :param additional_text: This can be used to encode some additional text
            (different from the captions) in the same forward of the backbone
        :return: Output dictionary with the following keys:
            - vision_features: The output of the vision backbone
            - language_features: The output of the language backbone
            - vision_mask: The attention mask of the vision backbone
            - language_mask: The attention mask of the language backbone
            - vision_pos_enc: The positional encoding of the vision backbone
            - (optional) additional_text_features: The output of the language
                backbone for the additional text
            - (optional) additional_text_mask: The attention mask of the
                language backbone for the additional text
        """
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(self.forward_text(captions, input_boxes, additional_text, device))
        return output

    def forward_image(self, samples: NestedTensor):
        return activation_ckpt_wrapper(self._forward_image_no_act_ckpt)(
            samples=samples,
            act_ckpt_enable=self.act_ckpt_whole_vision_backbone and self.training,
        )

    def _forward_image_no_act_ckpt(self, samples):
        # Forward through backbone
        features, pos = self.vision_backbone(samples)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        src, mask = features[-1].decompose()

        output = {
            "vision_features": src,
            "vision_mask": mask,
            "vision_pos_enc": pos,
            "backbone_fpn": features,  # Temporary, to not break anything else
        }
        return output

    def forward_text(
        self, captions, input_boxes=None, additional_text=None, device="cuda"
    ):
        return activation_ckpt_wrapper(self._forward_text_no_ack_ckpt)(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
            act_ckpt_enable=self.act_ckpt_whole_language_backbone and self.training,
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ):
        output = {}

        # Forward through text_encoder
        text_to_encode = copy(captions)
        if additional_text is not None:
            # if there are additional_text, we piggy-back them into this forward.
            # They'll be used later for output alignment
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )

        return output
