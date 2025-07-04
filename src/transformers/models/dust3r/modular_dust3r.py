# coding=utf-8
# Copyright 2025 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dust3R model using modular approach with ViT inheritance."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...utils import logging
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEmbeddings,
    ViTEncoder,
    ViTIntermediate,
    ViTLayer,
    ViTOutput,
    ViTPooler,
    ViTPreTrainedModel,
    ViTSelfAttention,
    ViTSelfOutput,
)
from .configuration_dust3r import Dust3RConfig


logger = logging.get_logger(__name__)

try:
    from .third_party import RoPE2D  # type: ignore
except (ImportError, ModuleNotFoundError):

    class RoPE2D:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs) if args else None


class Dust3REmbeddings(ViTEmbeddings):
    def __init__(self, config: Dust3RConfig, use_mask_token: bool = False) -> None:
        super().__init__(config, use_mask_token=use_mask_token)


class Dust3REncoder(ViTEncoder):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)
        self.layer = nn.ModuleList([Dust3RLayer(config) for _ in range(config.num_hidden_layers)])


class Dust3RDecoder(nn.Module):
    def __init__(self, config: Dust3RConfig):
        super().__init__()
        self.config = config

        self.decoder_embed = nn.Linear(config.hidden_size, config.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pos_encoding = RoPE2D() if RoPE2D else None

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Cross-attention decoder following the Dust3R architecture.

        Args:
            feat1: Features from first image [B, N, D]
            feat2: Features from second image [B, N, D]

        Returns:
            Tuple of decoder features at different layers (d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
        """
        dec1 = self.decoder_embed(feat1)
        dec2 = self.decoder_embed(feat2)

        d1_outputs = [dec1]
        d2_outputs = [dec2]

        for i, layer in enumerate(self.decoder_layers.layers):
            dec1_new = layer(dec1, dec2)
            dec2_new = layer(dec2, dec1)

            dec1, dec2 = dec1_new, dec2_new

            if i + 1 in [6, 9, 12]:
                d1_outputs.append(self.norm(dec1))
                d2_outputs.append(self.norm(dec2))

        while len(d1_outputs) < 4:
            d1_outputs.append(self.norm(dec1))
        while len(d2_outputs) < 4:
            d2_outputs.append(self.norm(dec2))

        return tuple(d1_outputs + d2_outputs)


class Dust3RHead(nn.Module):
    """
    Dust3R head for outputting pointmaps and confidence maps.
    """

    def __init__(self, config: Dust3RConfig):
        super().__init__()
        self.config = config

        self.pointmap_heads = nn.ModuleList([nn.Linear(config.hidden_size, 3) for _ in range(4)])

        self.confidence_heads = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(4)])

        # Optional projection layers
        self.proj_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                )
                for _ in range(4)
            ]
        )

    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12) -> tuple[torch.Tensor, ...]:
        """
        Apply heads to decoder outputs.

        Args:
            d1_0, d1_6, d1_9, d1_12: Decoder outputs for image 1
            d2_0, d2_6, d2_9, d2_12: Decoder outputs for image 2

        Returns:
            pt1, cf1, pt2, cf2: Pointmaps and confidence maps for both images
        """
        d1_features = [d1_0, d1_6, d1_9, d1_12]
        d2_features = [d2_0, d2_6, d2_9, d2_12]

        d1_combined = sum(self.proj_layers[i](feat) for i, feat in enumerate(d1_features)) / len(d1_features)
        d2_combined = sum(self.proj_layers[i](feat) for i, feat in enumerate(d2_features)) / len(d2_features)

        pt1 = self.pointmap_heads[-1](d1_combined)
        cf1 = self.confidence_heads[-1](d1_combined)
        pt2 = self.pointmap_heads[-1](d2_combined)
        cf2 = self.confidence_heads[-1](d2_combined)

        return pt1, cf1, pt2, cf2


class Dust3RSelfAttention(ViTSelfAttention):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)


class Dust3RSelfOutput(ViTSelfOutput):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)


class Dust3RAttention(ViTAttention):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)
        self.attention = Dust3RSelfAttention(config)
        self.output = Dust3RSelfOutput(config)


class Dust3RIntermediate(ViTIntermediate):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)


class Dust3ROutput(ViTOutput):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)


class Dust3RLayer(ViTLayer):
    def __init__(self, config: Dust3RConfig):
        super().__init__(config)
        self.attention = Dust3RAttention(config)
        self.intermediate = Dust3RIntermediate(config)
        self.output = Dust3ROutput(config)


class Dust3RPooler(ViTPooler):
    def __init__(self, config: Dust3RConfig):
        super().__init__(config)


class Dust3RPreTrainedModel(ViTPreTrainedModel):
    config_class = Dust3RConfig
    base_model_prefix = "dust3r"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dust3REmbeddings", "Dust3RLayer", "Dust3RAttention"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                module.in_proj_weight.data = nn.init.trunc_normal_(
                    module.in_proj_weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
                ).to(module.in_proj_weight.dtype)
            if hasattr(module, "in_proj_bias") and module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if hasattr(module, "out_proj") and hasattr(module.out_proj, "weight"):
                module.out_proj.weight.data = nn.init.trunc_normal_(
                    module.out_proj.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
                ).to(module.out_proj.weight.dtype)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()
        elif isinstance(module, Dust3REmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            if module.mask_token is not None:
                module.mask_token.data.zero_()


class Dust3RModel(Dust3RPreTrainedModel):
    def __init__(self, config: Dust3RConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = Dust3REmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = Dust3REncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = Dust3RPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        """
        Forward pass for Dust3R model.

        Args:
            pixel_values: First image tensor [B, C, H, W]
            pixel_values_2: Second image tensor [B, C, H, W]
            bool_masked_pos: Optional masked positions
            head_mask: Optional attention head mask
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            interpolate_pos_encoding: Whether to interpolate position encodings
            return_dict: Whether to return ModelOutput
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output,)
            if pooled_output is not None:
                outputs += (pooled_output,)
            if output_hidden_states:
                hidden_states = (
                    encoder_outputs.hidden_states
                    if hasattr(encoder_outputs, "hidden_states")
                    else (encoder_outputs[1] if len(encoder_outputs) > 1 else None)
                )
                if hidden_states is not None:
                    outputs += (hidden_states,)
            if output_attentions:
                attentions = (
                    encoder_outputs.attentions
                    if hasattr(encoder_outputs, "attentions")
                    else (encoder_outputs[2] if len(encoder_outputs) > 2 else None)
                )
                if attentions is not None:
                    outputs += (attentions,)
            return outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


class Dust3RForMaskedImageModeling(Dust3RPreTrainedModel):
    """
    Dust3R Model with a decoder on top for masked image modeling.
    """

    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)

        self.embeddings = Dust3REmbeddings(config, use_mask_token=True)
        self.encoder = Dust3REncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError("When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride`.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,)
            if output_hidden_states:
                hidden_states = (
                    encoder_outputs.hidden_states
                    if hasattr(encoder_outputs, "hidden_states")
                    else (encoder_outputs[1] if len(encoder_outputs) > 1 else None)
                )
                if hidden_states is not None:
                    output += (hidden_states,)
            if output_attentions:
                attentions = (
                    encoder_outputs.attentions
                    if hasattr(encoder_outputs, "attentions")
                    else (encoder_outputs[2] if len(encoder_outputs) > 2 else None)
                )
                if attentions is not None:
                    output += (attentions,)
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


class Dust3RForImageClassification(Dust3RPreTrainedModel):
    def __init__(self, config: Dust3RConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels

        # For classification, we only need the encoder part, not decoder/head
        self.embeddings = Dust3REmbeddings(config, use_mask_token=False)
        self.encoder = Dust3REncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=None, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = (
                    loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
                )
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                hidden_states = (
                    encoder_outputs.hidden_states
                    if hasattr(encoder_outputs, "hidden_states")
                    else (encoder_outputs[1] if len(encoder_outputs) > 1 else None)
                )
                if hidden_states is not None:
                    output += (hidden_states,)
            if output_attentions:
                attentions = (
                    encoder_outputs.attentions
                    if hasattr(encoder_outputs, "attentions")
                    else (encoder_outputs[2] if len(encoder_outputs) > 2 else None)
                )
                if attentions is not None:
                    output += (attentions,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


__all__ = ["Dust3RForImageClassification", "Dust3RForMaskedImageModeling", "Dust3RModel", "Dust3RPreTrainedModel"]
