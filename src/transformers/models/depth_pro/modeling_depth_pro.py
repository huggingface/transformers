# coding=utf-8
# Copyright 2024 The Apple Research Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch DepthPro model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ..auto import AutoModel
from .configuration_depth_pro import DepthProConfig


logger = logging.get_logger(__name__)


@dataclass
class DepthProOutput(ModelOutput):
    """
    Base class for DepthPro's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        features (`Union[torch.FloatTensor, List[torch.FloatTensor]]`, *optional*):
            Features from encoders. Can be a single feature or a list of features.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer and the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, n_patches_per_batch, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    features: Union[torch.FloatTensor, List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DepthProDepthEstimatorOutput(ModelOutput):
    """
    Base class for DepthProForDepthEstimation's output.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Predicted depth for each pixel.
        field_of_view (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned when `use_fov_model` is provided):
            Field of View Scaler.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer and the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, n_patches_per_batch, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_depth: torch.FloatTensor = None
    field_of_view: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def split_to_patches(pixel_values: torch.Tensor, patch_size: int, overlap_ratio: float) -> torch.Tensor:
    """Creates Patches from Batch."""
    batch_size, num_channels, height, width = pixel_values.shape

    if height == width == patch_size:
        # create patches only if scaled image is not already equal to patch size
        return pixel_values

    stride = torch_int(patch_size * (1 - overlap_ratio))

    patches = F.unfold(pixel_values, kernel_size=(patch_size, patch_size), stride=(stride, stride))
    patches = patches.permute(2, 0, 1)
    patches = patches.reshape(-1, num_channels, patch_size, patch_size)

    return patches


def reshape_features(hidden_states: torch.Tensor) -> torch.Tensor:
    """Discard class token and reshape 1D feature map to a 2D grid."""
    n_samples, seq_len, hidden_size = hidden_states.shape
    size = torch_int(seq_len**0.5)

    hidden_states = hidden_states[:, -(size**2) :, :]  # remove special tokens if there are any
    hidden_states = hidden_states.reshape(n_samples, size, size, hidden_size)
    hidden_states = hidden_states.permute(0, 3, 1, 2)

    return hidden_states


def merge_patches(patches: torch.Tensor, batch_size: int, padding: int) -> torch.Tensor:
    """Merges smaller patches into image-like feature map."""
    n_patches, hidden_size, out_size, out_size = patches.shape
    n_patches_per_batch = n_patches // batch_size
    sqrt_n_patches_per_batch = torch_int(n_patches_per_batch**0.5)
    new_out_size = sqrt_n_patches_per_batch * out_size

    if n_patches == batch_size:
        # merge only if the patches were created from scaled image
        # patches are not created when scaled image size is equal to patch size
        return patches

    if n_patches_per_batch < 4:
        # for each batch, atleast 4 small patches are required to
        # recreate a large square patch from merging them and later padding is applied
        # 3 x (8x8) patches becomes 1 x ( 8x8 ) patch (extra patch ignored, no padding)
        # 4 x (8x8) patches becomes 1 x (16x16) patch (padding later)
        # 5 x (8x8) patches becomes 1 x (16x16) patch (extra patch ignored, padding later)
        # 9 x (8x8) patches becomes 1 x (24x24) patch (padding later)
        # thus the following code only rearranges the patches and removes extra ones
        padding = 0

    # make sure padding is not large enough to remove more than half of the patch
    padding = min(out_size // 4, padding)

    if padding == 0:
        # faster when no padding is required
        merged = patches.reshape(n_patches_per_batch, batch_size, hidden_size, out_size, out_size)
        merged = merged.permute(1, 2, 0, 3, 4)
        merged = merged[:, :, : sqrt_n_patches_per_batch**2, :, :]
        merged = merged.reshape(
            batch_size, hidden_size, sqrt_n_patches_per_batch, sqrt_n_patches_per_batch, out_size, out_size
        )
        merged = merged.permute(0, 1, 2, 4, 3, 5)
        merged = merged.reshape(batch_size, hidden_size, new_out_size, new_out_size)
    else:
        # padding example:
        # let out_size = 8, new_out_size = 32, padding = 2
        # each patch is separated by "|"
        # and padding is applied to the merging edges of each patch
        # 00 01 02 03 04 05 06 07 | 08 09 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
        # 00 01 02 03 04 05 -- -- | -- -- 10 11 12 13 -- -- | -- -- 18 19 20 21 -- -- | -- -- 26 27 28 29 30 31
        i = 0
        boxes = []
        for h in range(sqrt_n_patches_per_batch):
            boxes_in_row = []
            for w in range(sqrt_n_patches_per_batch):
                box = patches[batch_size * i : batch_size * (i + 1)]

                # collect paddings
                paddings = [0, 0, 0, 0]
                if h != 0:
                    # remove pad from height if box is not at top border
                    paddings[0] = padding
                if w != 0:
                    # remove pad from width if box is not at left border
                    paddings[2] = padding
                if h != sqrt_n_patches_per_batch - 1:
                    # remove pad from height if box is not at bottom border
                    paddings[1] = padding
                if w != sqrt_n_patches_per_batch - 1:
                    # remove pad from width if box is not at right border
                    paddings[3] = padding

                # remove paddings
                _, _, box_h, box_w = box.shape
                pad_top, pad_bottom, pad_left, pad_right = paddings
                box = box[:, :, pad_top : box_h - pad_bottom, pad_left : box_w - pad_right]

                boxes_in_row.append(box)
                i += 1
            boxes_in_row = torch.cat(boxes_in_row, dim=-1)
            boxes.append(boxes_in_row)
        merged = torch.cat(boxes, dim=-2)

    return merged


def reconstruct_feature_maps(
    hidden_state: torch.Tensor, batch_size: int, padding: int, output_size: Tuple[float, float]
) -> torch.Tensor:
    """
    Reconstructs feature maps from the hidden state produced by any of the encoder. Converts the hidden state of shape
    `(n_patches_per_batch * batch_size, seq_len, hidden_size)` to feature maps of shape
    `(batch_size, hidden_size, output_size[0], output_size[1])`.

    Args:
        hidden_state (torch.Tensor): Input tensor of shape `(n_patches_per_batch * batch_size, seq_len, hidden_size)`
            representing the encoded patches.
        batch_size (int): The number of samples in a batch.
        padding (int): The amount of padding to be removed when merging patches.
        output_size (Tuple[float, float]): The desired output size for the feature maps, specified as `(height, width)`.

    Returns:
        torch.Tensor: Reconstructed feature maps of shape `(batch_size, hidden_size, output_size[0], output_size[1])`.
    """
    # reshape back to image like
    features = reshape_features(hidden_state)

    # merge all patches in a batch to create one large patch per batch
    features = merge_patches(
        features,
        batch_size=batch_size,
        padding=padding,
    )

    # interpolate patches to base size
    features = F.interpolate(
        features,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    return features


class DepthProPatchEncoder(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        self.intermediate_hook_ids = config.intermediate_hook_ids
        self.intermediate_feature_dims = config.intermediate_feature_dims
        self.scaled_images_ratios = config.scaled_images_ratios
        self.scaled_images_overlap_ratios = config.scaled_images_overlap_ratios
        self.scaled_images_feature_dims = config.scaled_images_feature_dims
        self.merge_padding_value = config.merge_padding_value

        self.n_scaled_images = len(config.scaled_images_ratios)
        self.n_intermediate_hooks = len(config.intermediate_hook_ids)
        self.out_size = config.image_model_config.image_size // config.image_model_config.patch_size

        self.model = AutoModel.from_config(config.patch_model_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        batch_size, num_channels, height, width = pixel_values.shape

        if min(self.scaled_images_ratios) * min(height, width) < self.config.patch_size:
            raise ValueError(
                f"Image size {height}x{width} is too small to be scaled "
                f"with scaled_images_ratios={self.scaled_images_ratios} "
                f"when patch_size={self.config.patch_size}."
            )

        # STEP 1: create 3-level image

        scaled_images = []
        for ratio in self.scaled_images_ratios:
            scaled_images.append(
                F.interpolate(
                    pixel_values,
                    scale_factor=ratio,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # STEP 2: create patches

        for i in range(self.n_scaled_images):
            scaled_images[i] = split_to_patches(
                scaled_images[i],
                patch_size=self.config.patch_size,
                overlap_ratio=self.scaled_images_overlap_ratios[i],
            )
        n_patches_per_scaled_image = [len(i) for i in scaled_images]
        patches = torch.cat(scaled_images[::-1], dim=0)  # -1 as patch encoder expects high res patches first

        # STEP 3: apply patch encoder

        encodings = self.model(
            # each patch is processed as a separate batch
            patches,
            head_mask=head_mask,
            # required for intermediate features
            output_hidden_states=self.n_intermediate_hooks > 0,
        )

        scaled_images_last_hidden_state = torch.split_with_sizes(encodings[0], n_patches_per_scaled_image[::-1])
        # -1 (reverse list) as patch encoder returns high res patches first, we need low res first
        scaled_images_last_hidden_state = scaled_images_last_hidden_state[::-1]

        # calculate base height and width
        # base height and width are the dimensions of the lowest resolution features
        exponent_value = torch_int(math.log2(width / self.out_size))
        base_height = height // 2**exponent_value
        base_width = width // 2**exponent_value

        # STEP 4: get patch features (high_res, med_res, low_res) - (3-5) in diagram

        scaled_images_features = []
        for i in range(self.n_scaled_images):
            hidden_state = scaled_images_last_hidden_state[i]
            batch_size = batch_size
            padding = torch_int(self.merge_padding_value * (1 / self.scaled_images_ratios[i]))
            output_height = base_height * 2**i
            output_width = base_width * 2**i
            features = reconstruct_feature_maps(
                hidden_state,
                batch_size=batch_size,
                padding=padding,
                output_size=(output_height, output_width),
            )
            scaled_images_features.append(features)

        # STEP 5: get intermediate features - (1-2) in diagram

        intermediate_features = []
        for i in range(self.n_intermediate_hooks):
            # +1 to correct index position as hidden_states contain embedding output as well
            hidden_state = encodings[2][self.intermediate_hook_ids[i] + 1]
            padding = torch_int(self.merge_padding_value * (1 / self.scaled_images_ratios[-1]))
            output_height = base_height * 2 ** (self.n_scaled_images - 1)
            output_width = base_width * 2 ** (self.n_scaled_images - 1)
            features = reconstruct_feature_maps(
                hidden_state,
                batch_size=batch_size,
                padding=padding,
                output_size=(output_height, output_width),
            )
            intermediate_features.append(features)

        # STEP 7: combine all features
        features = [*scaled_images_features, *intermediate_features]

        return features


class DepthProImageEncoder(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.out_size = config.image_model_config.image_size // config.image_model_config.patch_size

        self.model = AutoModel.from_config(config.image_model_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, DepthProOutput]:
        batch_size, num_channels, height, width = pixel_values.shape

        # scale the image for image_encoder
        size = self.config.image_model_config.image_size
        pixel_values = F.interpolate(
            pixel_values,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        encodings = self.model(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # calculate base height and width
        # base height and width are the dimensions of the lowest resolution features
        exponent_value = torch_int(math.log2(width / self.out_size))
        base_height = height // 2**exponent_value
        base_width = width // 2**exponent_value

        features = reconstruct_feature_maps(
            encodings[0],
            batch_size=batch_size,
            padding=0,
            output_size=(base_height, base_width),
        )

        if not return_dict:
            return (encodings[0], features) + encodings[2:]  # ignore last_hidden_state and poooler output

        return DepthProOutput(
            last_hidden_state=encodings.last_hidden_state,
            features=features,
            hidden_states=encodings.hidden_states,
            attentions=encodings.attentions,
        )


class DepthProEncoder(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.intermediate_hook_ids = config.intermediate_hook_ids
        self.intermediate_feature_dims = config.intermediate_feature_dims
        self.scaled_images_ratios = config.scaled_images_ratios
        self.scaled_images_overlap_ratios = config.scaled_images_overlap_ratios
        self.scaled_images_feature_dims = config.scaled_images_feature_dims
        self.merge_padding_value = config.merge_padding_value

        self.n_scaled_images = len(self.scaled_images_ratios)
        self.n_intermediate_hooks = len(self.intermediate_hook_ids)

        self.patch_encoder = DepthProPatchEncoder(config)
        self.image_encoder = DepthProImageEncoder(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, DepthProOutput]:
        batch_size, num_channels, height, width = pixel_values.shape

        patch_features = self.patch_encoder(
            pixel_values,
            head_mask=head_mask,
        )
        image_encodings = self.image_encoder(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_features = image_encodings[1]  # index 1 contains features

        features = [image_features, *patch_features]

        if not return_dict:
            return (image_encodings[0], features) + image_encodings[2:]

        return DepthProOutput(
            last_hidden_state=image_encodings.last_hidden_state,
            features=features,
            hidden_states=image_encodings.hidden_states,
            attentions=image_encodings.attentions,
        )


class DepthProFeatureUpsampleBlock(nn.Module):
    def __init__(
        self,
        config: DepthProConfig,
        input_dims: int,
        intermediate_dims: int,
        output_dims: int,
        n_upsample_layers: int,
        use_proj: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        # create first projection layer
        if use_proj:
            proj = nn.Conv2d(
                in_channels=input_dims,
                out_channels=intermediate_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
            self.layers.append(proj)

        # create following upsample layers
        for i in range(n_upsample_layers):
            in_channels = intermediate_dims if i == 0 else output_dims
            layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=output_dims,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            )
            self.layers.append(layer)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features)
        return features


class DepthProFeatureUpsample(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.n_scaled_images = len(self.config.scaled_images_ratios)
        self.n_intermediate_hooks = len(self.config.intermediate_hook_ids)

        # for image_features
        self.image_block = DepthProFeatureUpsampleBlock(
            config=config,
            input_dims=config.image_model_config.hidden_size,
            intermediate_dims=config.image_model_config.hidden_size,
            output_dims=config.scaled_images_feature_dims[0],
            n_upsample_layers=1,
            use_proj=False,
            bias=True,
        )

        # for scaled_images_features
        self.scaled_images = nn.ModuleList()
        for i, feature_dims in enumerate(config.scaled_images_feature_dims):
            block = DepthProFeatureUpsampleBlock(
                config=config,
                input_dims=config.patch_model_config.hidden_size,
                intermediate_dims=feature_dims,
                output_dims=feature_dims,
                n_upsample_layers=1,
            )
            self.scaled_images.append(block)

        # for intermediate_features
        self.intermediate = nn.ModuleList()
        for i, feature_dims in enumerate(config.intermediate_feature_dims):
            intermediate_dims = config.fusion_hidden_size if i == 0 else feature_dims
            block = DepthProFeatureUpsampleBlock(
                config=config,
                input_dims=config.patch_model_config.hidden_size,
                intermediate_dims=intermediate_dims,
                output_dims=feature_dims,
                n_upsample_layers=2 + i,
            )
            self.intermediate.append(block)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features[0] = self.image_block(features[0])

        for i in range(self.n_scaled_images):
            features[i + 1] = self.scaled_images[i](features[i + 1])

        for i in range(self.n_intermediate_hooks):
            features[self.n_scaled_images + i + 1] = self.intermediate[i](features[self.n_scaled_images + i + 1])

        return features


class DepthProFeatureProjection(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        combined_feature_dims = config.scaled_images_feature_dims + config.intermediate_feature_dims
        self.projections = nn.ModuleList()
        for i, in_channels in enumerate(combined_feature_dims):
            if i == len(combined_feature_dims) - 1 and in_channels == config.fusion_hidden_size:
                # projection for last layer can be ignored if input and output channels already match
                self.projections.append(nn.Identity())
            else:
                self.projections.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=config.fusion_hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected_features = []
        for i, projection in enumerate(self.projections):
            upsampled_feature = projection(features[i])
            projected_features.append(upsampled_feature)
        return projected_features


class DepthProNeck(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        self.feature_upsample = DepthProFeatureUpsample(config)
        self.fuse_image_with_low_res = nn.Conv2d(
            in_channels=config.scaled_images_feature_dims[0] * 2,
            out_channels=config.scaled_images_feature_dims[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.feature_projection = DepthProFeatureProjection(config)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features = self.feature_upsample(features)
        # global features = low res features + image features
        global_features = torch.cat((features[1], features[0]), dim=1)
        global_features = self.fuse_image_with_low_res(global_features)
        features = [global_features, *features[2:]]
        features = self.feature_projection(features)
        return features


# General docstring
_CONFIG_FOR_DOC = "DepthProConfig"


DEPTH_PRO_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthProConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEPTH_PRO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

DEPTH_PRO_FOR_DEPTH_ESTIMATION_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthProConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        use_fov_model (`bool`, *optional*, defaults to `True`):
            Whether to use `DepthProFovModel` to generate the field of view.
"""


class DepthProPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DepthProConfig
    base_model_prefix = "depth_pro"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["DepthProPreActResidualLayer"]
    _keys_to_ignore_on_load_unexpected = ["fov_model.*"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "The bare DepthPro Model transformer outputting raw hidden-states without any specific head on top.",
    DEPTH_PRO_START_DOCSTRING,
)
class DepthProModel(DepthProPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = DepthProEncoder(config)
        self.neck = DepthProNeck(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.image_encoder.model.get_input_embeddings()

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DepthProOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, DepthProModel

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> checkpoint = "apple/DepthPro-hf"
        >>> processor = AutoProcessor.from_pretrained(checkpoint)
        >>> model = DepthProModel.from_pretrained(checkpoint)

        >>> # prepare image for the model
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     output = model(**inputs)

        >>> output.last_hidden_state.shape
        torch.Size([1, 35, 577, 1024])
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encodings = self.encoder(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        features = encodings[1]  # index 1 contains features
        features = self.neck(features)

        if not return_dict:
            return (encodings[0], features) + encodings[2:]

        return DepthProOutput(
            last_hidden_state=encodings.last_hidden_state,
            features=features,
            hidden_states=encodings.hidden_states,
            attentions=encodings.attentions,
        )


# Copied from transformers.models.dpt.modeling_dpt.DPTPreActResidualLayer DPT->DepthPro
class DepthProPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthProConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


# Modified from transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer
# except it uses deconv and skip_add and needs no interpolation
class DepthProFeatureFusionLayer(nn.Module):
    def __init__(self, config: DepthProConfig, use_deconv: bool = True):
        super().__init__()
        self.config = config
        self.use_deconv = use_deconv

        self.residual_layer1 = DepthProPreActResidualLayer(config)
        self.residual_layer2 = DepthProPreActResidualLayer(config)

        if self.use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=config.fusion_hidden_size,
                out_channels=config.fusion_hidden_size,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

    def forward(self, hidden_state: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is not None:
            residual = self.residual_layer1(residual)
            hidden_state = hidden_state + residual

        hidden_state = self.residual_layer2(hidden_state)
        if self.use_deconv:
            hidden_state = self.deconv(hidden_state)
        hidden_state = self.projection(hidden_state)

        return hidden_state


# Modified from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage with DPT->DepthPro
# with deconv and reversed layers
class DepthProFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_layers = len(config.intermediate_hook_ids) + len(config.scaled_images_ratios)
        self.intermediate = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.intermediate.append(DepthProFeatureFusionLayer(config))

        # final layer doesnot require deconvolution
        self.final = DepthProFeatureFusionLayer(config, use_deconv=False)

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_layers != len(hidden_states):
            raise ValueError(
                f"num_layers={self.num_layers} in DepthProFeatureFusionStage"
                f"doesnot match len(hidden_states)={len(hidden_states)}"
            )

        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(hidden_states[:-1], self.intermediate):
            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        hidden_state = hidden_states[-1]
        fused_hidden_state = self.final(fused_hidden_state, hidden_state)
        fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DepthProFovEncoder(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.out_size = config.image_model_config.image_size // config.image_model_config.patch_size

        self.model = AutoModel.from_config(config.fov_model_config)
        self.neck = nn.Linear(config.fov_model_config.hidden_size, config.fusion_hidden_size // 2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        # scale the image for fov_encoder
        size = self.config.fov_model_config.image_size
        pixel_values = F.interpolate(
            pixel_values,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        encodings = self.model(
            pixel_values=pixel_values,
            head_mask=head_mask,
        )
        hidden_state = encodings[0]
        hidden_state = self.neck(hidden_state)

        # calculate base height and width
        # base height and width are the dimensions of the lowest resolution features
        exponent_value = torch_int(math.log2(width / self.out_size))
        base_height = height // 2**exponent_value
        base_width = width // 2**exponent_value

        features = reconstruct_feature_maps(
            hidden_state,
            batch_size=batch_size,
            padding=0,
            output_size=(base_height, base_width),
        )

        return features


class DepthProFovHead(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.fusion_hidden_size = config.fusion_hidden_size
        self.out_size = config.image_model_config.image_size // config.image_model_config.patch_size

        # create initial head layers
        self.layers = nn.ModuleList()
        for i in range(config.num_fov_head_layers):
            self.layers.append(
                nn.Conv2d(
                    math.ceil(self.fusion_hidden_size / 2 ** (i + 1)),
                    math.ceil(self.fusion_hidden_size / 2 ** (i + 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.layers.append(nn.ReLU(True))
        # calculate expected shapes to finally generate a scalar output from final head layer
        final_in_channels = math.ceil(self.fusion_hidden_size / 2 ** (config.num_fov_head_layers + 1))
        final_kernel_size = torch_int((self.out_size - 1) / 2**config.num_fov_head_layers + 1)
        self.layers.append(
            nn.Conv2d(
                in_channels=final_in_channels, out_channels=1, kernel_size=final_kernel_size, stride=1, padding=0
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(
            features,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )
        for layer in self.layers:
            features = layer(features)
        return features


class DepthProFovModel(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.fusion_hidden_size = config.fusion_hidden_size

        self.fov_encoder = DepthProFovEncoder(config)
        self.conv = nn.Conv2d(
            self.fusion_hidden_size, self.fusion_hidden_size // 2, kernel_size=3, stride=2, padding=1
        )
        self.activation = nn.ReLU(inplace=True)
        self.head = DepthProFovHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        global_features: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fov_features = self.fov_encoder(pixel_values, head_mask)

        global_features = self.conv(global_features)
        global_features = self.activation(global_features)

        fov_features = fov_features + global_features
        fov_output = self.head(fov_features)
        fov_output = fov_output.flatten()

        return fov_output


class DepthProDepthEstimationHead(nn.Module):
    """
    The DepthProDepthEstimationHead module serves as the output head for depth estimation tasks.
    This module comprises a sequence of convolutional and transposed convolutional layers
    that process the feature map from the fusion to produce a single-channel depth map.
    Key operations include dimensionality reduction and upsampling to match the input resolution.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        features = config.fusion_hidden_size
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(
                    in_channels=features // 2,
                    out_channels=features // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        predicted_depth = hidden_states.squeeze(dim=1)
        return predicted_depth


@add_start_docstrings(
    """
    DepthPro Model with a depth estimation head on top (consisting of 3 convolutional layers).
    """,
    DEPTH_PRO_FOR_DEPTH_ESTIMATION_START_DOCSTRING,
)
class DepthProForDepthEstimation(DepthProPreTrainedModel):
    def __init__(self, config, use_fov_model=None):
        super().__init__(config)
        self.config = config
        self.use_fov_model = use_fov_model if use_fov_model is not None else self.config.use_fov_model

        # dinov2 (vit) like encoders
        self.depth_pro = DepthProModel(config)

        # dpt (vit) like fusion stage
        self.fusion_stage = DepthProFeatureFusionStage(config)

        # depth estimation head
        self.head = DepthProDepthEstimationHead(config)

        # dinov2 (vit) like encoder
        self.fov_model = DepthProFovModel(config) if self.use_fov_model else None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthProDepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthProDepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DepthProForDepthEstimation
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> checkpoint = "apple/DepthPro-hf"
        >>> processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = DepthProForDepthEstimation.from_pretrained(checkpoint)

        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)

        >>> # prepare image for the model
        >>> inputs = processor(images=image, return_tensors="pt").to(device)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = processor.post_process_depth_estimation(
        ...     outputs, target_sizes=[(image.height, image.width)],
        ... )

        >>> # get the field of view (fov) predictions
        >>> field_of_view = post_processed_output[0]["field_of_view"]
        >>> focal_length = post_processed_output[0]["focal_length"]

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        depth_pro_outputs = self.depth_pro(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        features = depth_pro_outputs.features
        fused_hidden_states = self.fusion_stage(features)
        predicted_depth = self.head(fused_hidden_states[-1])

        if self.use_fov_model:
            # frozen features from encoder are used
            features_for_fov = features[0].detach()
            fov = self.fov_model(
                pixel_values=pixel_values,
                global_features=features_for_fov,
                head_mask=head_mask,
            )
        else:
            fov = None

        if not return_dict:
            outputs = [loss, predicted_depth, fov, depth_pro_outputs.hidden_states, depth_pro_outputs.attentions]
            return tuple(v for v in outputs if v is not None)

        return DepthProDepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            field_of_view=fov,
            hidden_states=depth_pro_outputs.hidden_states,
            attentions=depth_pro_outputs.attentions,
        )


__all__ = ["DepthProPreTrainedModel", "DepthProModel", "DepthProForDepthEstimation"]
