# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Swin2SR Transformer model."""

import collections.abc
import math
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..swinv2.modeling_swinv2 import (
    Swinv2Attention,
    Swinv2DropPath,
    Swinv2Layer,
    Swinv2MLP,
    Swinv2PatchMerging,
    Swinv2PreTrainedModel,
)
from .configuration_swin2sr import Swin2SRConfig


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Swin2SR encoder's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class Swin2SREncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class Swin2SRDropPath(Swinv2DropPath):
    pass


class Swin2SRPatchMerging(Swinv2PatchMerging):
    pass


class Swin2SRAttention(Swinv2Attention):
    pass


class Swin2SRMLP(Swinv2MLP):
    pass


class Swin2SRLayer(Swinv2Layer):
    pass


class Swin2SREmbeddings(nn.Module):
    """
    Construct the patch and optional position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = Swin2SRPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.window_size = config.window_size

    def forward(self, pixel_values: torch.FloatTensor | None) -> tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class Swin2SRPatchEmbeddings(nn.Module):
    def __init__(self, config, normalize_patches=True):
        super().__init__()
        num_channels = config.embed_dim
        image_size, patch_size = config.image_size, config.patch_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.projection = nn.Conv2d(num_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layernorm = nn.LayerNorm(config.embed_dim) if normalize_patches else None

    def forward(self, embeddings: torch.FloatTensor | None) -> tuple[torch.Tensor, tuple[int]]:
        embeddings = self.projection(embeddings)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)

        return embeddings, output_dimensions


class Swin2SRPatchUnEmbeddings(nn.Module):
    r"""Image to Patch Unembedding"""

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim

    def forward(self, embeddings, x_size):
        batch_size, height_width, num_channels = embeddings.shape
        embeddings = embeddings.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return embeddings


class Swin2SRStage(GradientCheckpointingLayer):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=0):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                Swin2SRLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        if config.resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif config.resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = Swin2SRPatchEmbeddings(config, normalize_patches=False)

        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states

        last_attn_weights = None
        for layer_module in self.layers:
            hidden_states, last_attn_weights = layer_module(hidden_states, input_dimensions, **kwargs)

        hidden_states = self.patch_unembed(hidden_states, input_dimensions)
        hidden_states = self.conv(hidden_states)
        hidden_states, _ = self.patch_embed(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states, last_attn_weights


@auto_docstring
class Swin2SRPreTrainedModel(Swinv2PreTrainedModel):
    config: Swin2SRConfig
    base_model_prefix = "swin2sr"
    _no_split_modules = ["Swin2SRStage"]
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Swin2SRStage, index=0, capture_initial_hidden_state=True),
        "attentions": OutputRecorder(Swin2SRStage, index=1, capture_initial_hidden_state=False),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Swin2SRAttention):
            init.constant_(module.logit_scale, math.log(10))
            relative_coords_table, relative_position_index = module.create_coords_table_and_index()
            init.copy_(module.relative_coords_table, relative_coords_table)
            init.copy_(module.relative_position_index, relative_position_index)
        elif isinstance(module, Swin2SRModel):
            if module.config.num_channels == 3 and module.config.num_channels_out == 3:
                mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
            else:
                mean = torch.zeros(1, 1, 1, 1)
            init.copy_(module.mean, mean)


class Swin2SREncoder(Swin2SRPreTrainedModel):
    def __init__(self, config, grid_size):
        super().__init__(config)
        self.num_stages = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]
        self.stages = nn.ModuleList(
            [
                Swin2SRStage(
                    config=config,
                    dim=config.embed_dim,
                    input_resolution=(grid_size[0], grid_size[1]),
                    depth=config.depths[stage_idx],
                    num_heads=config.num_heads[stage_idx],
                    drop_path=dpr[sum(config.depths[:stage_idx]) : sum(config.depths[: stage_idx + 1])],
                    pretrained_window_size=0,
                )
                for stage_idx in range(self.num_stages)
            ]
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Swin2SREncoderOutput:
        r"""
        input_dimensions (`tuple[int, int]`):
            Spatial `(height, width)` of the patch grid entering the encoder.
        """
        for stage_module in self.stages:
            hidden_states, _ = stage_module(hidden_states, input_dimensions, **kwargs)

        return Swin2SREncoderOutput(last_hidden_state=hidden_states)


@auto_docstring
class Swin2SRModel(Swin2SRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.num_channels == 3 and config.num_channels_out == 3:
            mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        else:
            mean = torch.zeros(1, 1, 1, 1)
        self.mean = nn.Buffer(mean, persistent=False)

        self.img_range = config.img_range

        self.first_convolution = nn.Conv2d(config.num_channels, config.embed_dim, 3, 1, 1)
        self.embeddings = Swin2SREmbeddings(config)
        self.encoder = Swin2SREncoder(config, grid_size=self.embeddings.patch_embeddings.patches_resolution)

        self.layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def pad_and_normalize(self, pixel_values):
        _, _, height, width = pixel_values.size()

        # 1. pad
        window_size = self.config.window_size
        modulo_pad_height = (window_size - height % window_size) % window_size
        modulo_pad_width = (window_size - width % window_size) % window_size
        pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), "reflect")

        # 2. normalize
        mean = self.mean.to(device=pixel_values.device, dtype=pixel_values.dtype)
        pixel_values = (pixel_values - mean) * self.img_range

        return pixel_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        _, _, height, width = pixel_values.shape

        # some preprocessing: padding + normalization
        pixel_values = self.pad_and_normalize(pixel_values)

        embeddings = self.first_convolution(pixel_values)
        embedding_output, input_dimensions = self.embeddings(embeddings)

        encoder_outputs = self.encoder(embedding_output, input_dimensions, **kwargs)

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        sequence_output = self.patch_unembed(sequence_output, (height, width))
        sequence_output = self.conv_after_body(sequence_output) + embeddings

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Swin2SRUpsample(nn.Module):
    """Upsample module.

    Args:
        scale (`int`):
            Scale factor. Supported scales: 2^n and 3.
        num_features (`int`):
            Channel number of intermediate features.
    """

    def __init__(self, scale, num_features):
        super().__init__()

        self.scale = scale
        if (scale & (scale - 1)) == 0:
            # scale = 2^n
            for i in range(int(math.log2(scale))):
                self.add_module(f"convolution_{i}", nn.Conv2d(num_features, 4 * num_features, 3, 1, 1))
                self.add_module(f"pixelshuffle_{i}", nn.PixelShuffle(2))
        elif scale == 3:
            self.convolution = nn.Conv2d(num_features, 9 * num_features, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(3)
        else:
            raise ValueError(f"Scale {scale} is not supported. Supported scales: 2^n and 3.")

    def forward(self, hidden_state):
        if (self.scale & (self.scale - 1)) == 0:
            for i in range(int(math.log2(self.scale))):
                hidden_state = self.__getattr__(f"convolution_{i}")(hidden_state)
                hidden_state = self.__getattr__(f"pixelshuffle_{i}")(hidden_state)

        elif self.scale == 3:
            hidden_state = self.convolution(hidden_state)
            hidden_state = self.pixelshuffle(hidden_state)

        return hidden_state


class Swin2SRUpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)

    Used in lightweight SR to save parameters.

    Args:
        scale (int):
            Scale factor. Supported scales: 2^n and 3.
        in_channels (int):
            Channel number of intermediate features.
        out_channels (int):
            Channel number of output features.
    """

    def __init__(self, scale, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, (scale**2) * out_channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)

        return x


class Swin2SRPixelShuffleUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.upsample = Swin2SRUpsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output):
        x = self.conv_before_upsample(sequence_output)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.final_convolution(x)

        return x


class Swin2SRNearestConvUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        if config.upscale != 4:
            raise ValueError("The nearest+conv upsampler only supports an upscale factor of 4 at the moment.")

        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, sequence_output):
        sequence_output = self.conv_before_upsample(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.lrelu(
            self.conv_up1(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        sequence_output = self.lrelu(
            self.conv_up2(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        reconstruction = self.final_convolution(self.lrelu(self.conv_hr(sequence_output)))
        return reconstruction


class Swin2SRPixelShuffleAuxUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()

        self.upscale = config.upscale
        self.conv_bicubic = nn.Conv2d(config.num_channels, num_features, 3, 1, 1)
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv_aux = nn.Conv2d(num_features, config.num_channels, 3, 1, 1)
        self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_features, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Swin2SRUpsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output, bicubic, height, width):
        bicubic = self.conv_bicubic(bicubic)
        sequence_output = self.conv_before_upsample(sequence_output)
        sequence_output = self.activation(sequence_output)
        aux = self.conv_aux(sequence_output)
        sequence_output = self.conv_after_aux(aux)
        sequence_output = (
            self.upsample(sequence_output)[:, :, : height * self.upscale, : width * self.upscale]
            + bicubic[:, :, : height * self.upscale, : width * self.upscale]
        )
        reconstruction = self.final_convolution(sequence_output)

        return reconstruction, aux


@auto_docstring(
    custom_intro="""
    Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.
    """
)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.swin2sr = Swin2SRModel(config)
        self.upsampler = config.upsampler
        self.upscale = config.upscale

        # Upsampler
        num_features = 64
        if self.upsampler == "pixelshuffle":
            self.upsample = Swin2SRPixelShuffleUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffle_aux":
            self.upsample = Swin2SRPixelShuffleAuxUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = Swin2SRUpsampleOneStep(config.upscale, config.embed_dim, config.num_channels_out)
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            self.upsample = Swin2SRNearestConvUpsampler(config, num_features)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.final_convolution = nn.Conv2d(config.embed_dim, config.num_channels_out, 3, 1, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageSuperResolutionOutput:
        r"""
        Example:
         ```python
         >>> import torch
         >>> import numpy as np
         >>> from PIL import Image
         >>> import httpx
        >>> from io import BytesIO

         >>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

         >>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
         >>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

         >>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
         >>> with httpx.stream("GET", url) as response:
         ...     image = Image.open(BytesIO(response.read()))
         >>> # prepare image for the model
         >>> inputs = processor(image, return_tensors="pt")

         >>> # forward pass
         >>> with torch.no_grad():
         ...     outputs = model(**inputs)

         >>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
         >>> output = np.moveaxis(output, source=0, destination=-1)
         >>> output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
         >>> # you can visualize `output` with `Image.fromarray`
         ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not supported at the moment")

        height, width = pixel_values.shape[2:]

        if self.config.upsampler == "pixelshuffle_aux":
            bicubic = nn.functional.interpolate(
                pixel_values,
                size=(height * self.upscale, width * self.upscale),
                mode="bicubic",
                align_corners=False,
            )

        outputs = self.swin2sr(pixel_values, **kwargs)

        sequence_output = outputs.last_hidden_state

        if self.upsampler in ["pixelshuffle", "pixelshuffledirect", "nearest+conv"]:
            reconstruction = self.upsample(sequence_output)
        elif self.upsampler == "pixelshuffle_aux":
            reconstruction, aux = self.upsample(sequence_output, bicubic, height, width)
            aux = aux / self.swin2sr.img_range + self.swin2sr.mean
        else:
            reconstruction = pixel_values + self.final_convolution(sequence_output)

        reconstruction = reconstruction / self.swin2sr.img_range + self.swin2sr.mean
        reconstruction = reconstruction[:, :, : height * self.upscale, : width * self.upscale]

        return ImageSuperResolutionOutput(
            loss=loss,
            reconstruction=reconstruction,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Swin2SRForImageSuperResolution", "Swin2SRModel", "Swin2SRPreTrainedModel"]
