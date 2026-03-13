# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT MAE (masked autoencoder) model - modular implementation."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_pos_embed_utils import build_2d_sinusoidal_position_embedding
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEncoder,
    ViTLayer,
    ViTMLP,
    ViTPatchEmbeddings,
    ViTPreTrainedModel,
)
from .configuration_vit_mae import ViTMAEConfig


logger = logging.get_logger(__name__)



@dataclass
@auto_docstring(
    custom_intro="""
    Class for ViTMAEModel's outputs, with potential hidden states and attentions.
    """
)
class ViTMAEModelOutput(ModelOutput):
    r"""
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    last_hidden_state: torch.FloatTensor | None = None
    mask: torch.LongTensor | None = None
    ids_restore: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Class for ViTMAEDecoder's outputs, with potential hidden states and attentions.
    """
)
class ViTMAEDecoderOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
        Pixel reconstruction logits.
    """

    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.
    """
)
class ViTMAEForPreTrainingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`):
        Pixel reconstruction loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
        Pixel reconstruction logits.
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    mask: torch.LongTensor | None = None
    ids_restore: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class ViTMAEPatchEmbeddings(ViTPatchEmbeddings):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer. MAE variant with interpolate_pos_encoding and image size validation.
    """

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings for MAE.
    """

    def __init__(self, config: ViTMAEConfig):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.patch_size = config.patch_size
        self.config = config

    def initialize_weights(self):
        if getattr(self.patch_embeddings.projection, "_is_hf_initialized", False):
            return
        # initialize (and freeze) position embeddings by sin-cos embedding
        grid_size = int(self.patch_embeddings.num_patches**0.5)
        pos_embed = build_2d_sinusoidal_position_embedding(
            height=grid_size,
            width=grid_size,
            embed_dim=self.position_embeddings.shape[-1],
            cls_token=True,
        )
        # The original ViT-MAE implementation had a variable naming bug that
        # swapped h and w, producing [sin_w|cos_w|sin_h|cos_h] instead of the
        # canonical [sin_h|cos_h|sin_w|cos_w]. Pretrained weights rely on this
        # layout, so we rotate the two halves to match.
        half = pos_embed.shape[-1] // 2
        pos_embed = torch.cat([pos_embed[..., half:], pos_embed[..., :half]], dim=-1)
        init.copy_(self.position_embeddings, pos_embed.unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight
        init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        init.normal_(self.cls_token, std=self.config.initializer_range)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
    ):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


# Pass-through: ViT MAE encoder uses same architecture as ViT
class ViTMAEAttention(ViTAttention):
    pass


class ViTMAEMLP(ViTMLP):
    pass


class ViTMAELayer(ViTLayer):
    pass


class ViTMAEEncoder(ViTEncoder):
    pass


class ViTMAEDecoder(nn.Module):
    def __init__(self, config: ViTMAEConfig, num_patches: int):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels,
            bias=True,
        )
        self.gradient_checkpointing = False
        self.config = decoder_config
        self.initialize_weights(num_patches)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        This method is a modified version of the interpolation function for ViT-mae model at the decoder, that
        allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
        resolution images.

        Adapted from:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # -1 removes the class dimension since we later append it without interpolation
        embeddings_positions = embeddings.shape[1] - 1

        # Separation of class token and patch tokens
        class_pos_embed = self.decoder_pos_embed[:, :1]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]

        dim = self.decoder_pos_embed.shape[-1]

        # Increasing a dimension to enable bicubic interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)

        # permute to bring the dimension to be interpolated, to the last
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolating the decoder position embeddings shape wrt embeddings shape i.e (x).
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(patch_pos_embed.shape[-2], embeddings_positions),
            mode="bicubic",
            align_corners=False,
        )

        # Converting back to the original shape
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def initialize_weights(self, num_patches: int):
        grid_size = int(num_patches**0.5)
        decoder_pos_embed = build_2d_sinusoidal_position_embedding(
            height=grid_size,
            width=grid_size,
            embed_dim=self.decoder_pos_embed.shape[-1],
            cls_token=True,
        )
        # See comment in initialize_weights above: rotate h/w blocks to match pretrained layout.
        half = decoder_pos_embed.shape[-1] // 2
        decoder_pos_embed = torch.cat([decoder_pos_embed[..., half:], decoder_pos_embed[..., :half]], dim=-1)
        init.copy_(self.decoder_pos_embed, decoder_pos_embed.unsqueeze(0))

        init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> ViTMAEDecoderOutput:
        # Embed tokens
        x = self.decoder_embed(hidden_states)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        # Unshuffle
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add pos embed
        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed

        # Apply Transformer layers (blocks)
        for layer_module in self.decoder_layers:
            hidden_states = layer_module(hidden_states)

        hidden_states = self.decoder_norm(hidden_states)

        # Predictor projection
        logits = self.decoder_pred(hidden_states)

        # Remove cls token
        logits = logits[:, 1:, :]

        return ViTMAEDecoderOutput(logits=logits)


@auto_docstring
class ViTMAEPreTrainedModel(ViTPreTrainedModel):
    config: ViTMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTMAEEmbeddings", "ViTMAELayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": ViTMAELayer,
        "attentions": ViTMAEAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, ViTMAEEmbeddings):
            module.initialize_weights()
        elif isinstance(module, ViTMAEDecoder):
            init.zeros_(module.mask_token)
            init.zeros_(module.decoder_pos_embed)


@auto_docstring
class ViTMAEModel(ViTMAEPreTrainedModel):
    def __init__(self, config: ViTMAEConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        interpolate_pos_encoding: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ViTMAEModelOutput:
        r"""
        noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mainly used for testing purposes to control randomness and maintain the reproducibility
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values,
            noise=noise,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, attention_mask, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return ViTMAEModelOutput(last_hidden_state=sequence_output, mask=mask, ids_restore=ids_restore)


@auto_docstring(
    custom_intro="""
    The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """
)
class ViTMAEForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config: ViTMAEConfig):
        super().__init__(config)
        self.config = config

        self.vit = ViTMAEModel(config)
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def patchify(self, pixel_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        if not interpolate_pos_encoding and (
            pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0
        ):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        batch_size = pixel_values.shape[0]
        num_patches_h = pixel_values.shape[2] // patch_size
        num_patches_w = pixel_values.shape[3] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h,
            patch_size,
            num_patches_w,
            patch_size,
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h * num_patches_w,
            patch_size**2 * num_channels,
        )
        return patchified_pixel_values

    def unpatchify(
        self,
        patchified_pixel_values,
        original_image_size: tuple[int, int] | None = None,
    ):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`):
                Patchified pixel values.
            original_image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward_loss(
        self,
        pixel_values,
        pred,
        mask,
        interpolate_pos_encoding: bool = False,
    ):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`):
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        interpolate_pos_encoding: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ViTMAEForPreTrainingOutput:
        r"""
        noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mainly used for testing purposes to control randomness and maintain the reproducibility
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read())).convert("RGB")

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""

        outputs: ViTMAEModelOutput = self.vit(
            pixel_values,
            noise=noise,
            interpolate_pos_encoding=interpolate_pos_encoding,
            attention_mask=attention_mask,
            **kwargs,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs: ViTMAEDecoderOutput = self.decoder(
            latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding
        )
        logits = decoder_outputs.logits

        loss = self.forward_loss(pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ViTMAEForPreTraining",
    "ViTMAEModel",
    "ViTMAEPreTrainedModel",
]
