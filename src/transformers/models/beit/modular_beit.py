# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BEiT model."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ... import initialization as init
from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import TransformersKwargs, auto_docstring, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..resnet.modeling_resnet import ResNetConvLayer
from ..swin.modeling_swin import SwinDropPath
from ..vit.modeling_vit import ViTAttention, ViTEmbeddings, ViTLayer, ViTMLP, ViTPatchEmbeddings, ViTPreTrainedModel
from .configuration_beit import BeitConfig


@auto_docstring(
    custom_intro="""
    Class for outputs of [`BeitModel`].
    """
)
@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
        *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
        will be returned.
    """


class BeitPatchEmbeddings(ViTPatchEmbeddings):
    pass


class BeitEmbeddings(ViTEmbeddings):
    def __init__(self, config: BeitConfig) -> None:
        nn.Module.__init__(self)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_mask_token else None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        self.patch_size = config.patch_size
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = (
            nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
            if config.use_absolute_position_embeddings
            else None
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class BeitRelativePositionBias(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        image_size = config.image_size
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        self.window_size = (image_size[0] // config.patch_size, image_size[1] // config.patch_size)
        self.num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

    @staticmethod
    @compile_compatible_method_lru_cache(maxsize=10)
    def generate_relative_position_index(window_size: tuple[int, int]) -> torch.Tensor:
        """
        This method creates the relative position index, modified to support arbitrary window sizes,
        as introduced in [MiDaS v3.1](https://huggingface.co/papers/2307.14460).
        """
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        window_area = window_size[0] * window_size[1]

        # Pair-wise relative position index for each token inside the window
        coords_flatten = torch.flatten(
            torch.stack(torch.meshgrid(torch.arange(window_size[0]), torch.arange(window_size[1]), indexing="ij")),
            start_dim=1,
        )  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        # Wh*Ww, Wh*Ww, 2 — shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = num_relative_distance - 3  # cls to token
        relative_position_index[0:, 0] = num_relative_distance - 2  # token to cls
        relative_position_index[0, 0] = num_relative_distance - 1  # cls to cls
        return relative_position_index

    def forward(self, window_size, interpolate_pos_encoding: bool = False, dim_size=None) -> torch.Tensor:
        """
        Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
        """
        old_height = 2 * self.window_size[0] - 1
        old_width = 2 * self.window_size[1] - 1

        new_height = 2 * window_size[0] - 1
        new_width = 2 * window_size[1] - 1

        old_relative_position_bias_table = self.relative_position_bias_table

        old_num_relative_distance = self.num_relative_distance
        new_num_relative_distance = new_height * new_width + 3

        old_sub_table = old_relative_position_bias_table[: old_num_relative_distance - 3]

        old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
        new_sub_table = nn.functional.interpolate(
            old_sub_table, size=(torch_int(new_height), torch_int(new_width)), mode="bilinear"
        )
        new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

        new_relative_position_bias_table = torch.cat(
            [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3 :]]
        )

        relative_position_index = self.generate_relative_position_index(window_size)
        relative_position_bias = new_relative_position_bias_table[relative_position_index.view(-1)]

        # patch_size*num_patches_height, patch_size*num_patches_width, num_attention_heads
        relative_position_bias = relative_position_bias.view(
            window_size[0] * window_size[1] + 1, window_size[0] * window_size[1] + 1, -1
        )
        # num_attention_heads, patch_size*num_patches_width, patch_size*num_patches_height
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if interpolate_pos_encoding:
            relative_position_bias = nn.functional.interpolate(
                relative_position_bias.unsqueeze(1),
                size=(dim_size, dim_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return relative_position_bias.unsqueeze(0)


class BeitAttention(ViTAttention):
    def __init__(self, config: BeitConfig):
        super().__init__(config)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)


class BeitMLP(ViTMLP):
    pass


class BeitDropPath(SwinDropPath):
    pass


class BeitLayer(ViTLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: BeitConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.patch_size = config.patch_size
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        init_values = config.layer_scale_init_value
        self.lambda_1 = (
            nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True) if init_values > 0 else 1.0
        )
        self.lambda_2 = (
            nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True) if init_values > 0 else 1.0
        )
        self.relative_position_bias = BeitRelativePositionBias(config) if config.use_relative_position_bias else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        resolution: tuple[int, int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self.relative_position_bias is not None:
            height, width = resolution
            window_size = (height // self.patch_size, width // self.patch_size)
            relative_position_bias = self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )
            attention_mask = (
                relative_position_bias + attention_mask if attention_mask is not None else relative_position_bias
            )

        # Self Attention
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.lambda_1 * hidden_states
        hidden_states = self.drop_path(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.lambda_2 * hidden_states
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


@auto_docstring
class BeitPreTrainedModel(ViTPreTrainedModel):
    _no_split_modules = ["BeitLayer"]
    _keys_to_ignore_on_load_unexpected = [r".*relative_position_index.*"]
    _supports_flash_attn = False
    _supports_flex_attn = False

    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, BeitEmbeddings):
            init.zeros_(module.cls_token)
            if module.mask_token is not None:
                init.zeros_(module.mask_token)
            if module.position_embeddings is not None:
                init.zeros_(module.position_embeddings)
        elif isinstance(module, BeitRelativePositionBias):
            init.zeros_(module.relative_position_bias_table)
        elif isinstance(module, BeitLayer):
            if isinstance(module.lambda_1, nn.Parameter):
                init.constant_(module.lambda_1, self.config.layer_scale_init_value)
                init.constant_(module.lambda_2, self.config.layer_scale_init_value)


@auto_docstring
class BeitModel(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.shared_position_bias = (
            BeitRelativePositionBias(config) if config.use_shared_relative_position_bias else None
        )
        drop_path_rates = [
            config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) for i in range(config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList([BeitLayer(config, drop_path_rate=r) for r in drop_path_rates])

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BeitModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        resolution = pixel_values.shape[2:]

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        if self.shared_position_bias is not None:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            shared_relative_position_bias = self.shared_position_bias(
                window_size, interpolate_pos_encoding=interpolate_pos_encoding, dim_size=embedding_output.shape[1]
            )
            attention_mask = (
                shared_relative_position_bias + attention_mask
                if attention_mask is not None
                else shared_relative_position_bias
            )

        hidden_states = embedding_output
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                interpolate_pos_encoding=interpolate_pos_encoding,
                resolution=resolution,
                **kwargs,
            )
        sequence_output = self.layernorm(hidden_states)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BeitModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


class BeitPooler(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mean pool patch tokens with layernorm, or take the [CLS] token
        return self.layernorm(hidden_states[:, 1:, :].mean(1)) if self.layernorm is not None else hidden_states[:, 0]


@auto_docstring(
    custom_intro="""
    Beit Model transformer with a 'language' modeling head on top. BEiT does masked image modeling by predicting
    visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT
    predict RGB pixel values. As a result, this class is incompatible with [`AutoModelForMaskedImageModeling`], so you
    will need to use [`BeitForMaskedImageModeling`] directly if you wish to do masked image modeling with BEiT.
    """
)
class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        labels: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MaskedLMOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> list(logits.shape)
        [1, 196, 8192]
        ```"""

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """
)
class BeitForImageClassification(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.beit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        pooled_output = outputs.pooler_output

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitConvLayer(ResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        bias: bool = False,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class BeitPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(pool_scale)
        self.conv = BeitConvLayer(in_channels, channels, kernel_size=1)

    def forward(self, input: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        hidden_state = self.pooling(input)
        hidden_state = self.conv(hidden_state)
        hidden_state = nn.functional.interpolate(hidden_state, size=size, mode="bilinear", align_corners=False)
        return hidden_state


class BeitPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: tuple[int, ...], in_channels: int, channels: int) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = nn.ModuleList(
            [
                BeitPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
                for pool_scale in pool_scales
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        original_size = hidden_states.size()[2:]
        return [block(hidden_states, size=original_size) for block in self.blocks]


class BeitUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://huggingface.co/papers/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
        )
        self.psp_bottleneck = BeitConvLayer(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            self.lateral_convs.append(BeitConvLayer(in_channels, self.channels, kernel_size=1))
            self.fpn_convs.append(BeitConvLayer(self.channels, self.channels, kernel_size=3, padding=1))

        self.fpn_bottleneck = BeitConvLayer(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        hidden_state = hidden_states[-1]
        hidden_state = torch.cat([hidden_state, *self.psp_modules(hidden_state)], dim=1)
        return self.psp_bottleneck(hidden_state)

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # build laterals
        laterals = []
        for lateral_conv, hidden_state in zip(self.lateral_convs, encoder_hidden_states):
            laterals.append(lateral_conv(hidden_state))

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # build outputs
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class BeitFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://huggingface.co/papers/1411.4038>).

    Args:
        config (BeitConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self, config: BeitConfig, in_index: int = 2, kernel_size: int = 3, dilation: int | tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        self.convs = nn.ModuleList()
        if self.num_convs > 0:
            self.convs.append(
                BeitConvLayer(
                    self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
            for _ in range(self.num_convs - 1):
                self.convs.append(
                    BeitConvLayer(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                    )
                )
        if self.concat_input:
            self.conv_cat = BeitConvLayer(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        residual = encoder_hidden_states[self.in_index]
        hidden_states = residual
        for conv in self.convs:
            hidden_states = conv(hidden_states)
        if self.concat_input:
            hidden_states = self.conv_cat(torch.cat([residual, hidden_states], dim=1))
        hidden_states = self.classifier(hidden_states)
        return hidden_states


class BeitFPNUpBlock(nn.Module):
    """4x upsampling block: ConvTranspose → BN → GELU → ConvTranspose."""

    def __init__(self, hidden_size: int, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride)
        self.normalization = nn.BatchNorm2d(hidden_size)
        self.activation = nn.GELU()
        self.conv_transpose2 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_transpose1(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv_transpose2(hidden_states)
        return hidden_states


class BeitFPNNeck(nn.Module):
    """
    4-level feature pyramid neck for BeiT. Produces x4 upsample, x2 upsample,
    identity, and x2 downsample outputs from the four selected ViT feature maps.
    """

    def __init__(self, config: BeitConfig):
        super().__init__()
        self.fpn1 = BeitFPNUpBlock(config.hidden_size)
        self.fpn2 = nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2)
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, feature_maps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (
            self.fpn1(feature_maps[0]),
            self.fpn2(feature_maps[1]),
            feature_maps[2],  # identity: native patch-grid resolution
            self.fpn4(feature_maps[3]),
        )


@auto_docstring
class BeitForSemanticSegmentation(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        if len(self.config.out_indices) != 4:
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        self.fpn = BeitFPNNeck(config)

        # Semantic segmentation head(s)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, BeitForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        >>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")
        kwargs["output_hidden_states"] = True
        outputs = self.beit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        encoder_hidden_states = outputs.hidden_states
        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        # out_indices are 1-based into encoder_hidden_states (index 0 is the initial patch embedding).
        # Remove the CLS token ([:, 1:]) and reshape from sequence to 2D spatial feature maps.
        feature_maps = tuple(
            encoder_hidden_states[i - 1][:, 1:].transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)
            for i in self.config.out_indices
        )
        feature_maps = self.fpn(feature_maps)

        logits = self.decode_head(feature_maps)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(feature_maps)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                ignore_index=self.config.semantic_loss_ignore_index,
                auxiliary_logits=auxiliary_logits,
                auxiliary_loss_weight=self.config.auxiliary_loss_weight,
            )
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    BEiT backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class BeitBackbone(BackboneMixin, BeitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.beit = BeitModel(config, add_pooling_layer=False)
        self.fpn = BeitFPNNeck(config) if config.add_fpn else nn.Identity()

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/beit-base-patch16-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""
        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size
        kwargs["output_hidden_states"] = True  # required to extract per-stage feature maps from hidden_states
        outputs = self.beit(pixel_values, **kwargs)

        hidden_states = outputs.hidden_states
        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:, :]
                    hidden_state = hidden_state.transpose(1, 2)
                    hidden_state = hidden_state.reshape(batch_size, -1, patch_height, patch_width)

                feature_maps += (hidden_state,)

        feature_maps = self.fpn(feature_maps)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BeitForImageClassification",
    "BeitForMaskedImageModeling",
    "BeitForSemanticSegmentation",
    "BeitModel",
    "BeitPreTrainedModel",
    "BeitBackbone",
]
