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
"""PyTorch Swinv2 Transformer model."""

import collections.abc
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..swin.modeling_swin import (
    SwinDropPath,
    SwinEmbeddings,
    SwinEncoder,
    SwinEncoderOutput,
    SwinForImageClassification,
    SwinForMaskedImageModeling,
    SwinImageClassifierOutput,
    SwinLayer,
    SwinMaskedImageModelingOutput,
    SwinMLP,
    SwinModel,
    SwinModelOutput,
    SwinPatchEmbeddings,
    SwinPatchMerging,
    SwinPreTrainedModel,
    SwinStage,
    eager_attention_forward,
    window_partition,
    window_reverse,
)
from .configuration_swinv2 import Swinv2Config


logger = logging.get_logger(__name__)

# Swinv2PatchEmbeddings and Swinv2PatchMerging are from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py.


@auto_docstring(
    custom_intro="""
    Swinv2 encoder's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class Swinv2EncoderOutput(SwinEncoderOutput):
    pass


@auto_docstring(
    custom_intro="""
    Swinv2 model's outputs that also contains a pooling of the last hidden states.
    """
)
@dataclass
class Swinv2ModelOutput(SwinModelOutput):
    pass


@auto_docstring(
    custom_intro="""
    Swinv2 masked image model outputs.
    """
)
@dataclass
class Swinv2MaskedImageModelingOutput(SwinMaskedImageModelingOutput):
    pass


@auto_docstring(
    custom_intro="""
    Swinv2 outputs for image classification.
    """
)
@dataclass
class Swinv2ImageClassifierOutput(SwinImageClassifierOutput):
    pass


class Swinv2DropPath(SwinDropPath):
    pass


class Swinv2PatchEmbeddings(SwinPatchEmbeddings):
    pass


class Swinv2Embeddings(SwinEmbeddings):
    pass


class Swinv2PatchMerging(SwinPatchMerging):
    def __init__(self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__(dim)
        self.input_resolution = input_resolution
        self.dim = dim
        del self.norm
        self.norm = norm_layer(2 * dim)

    def forward(self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature = torch.cat(
            [input_feature[:, row::2, col::2, :] for col in range(2) for row in range(2)], dim=-1
        )
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature)

        return input_feature


class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        super().__init__()
        self.config = config
        self.num_attention_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_dropout = config.attention_probs_dropout_prob
        self.is_causal = False
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.pretrained_window_size = (
            pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size)
        )

        self.q_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(dim, dim)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        relative_coords_table, relative_position_index = self.create_coords_table_and_index()
        self.relative_coords_table = nn.Buffer(relative_coords_table, persistent=False)
        self.relative_position_index = nn.Buffer(relative_position_index, persistent=False)

    def _continuous_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            window_area, window_area, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return 16 * torch.sigmoid(relative_position_bias)

    def create_coords_table_and_index(self):
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= self.pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.pretrained_window_size[1] - 1
        elif self.window_size[0] > 1:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        )
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_coords_table, relative_position_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        normalize_eps = torch.finfo(query_states.dtype).eps
        query_states = nn.functional.normalize(query_states, dim=-1, eps=normalize_eps)
        key_states = nn.functional.normalize(key_states, dim=-1, eps=normalize_eps)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        query_states = query_states * logit_scale

        relative_position_bias = self._continuous_relative_position_bias()
        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            batch_size = input_shape[0] // num_windows
            seq_len = input_shape[1]
            attention_mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1, -1)
                .reshape(-1, 1, seq_len, seq_len)
            )
            combined_mask = relative_position_bias + attention_mask
        else:
            combined_mask = relative_position_bias

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Swinv2MLP(SwinMLP):
    pass


class Swinv2Layer(SwinLayer):
    def __init__(
        self, config, dim, input_resolution, num_heads, drop_path_rate=0.0, shift_size=0, pretrained_window_size=0
    ):
        super().__init__(config, dim, input_resolution, num_heads, drop_path_rate, shift_size)
        del self.attention
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape

        hidden_states_windows = window_partition(self.cyclic_shift(hidden_states), self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype, device=hidden_states_windows.device
        )

        attention_output, attn_weights = self.attention(hidden_states_windows, attn_mask, **kwargs)

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        attention_windows = self.cyclic_shift(
            window_reverse(attention_windows, self.window_size, height_pad, width_pad), reverse=True
        )

        if pad_values[3] > 0 or pad_values[5] > 0:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path(hidden_states)

        layer_output = self.dropout(self.mlp(hidden_states))
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        return layer_output, attn_weights


class Swinv2Stage(SwinStage):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, pretrained_window_size=0
    ):
        GradientCheckpointingLayer.__init__(self)
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                Swinv2Layer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    drop_path_rate=drop_path[i],
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_hidden_states_before_downsampling: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        height, width = input_dimensions
        last_attn_weights = None
        for layer_module in self.blocks:
            hidden_states, last_attn_weights = layer_module(hidden_states, input_dimensions, **kwargs)

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)

        reshaped_hidden_states = self.get_reshaped_hidden_states(
            hidden_states, hidden_states_before_downsampling, height, width, output_hidden_states_before_downsampling
        )

        return hidden_states, reshaped_hidden_states, last_attn_weights


@auto_docstring
class Swinv2PreTrainedModel(SwinPreTrainedModel):
    config: Swinv2Config
    base_model_prefix = "swinv2"
    _no_split_modules = ["Swinv2Stage"]
    _keys_to_ignore_on_load_unexpected = [
        r"relative_position_index",
        r"relative_coords_table",
    ]
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Swinv2Stage, index=0, capture_initial_hidden_state=True),
        "attentions": OutputRecorder(Swinv2Stage, index=2, capture_initial_hidden_state=False),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Swinv2Embeddings):
            if module.mask_token is not None:
                init.zeros_(module.mask_token)
            if module.position_embeddings is not None:
                init.zeros_(module.position_embeddings)
        elif isinstance(module, Swinv2Attention):
            init.constant_(module.logit_scale, math.log(10))
            relative_coords_table, relative_position_index = module.create_coords_table_and_index()
            init.copy_(module.relative_coords_table, relative_coords_table)
            init.copy_(module.relative_position_index, relative_position_index)


class Swinv2Encoder(SwinEncoder):
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__(config, grid_size)
        del self.layers
        if config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        dpr = [config.drop_path_rate * i / max(sum(config.depths) - 1, 1) for i in range(sum(config.depths))]
        self.layers = nn.ModuleList(
            [
                Swinv2Stage(
                    config=config,
                    dim=int(config.embed_dim * 2**layer_idx),
                    input_resolution=(grid_size[0] // (2**layer_idx), grid_size[1] // (2**layer_idx)),
                    depth=config.depths[layer_idx],
                    num_heads=config.num_heads[layer_idx],
                    drop_path=dpr[sum(config.depths[:layer_idx]) : sum(config.depths[: layer_idx + 1])],
                    downsample=Swinv2PatchMerging if (layer_idx < self.num_layers - 1) else None,
                    pretrained_window_size=pretrained_window_sizes[layer_idx],
                )
                for layer_idx in range(self.num_layers)
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
        output_hidden_states: bool = False,
        output_hidden_states_before_downsampling: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Swinv2EncoderOutput:
        r"""
        input_dimensions (`tuple[int, int]`):
            Spatial `(height, width)` of the patch grid entering the encoder.
        output_hidden_states_before_downsampling (`bool`, *optional*, defaults to `False`):
            If `True`, `reshaped_hidden_states` contains pre-downsampling feature maps.
        """
        all_reshaped_hidden_states = None
        if output_hidden_states:
            # Prepend the stem: hidden_states is the patch embedding output (B, N, C),
            # reshape it to spatial (B, C, H, W) as the first reshaped hidden state.
            batch_size, _, hidden_size = hidden_states.shape
            stem_spatial = (
                hidden_states.view(batch_size, *input_dimensions, hidden_size).permute(0, 3, 1, 2).contiguous()
            )
            all_reshaped_hidden_states = (stem_spatial,)

        for layer_module in self.layers:
            hidden_states, reshaped_hidden_state, _ = layer_module(
                hidden_states,
                input_dimensions,
                output_hidden_states_before_downsampling=output_hidden_states_before_downsampling,
                **kwargs,
            )
            if output_hidden_states:
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if layer_module.downsample is not None:
                input_dimensions = ((input_dimensions[0] + 1) // 2, (input_dimensions[1] + 1) // 2)

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class Swinv2Model(SwinModel):
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings


@auto_docstring(
    custom_intro="""
        Swinv2 Model with a decoder on top for masked image modeling, as proposed in
    [SimMIM](https://huggingface.co/papers/2111.09886).

        <Tip>

        Note that we provide a script to pre-train this model on custom data in our [examples
        directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

        </Tip>
    """
)
class Swinv2ForMaskedImageModeling(SwinForMaskedImageModeling):
    pass


@auto_docstring(
    custom_intro="""
    Swinv2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune SwinV2 on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """
)
class Swinv2ForImageClassification(SwinForImageClassification):
    accepts_loss_kwargs = False


@auto_docstring(
    custom_intro="""
    Swinv2 backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class Swinv2Backbone(BackboneMixin, Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        self.embeddings = Swinv2Embeddings(config)
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)

        # initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swinv2-tiny-patch4-window8-256", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        embedding_output, input_dimensions = self.embeddings(pixel_values)
        # filter_output_hidden_states already forces output_hidden_states=True — do NOT also pass it
        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_hidden_states_before_downsampling=True,
            **kwargs,
        )

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, outputs.reshaped_hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Swinv2Model",
    "Swinv2ForMaskedImageModeling",
    "Swinv2ForImageClassification",
    "Swinv2PreTrainedModel",
    "Swinv2Backbone",
]
