# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from collections.abc import Callable
from typing import Optional, Union

import torch
import torchvision.transforms.v2.functional as tvF
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.models.beit.image_processing_beit_fast import BeitImageProcessorFast

from ...activations import ACT2FN
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...modeling_layers import DropPath, GradientCheckpointingLayer, drop_path_schedule
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..vit.modeling_vit import eager_attention_forward
from .configuration_segformer import SegformerConfig
from .image_processing_segformer import SegformerImageProcessorKwargs


class SegformerImageProcessorFast(BeitImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_reduce_labels = False
    do_center_crop = None
    crop_size = None

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[SegformerImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        batch_feature = self._preprocess(images, **images_kwargs)

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    # Nearest interpolation is used for segmentation maps instead of BILINEAR.
                    "interpolation": tvF.InterpolationMode.NEAREST_EXACT,
                }
            )
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            ).pixel_values
            batch_feature["labels"] = processed_segmentation_maps.squeeze(1).to(torch.int64)

        return batch_feature

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        interpolation: Optional["tvF.InterpolationMode"],
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: SizeDict,
        rescale_factor: float,
        image_mean: float | list[float],
        image_std: float | list[float],
        disable_grouping: bool,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:  # Return type can be list if return_tensors=None
        if do_reduce_labels:
            images = self.reduce_label(images)  # Apply reduction if needed

        # Group images by size for batched resizing
        resized_images = images
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                resized_stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
                resized_images_grouped[shape] = resized_stacked_images
            resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing (rescale/normalize)
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack images into a single tensor if return_tensors is set

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring
class SegFormerImageClassifierOutput(ImageClassifierOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Classification (or regression if config.num_labels==1) loss.
    logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
        Classification (or regression if config.num_labels==1) scores (before SoftMax).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
        called feature maps) of the model at the output of each stage.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
        sequence_length)`.

        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Overlapping patch embeddings via strided convolution with symmetric padding."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerAttention(nn.Module):
    """Efficient self-attention where keys/values are spatially reduced via strided convolution.

    Introduced in [PvT](https://huggingface.co/papers/2102.12122): queries attend to the full
    sequence while key/value tokens are downsampled, reducing the O(N²) attention cost.
    """

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.config = config
        self.num_attention_heads = num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.attention_dropout = config.attention_probs_dropout_prob
        self.hidden_dropout = config.hidden_dropout_prob

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Sequence reduction: reshape to spatial, apply strided conv, reshape back
        kv_hidden_states = hidden_states
        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            kv_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            kv_hidden_states = self.sr(kv_hidden_states)
            kv_hidden_states = kv_hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            kv_hidden_states = self.layer_norm(kv_hidden_states)

        kv_input_shape = kv_hidden_states.shape[:-1]
        kv_hidden_shape = (*kv_input_shape, -1, self.head_dim)
        key_states = self.k_proj(kv_hidden_states).view(kv_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(kv_hidden_states).view(kv_hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = nn.functional.dropout(attn_output, p=self.hidden_dropout, training=self.training)

        return attn_output, attn_weights


class SegformerDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class SegformerMixMLP(nn.Module):
    """Mix-FFN: fc1 → DWConv → activation → fc2.

    The depthwise convolution implicitly encodes positional information, replacing the explicit
    position embedding used in standard ViT/BeiT MLPs.
    """

    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.activation_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerLayer(GradientCheckpointingLayer):
    """Transformer block with DropPath on both branches and a MixFFN instead of a plain MLP."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixMLP(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.attention(self.layernorm_before(hidden_states), height, width, **kwargs)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.mlp(self.layernorm_after(hidden_states), height, width)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class SegformerStage(nn.Module):
    """One encoder stage: OverlapPatchEmbeddings → SegformerLayer blocks → LayerNorm."""

    def __init__(self, config, stage_idx: int, drop_path_decays: list[float]):
        super().__init__()
        depth_start = sum(config.depths[:stage_idx])
        # All stages reshape to (B, C, H, W); only the last stage skips it when reshape_last_stage=False.
        self.reshape = stage_idx < config.num_encoder_blocks - 1 or config.reshape_last_stage
        self.patch_embeddings = SegformerOverlapPatchEmbeddings(
            patch_size=config.patch_sizes[stage_idx],
            stride=config.strides[stage_idx],
            num_channels=config.num_channels if stage_idx == 0 else config.hidden_sizes[stage_idx - 1],
            hidden_size=config.hidden_sizes[stage_idx],
        )
        self.blocks = nn.ModuleList(
            [
                SegformerLayer(
                    config,
                    hidden_size=config.hidden_sizes[stage_idx],
                    num_attention_heads=config.num_attention_heads[stage_idx],
                    drop_path=drop_path_decays[depth_start + layer_idx],
                    sequence_reduction_ratio=config.sr_ratios[stage_idx],
                    mlp_ratio=config.mlp_ratios[stage_idx],
                )
                for layer_idx in range(config.depths[stage_idx])
            ]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[stage_idx])

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states, height, width = self.patch_embeddings(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, height, width, **kwargs)
        hidden_states = self.layer_norm(hidden_states)
        if self.reshape:
            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return hidden_states


class SegformerEncoder(nn.Module):
    """Hierarchical mix-transformer encoder with `num_encoder_blocks` stages.

    Stage outputs (shape `(B, C, H, W)`) are the multi-scale feature maps consumed by
    `SegformerDecodeHead`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        drop_path_decays = drop_path_schedule(config, num_layers=sum(config.depths))
        self.stages = nn.ModuleList(
            [SegformerStage(config, stage_idx, drop_path_decays) for stage_idx in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = pixel_values
        for stage in self.stages:
            hidden_states = stage(hidden_states, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class SegformerPreTrainedModel(PreTrainedModel):
    config: SegformerConfig
    base_model_prefix = "segformer"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = None
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        # capture_initial_input=False: stage 0's input is raw pixel values, not a meaningful embedding.
        "hidden_states": OutputRecorder(SegformerStage, capture_initial_input=False),
        "attentions": SegformerAttention,
    }


@auto_docstring
class SegformerModel(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        encoder_outputs = self.encoder(pixel_values, **kwargs)
        return encoder_outputs


@auto_docstring(
    custom_intro="""
    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
    states) e.g. for ImageNet.
    """
)
class SegformerForImageClassification(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.segformer = SegformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SegFormerImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.segformer(pixel_values, **kwargs)

        sequence_output = outputs.last_hidden_state

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SegformerMLP(nn.Module):
    """Projects each encoder stage's feature map to a common `decoder_hidden_size`."""

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        linear_projections = []
        for stage_idx in range(config.num_encoder_blocks):
            linear_proj = SegformerMLP(config, input_dim=config.hidden_sizes[stage_idx])
            linear_projections.append(linear_proj)
        self.linear_c = nn.ModuleList(linear_projections)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor, **kwargs) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, linear_proj in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = linear_proj(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


@auto_docstring(
    custom_intro="""
    SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.
    """
)
class SegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""
        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        # Track whether the user requested hidden states before we override the kwarg.
        output_hidden_states = kwargs.get("output_hidden_states", getattr(self.config, "output_hidden_states", False))

        # The decode head always needs all stage outputs, so force hidden_states on internally.
        kwargs["output_hidden_states"] = True
        outputs = self.segformer(pixel_values, **kwargs)

        encoder_hidden_states = outputs.hidden_states

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = [
    "SegformerImageProcessorFast",
    "SegformerDecodeHead",
    "SegformerForImageClassification",
    "SegformerForSemanticSegmentation",
    "SegformerLayer",
    "SegformerModel",
    "SegformerPreTrainedModel",
    "SegformerStage",
]
