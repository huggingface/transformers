# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoModel
from ..sam3.configuration_sam3 import Sam3VisionConfig
from ..sam3.modeling_sam3 import (
    Sam3FPNLayer,
    Sam3SinePositionEmbedding,
    Sam3VisionEncoderOutput,
)
from ..sam3_tracker_video.configuration_sam3_tracker_video import (
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoMaskDecoderConfig,
    Sam3TrackerVideoPromptEncoderConfig,
)
from ..sam3_tracker_video.modeling_sam3_tracker_video import (
    NO_OBJ_SCORE,
    Sam3TrackerVideoAttention,
    Sam3TrackerVideoFeedForward,
    Sam3TrackerVideoImageSegmentationOutput,
    Sam3TrackerVideoInferenceCache,
    Sam3TrackerVideoInferenceSession,
    Sam3TrackerVideoLayerNorm,
    Sam3TrackerVideoMaskDecoder,
    Sam3TrackerVideoMaskDownSampler,
    Sam3TrackerVideoMaskDownSamplerLayer,
    Sam3TrackerVideoMaskEmbedding,
    Sam3TrackerVideoMemoryEncoder,
    Sam3TrackerVideoMemoryFuser,
    Sam3TrackerVideoMemoryFuserCXBlock,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoPositionalEmbedding,
    Sam3TrackerVideoPositionEmbeddingSine,
    Sam3TrackerVideoPreTrainedModel,
    Sam3TrackerVideoPromptEncoder,
    Sam3TrackerVideoRoPEAttention,
    Sam3TrackerVideoSegmentationOutput,
    Sam3TrackerVideoTwoWayAttentionBlock,
    Sam3TrackerVideoTwoWayTransformer,
    Sam3TrackerVideoVisionEncoderOutput,
    Sam3TrackerVideoVisionRotaryEmbedding,
    apply_rotary_pos_emb_2d,
    eager_attention_forward,
)
from ..sam3_tracker_video.processing_sam3_tracker_video import Sam3TrackerVideoProcessor


logger = logging.get_logger(__name__)


# =============================================================================
# Vision encoder (shared ViT backbone + tri-neck FPN producing three feature
# streams: SAM3 detection, interactive, and propagation).
# =============================================================================


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31VisionConfig(Sam3VisionConfig):
    r"""
    fpn_hidden_size (`int`, *optional*, defaults to 256):
        The hidden dimension of the FPN.
    backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
        The spatial sizes (height, width) of the feature maps from the backbone at different scales.
    scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0]`):
        Scale factors for FPN multi-scale features. SAM3.1 uses a three-level pyramid
        (4x, 2x, 1x upsampling) without the 0.5x downsampling level present in SAM3.
    """

    model_type = "sam3_1_vision_model"

    def __post_init__(self, **kwargs):
        if self.scale_factors is None:
            self.scale_factors = [4.0, 2.0, 1.0]
        if self.backbone_feature_sizes is None:
            self.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]
        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam3_vit_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = CONFIG_MAPPING["sam3_vit_model"]()
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@dataclass
class Sam31VisionEncoderOutput(Sam3VisionEncoderOutput):
    r"""
    sam3_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps from the SAM3 detection stream.
    sam3_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each level of the SAM3 detection stream.
    interactive_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps from the interactive (initial conditioning) stream.
    interactive_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each level of the interactive stream.
    propagation_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps from the propagation (memory) stream.
    propagation_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each level of the propagation stream.
    """

    fpn_hidden_states = AttributeError()
    fpn_position_encoding = AttributeError()
    sam3_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    sam3_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
    interactive_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    interactive_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
    propagation_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    propagation_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None


class Sam31SinePositionEmbedding(Sam3SinePositionEmbedding):
    pass


class Sam31FPNLayer(Sam3FPNLayer):
    pass


class Sam31VisionNeck(nn.Module):
    def __init__(self, config: Sam31VisionConfig):
        super().__init__()
        self.config = config

        self.position_encoding = Sam31SinePositionEmbedding(
            num_position_features=config.fpn_hidden_size // 2, normalize=True
        )

        self.sam3_fpn_layers = nn.ModuleList(
            [
                Sam31FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )
        self.interactive_fpn_layers = nn.ModuleList(
            [
                Sam31FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )
        self.propagation_fpn_layers = nn.ModuleList(
            [
                Sam31FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        sam3_fpn_hidden_states = ()
        sam3_fpn_position_encoding = ()
        interactive_fpn_hidden_states = ()
        interactive_fpn_position_encoding = ()
        propagation_fpn_hidden_states = ()
        propagation_fpn_position_encoding = ()

        for sam3_fpn_layer, interactive_fpn_layer, propagation_fpn_layer in zip(
            self.sam3_fpn_layers, self.interactive_fpn_layers, self.propagation_fpn_layers
        ):
            sam3_out = sam3_fpn_layer(hidden_states)
            sam3_fpn_hidden_states += (sam3_out,)
            sam3_fpn_position_encoding += (self.position_encoding(sam3_out.shape, sam3_out.device, sam3_out.dtype),)

            interactive_out = interactive_fpn_layer(hidden_states)
            interactive_fpn_hidden_states += (interactive_out,)
            interactive_fpn_position_encoding += (
                self.position_encoding(interactive_out.shape, interactive_out.device, interactive_out.dtype),
            )

            propagation_out = propagation_fpn_layer(hidden_states)
            propagation_fpn_hidden_states += (propagation_out,)
            propagation_fpn_position_encoding += (
                self.position_encoding(propagation_out.shape, propagation_out.device, propagation_out.dtype),
            )

        return (
            sam3_fpn_hidden_states,
            sam3_fpn_position_encoding,
            interactive_fpn_hidden_states,
            interactive_fpn_position_encoding,
            propagation_fpn_hidden_states,
            propagation_fpn_position_encoding,
        )


# =============================================================================
# Tracker scaffolding. Most subclasses are pass-throughs; the multiplex-aware
# overrides (mask downsampler with multiplex_count input channels, memory-attention
# layer with 256-channel cross-attention, multiplex mask decoder, and the decoupled
# memory cross-attention used by the standalone PVS model) follow below.
# =============================================================================


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31TrackerVideoPromptEncoderConfig(Sam3TrackerVideoPromptEncoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31TrackerVideoMaskDecoderConfig(Sam3TrackerVideoMaskDecoderConfig):
    r"""
    mlp_dim (`int`, *optional*, defaults to 2048):
        The dimension of the MLP in the two-way transformer.
    attention_downsample_rate (`int`, *optional*, defaults to 2):
        The downsample rate for the attention layers.
    num_multimask_outputs (`int`, *optional*, defaults to 3):
        The number of multimask outputs.
    iou_head_depth (`int`, *optional*, defaults to 3):
        The depth of the IoU head.
    iou_head_hidden_dim (`int`, *optional*, defaults to 256):
        The hidden dimension of the IoU head.
    dynamic_multimask_via_stability (`bool`, *optional*, defaults to `True`):
        Whether to use dynamic multimask via stability.
    dynamic_multimask_stability_delta (`float`, *optional*, defaults to 0.05):
        The stability delta for the dynamic multimask.
    dynamic_multimask_stability_thresh (`float`, *optional*, defaults to 0.98):
        The stability threshold for the dynamic multimask.
    multiplex_count (`int`, *optional*, defaults to 16):
        Number of masks multiplexed into a single decoder forward pass. Each multiplex slot
        receives its own iou / object-score / mask token bank.
    pred_obj_scores (`bool`, *optional*, defaults to `True`):
        Whether to predict per-object presence scores from a dedicated object-score token.
    pred_obj_scores_mlp (`bool`, *optional*, defaults to `True`):
        Whether to use an MLP (vs a linear layer) to compute object-presence scores.
    use_multimask_token_for_obj_ptr (`bool`, *optional*, defaults to `True`):
        Whether to return multimask tokens as object pointers (used by the memory encoder).
    multimask_outputs_only (`bool`, *optional*, defaults to `True`):
        If `True`, the decoder only emits `num_multimask_outputs` masks (without the extra
        single-mask token output).
    decode_mask_with_shared_tokens (`bool`, *optional*, defaults to `False`):
        Whether to reuse a single mask token bank across multimask outputs via separate
        hypernetwork projections.
    decode_mask_attribute_with_shared_tokens (`bool`, *optional*, defaults to `False`):
        Whether to predict iou / object-score attributes from the mask tokens themselves
        (without a dedicated iou / obj-score embedding bank).
    iou_prediction_use_sigmoid (`bool`, *optional*, defaults to `False`):
        Whether to apply a sigmoid to the predicted iou scores.
    """

    multiplex_count: int = 16
    pred_obj_scores: bool = True
    pred_obj_scores_mlp: bool = True
    use_multimask_token_for_obj_ptr: bool = True
    multimask_outputs_only: bool = True
    decode_mask_with_shared_tokens: bool = False
    decode_mask_attribute_with_shared_tokens: bool = False
    iou_prediction_use_sigmoid: bool = False


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31TrackerVideoConfig(Sam3TrackerVideoConfig):
    r"""
    prompt_encoder_config (Union[`dict`, `Sam31TrackerVideoPromptEncoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`Sam31TrackerVideoPromptEncoderConfig`].
    mask_decoder_config (Union[`dict`, `Sam31TrackerVideoMaskDecoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`Sam31TrackerVideoMaskDecoderConfig`].
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for parameter initialization.
    num_maskmem (`int`, *optional*, defaults to 7):
        The number of memory slots for the mask memory.
    sigmoid_scale_for_mem_enc (`float`, *optional*, defaults to 20.0):
        Scale factor for the sigmoid function in the memory encoder.
    sigmoid_bias_for_mem_enc (`float`, *optional*, defaults to -10.0):
        Bias for the sigmoid function in the memory encoder.
    enable_occlusion_spatial_embedding (`bool`, *optional*, defaults to `True`):
        Whether to enable spatial embedding for occlusions.
    multimask_output_in_sam (`bool`, *optional*, defaults to `True`):
        Whether to output multiple masks from the SAM head.
    multimask_min_pt_num (`int`, *optional*, defaults to 0):
        The minimum number of points to trigger multimask output.
    multimask_max_pt_num (`int`, *optional*, defaults to 1):
        The maximum number of points to trigger multimask output.
    multimask_output_for_tracking (`bool`, *optional*, defaults to `True`):
        Whether to use multimask output for tracking.
    max_object_pointers_in_encoder (`int`, *optional*, defaults to 16):
        The maximum number of object pointers in the encoder.
    max_cond_frame_num (`int`, *optional*, defaults to 4):
        Maximum number of conditioning frames to use in memory attention.
    enable_temporal_pos_encoding_for_object_pointers (`bool`, *optional*, defaults to `True`):
        Whether to enable temporal positional encoding for object pointers.
    memory_attention_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the memory attention hidden states.
    memory_attention_num_layers (`int`, *optional*, defaults to 4):
        The number of layers in the memory attention module.
    memory_attention_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the memory attention.
        Bumped from 1 (SAM3) to 8 in SAM3.1.
    memory_attention_downsample_rate (`int`, *optional*, defaults to 1):
        The downsample rate for the attention layers.
    memory_attention_feed_forward_hidden_size (`int`, *optional*, defaults to 2048):
        The dimension of the feedforward network in the memory attention module.
    memory_attention_feed_forward_hidden_act (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function in the feedforward network in the memory attention module.
    memory_attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout rate for the memory attention module.
    memory_attention_rope_theta (`float`, *optional*, defaults to 10000):
        The Rope theta parameter.
    memory_attention_rope_feat_sizes (`list[int]`, *optional*, defaults to `[72, 72]`):
        The feature sizes for the Rope positional encoding.
    memory_attention_rope_dropout (`float`, *optional*, defaults to 0.1):
        The dropout rate for the Rope positional encoding.
    memory_encoder_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the memory encoder hidden states.
    memory_encoder_output_channels (`int`, *optional*, defaults to 256):
        The number of output channels for the memory encoder. Bumped from 64 (SAM3) to 256
        in SAM3.1 to support richer per-slot memory features.
    mask_downsampler_embed_dim (`int`, *optional*, defaults to 256):
        The dimension of the mask downsampler embedding.
    mask_downsampler_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size for the mask downsampler.
    mask_downsampler_stride (`int`, *optional*, defaults to 2):
        The stride for the mask downsampler.
    mask_downsampler_padding (`int`, *optional*, defaults to 1):
        The padding for the mask downsampler.
    mask_downsampler_total_stride (`int`, *optional*, defaults to 16):
        The total stride for the mask downsampler.
    mask_downsampler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the mask downsampler.
    mask_downsampler_starting_out_chan (`int`, *optional*, defaults to 4):
        Initial output channel count for the first conv layer of the mask downsampler.
        Subsequent layers grow by `mask_downsampler_stride ** 2` per layer.
    mask_downsampler_input_channel_multiplier (`int`, *optional*, defaults to 2):
        Multiplier applied to `multiplex_count` to compute the number of input channels of
        the mask downsampler. The mask downsampler receives `multiplex_count *
        input_channel_multiplier` channels (e.g. raw + ranked masks).
    mask_downsampler_interpol_size (`list[int]`, *optional*, defaults to `[1152, 1152]`):
        Optional spatial size to bilinearly interpolate the input mask to before
        downsampling. The total stride is applied after this interpolation.
    memory_fuser_num_layers (`int`, *optional*, defaults to 2):
        The number of layers in the memory fuser.
    memory_fuser_embed_dim (`int`, *optional*, defaults to 256):
        The dimension of the embedding layer in the memory fuser.
    memory_fuser_intermediate_dim (`int`, *optional*, defaults to 1024):
        The dimension of the intermediate layer in the memory fuser.
    memory_fuser_kernel_size (`int`, *optional*, defaults to 7):
        The kernel size for the memory fuser.
    memory_fuser_padding (`int`, *optional*, defaults to 3):
        The padding for the memory fuser.
    memory_fuser_layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
        The initial value for the layer scale in the memory fuser.
    memory_fuser_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the memory fuser.
    multiplex_count (`int`, *optional*, defaults to 16):
        Number of multiplex slots maintained per object in the SAM3.1 tracker memory.
        Each slot stores an independent memory feature and pointer; the
        `Sam31MultiplexController` routes objects into these slots at runtime.
    add_output_suppression_embeddings (`bool`, *optional*, defaults to `True`):
        Per-slot embeddings added to multiplex mask tokens so padding slots are suppressed
        during propagation (Meta `add_output_suppression_embeddings`).

    Example:

    ```python
    >>> from transformers import (
    ...     Sam31VisionConfig,
    ...     Sam31TrackerVideoPromptEncoderConfig,
    ...     Sam31TrackerVideoMaskDecoderConfig,
    ...     Sam31TrackerVideoModel,
    ... )

    >>> # Initializing a Sam31TrackerVideoConfig with `"facebook/sam3.1"` style configuration
    >>> configuration = Sam31TrackerVideoConfig()

    >>> # Initializing a Sam31TrackerVideoModel (with random weights) from the configuration
    >>> model = Sam31TrackerVideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sam3_1_tracker_video"

    multiplex_count: int = 16
    add_output_suppression_embeddings: bool = True
    memory_attention_num_attention_heads: int = 8
    memory_encoder_output_channels: int = 256
    mask_downsampler_starting_out_chan: int = 4
    mask_downsampler_input_channel_multiplier: int = 2
    mask_downsampler_interpol_size: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.mask_downsampler_interpol_size is None:
            self.mask_downsampler_interpol_size = [1152, 1152]
        self.memory_attention_rope_feat_sizes = (
            [72, 72] if self.memory_attention_rope_feat_sizes is None else self.memory_attention_rope_feat_sizes
        )

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_1_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_1_vision_model"](
                backbone_feature_sizes=[[288, 288], [144, 144], [72, 72]]
            )

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam31TrackerVideoPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam31TrackerVideoPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam31TrackerVideoMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam31TrackerVideoMaskDecoderConfig(multiplex_count=self.multiplex_count)

        self.image_size = kwargs.pop("image_size", 1008)
        PreTrainedConfig.__post_init__(self, **kwargs)


class Sam31TrackerVideoProcessor(Sam3TrackerVideoProcessor):
    """SAM3.1 video processor with optional Meta-style relative point coordinates."""

    def add_inputs_to_inference_session(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int] | int,
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = None,
        input_labels: list[list[list[int]]] | torch.Tensor | None = None,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = None,
        input_masks: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor] | None = None,
        original_size: tuple[int, int] | None = None,
        clear_old_inputs: bool = True,
        rel_coordinates: bool = False,
    ) -> Sam31TrackerVideoInferenceSession:
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]

        if (input_points is not None) != (input_labels is not None):
            raise ValueError("points and labels must be provided together")
        if input_points is None and input_boxes is None and input_masks is None:
            raise ValueError("at least one of points, boxes, or masks must be provided as input")
        if input_masks is not None and (input_points is not None or input_boxes is not None):
            raise ValueError("masks cannot be provided together with points or boxes")

        if input_masks is not None:
            return self.process_new_mask_for_video_frame(inference_session, frame_idx, obj_ids, input_masks)
        return self.process_new_points_or_boxes_for_video_frame(
            inference_session,
            frame_idx,
            obj_ids,
            input_points,
            input_labels,
            input_boxes,
            original_size,
            clear_old_inputs,
            rel_coordinates=rel_coordinates,
        )

    def process_new_points_or_boxes_for_video_frame(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = None,
        input_labels: list[list[list[int]]] | torch.Tensor | None = None,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = None,
        original_size: tuple[int, int] | None = None,
        clear_old_inputs: bool = True,
        rel_coordinates: bool = False,
    ) -> Sam31TrackerVideoInferenceSession:
        if original_size is not None:
            inference_session.video_height = original_size[0]
            inference_session.video_width = original_size[1]
        elif inference_session.video_height is None or inference_session.video_width is None:
            raise ValueError("original_size must be provided when adding points or boxes on a first streamed frame")

        original_sizes = [[inference_session.video_height, inference_session.video_width]]

        if rel_coordinates:
            if input_boxes is not None:
                raise ValueError(
                    "rel_coordinates=True is only supported for point prompts. "
                    "Use pixel-space boxes with rel_coordinates=False (default)."
                )
            processed_points = self._validate_single_input(
                input_points,
                expected_depth=4,
                input_name="points",
                expected_format="[image level, object level, point level, point coordinates]",
                expected_coord_size=2,
            )
            processed_labels = self._validate_single_input(
                input_labels,
                expected_depth=3,
                input_name="labels",
                expected_format="[image level, object level, point level]",
            )
            points_max_dims = self._get_nested_dimensions(processed_points)[:3]
            labels_max_dims = self._get_nested_dimensions(processed_labels)[:3]
            if points_max_dims != labels_max_dims:
                raise ValueError(
                    "Input points and labels have inconsistent dimensions. Please ensure they have the same dimensions."
                )
            padded_points = self._pad_nested_list(processed_points, points_max_dims + [2])
            final_points = torch.tensor(padded_points, dtype=torch.float32)
            mask = final_points != self.point_pad_value
            coord_mask = mask.all(dim=-1, keepdim=True)
            scale = float(self.target_size)
            final_points = torch.where(coord_mask.expand_as(final_points), final_points * scale, final_points)
            padded_labels = self._pad_nested_list(processed_labels, labels_max_dims)
            final_labels = torch.tensor(padded_labels, dtype=torch.int64)
            input_points = final_points
            input_labels = final_labels
            input_boxes = None
        else:
            encoded_inputs = self(
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                original_sizes=original_sizes,
                return_tensors="pt",
            )
            input_points = encoded_inputs.get("input_points", None)
            input_labels = encoded_inputs.get("input_labels", None)
            input_boxes = encoded_inputs.get("input_boxes", None)

        if input_points is not None:
            if input_points.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of points ({input_points.shape[1]})"
                )
        else:
            input_points = torch.zeros(1, len(obj_ids), 0, 2, dtype=torch.float32)
        if input_labels is not None:
            if input_labels.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of labels ({input_labels.shape[1]})"
                )
        else:
            input_labels = torch.zeros(1, len(obj_ids), 0, dtype=torch.int32)
        if input_boxes is not None:
            if input_boxes.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of boxes ({input_boxes.shape[1]})"
                )

        if input_boxes is not None:
            if not clear_old_inputs:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            box_coords = input_boxes.reshape(1, -1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32).repeat(1, box_coords.shape[1], 1)
            input_points = torch.cat([box_coords, input_points], dim=2)
            input_labels = torch.cat([box_labels, input_labels], dim=2)

        for obj_id, idx in zip(obj_ids, range(len(obj_ids))):
            obj_idx = inference_session.obj_id_to_idx(obj_id)
            input_points_for_obj = input_points[:, idx, :, :].unsqueeze(1)
            input_labels_for_obj = input_labels[:, idx, :].unsqueeze(1)
            if not clear_old_inputs:
                existing_points = inference_session.point_inputs_per_obj[obj_idx].get(frame_idx, None)
                if existing_points is not None:
                    input_points_for_obj = torch.cat(
                        [existing_points["point_coords"].to(input_points_for_obj.device), input_points_for_obj], dim=2
                    )
                    input_labels_for_obj = torch.cat(
                        [existing_points["point_labels"].to(input_labels_for_obj.device), input_labels_for_obj], dim=2
                    )
            point_inputs = {
                "point_coords": input_points_for_obj,
                "point_labels": input_labels_for_obj,
            }

            inference_session.add_point_inputs(obj_idx, frame_idx, point_inputs)
            inference_session.remove_mask_inputs(obj_idx, frame_idx)

        merged_ids = list(dict.fromkeys([*inference_session.obj_with_new_inputs, *obj_ids]))
        inference_session.obj_with_new_inputs = merged_ids

        return inference_session


# =============================================================================
# Multiplexing primitives. `Sam31MultiplexController` is a stateless `nn.Module`
# that returns a `Sam31MultiplexState`; the state captures the assignment of
# objects to (bucket, slot) coordinates and exposes mux / demux operations that
# convert tensors between data space (batch_size, channels, ...) and multiplex
# space (num_buckets, multiplex_count, channels, ...).
# =============================================================================


_PADDING_NUM = -1
_REMOVED_NUM = -1116


class Sam31MultiplexState:
    """Holds the assignment of objects to multiplex buckets and slots.

    A state with `num_buckets` buckets, each of capacity `multiplex_count`, is built from
    `assignments: list[list[int]]` where each inner list of length `multiplex_count` stores
    either valid object indices (non-negative) or `_PADDING_NUM` for empty slots /
    `_REMOVED_NUM` for slots whose object has been removed. The state exposes precomputed
    permutation matrices (`mux_matrix`, `demux_matrix`) so `mux` / `demux` operations are
    plain matmuls.
    """

    def __init__(
        self,
        assignments: list[list[int]],
        device: torch.device,
        dtype: torch.dtype,
        allowed_bucket_capacity: int,
        *,
        object_ids: list[int] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(self, assignments: list[list[int]], *, object_ids: list[int] | None = None):
        self.assignments = assignments
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(self.assignments[0])
        if not all(len(self.assignments[i]) == self.multiplex_count for i in range(self.num_buckets)):
            raise ValueError("All buckets must have the same `multiplex_count`")

        self.total_valid_entries = sum(sum(1 for x in bucket if x >= 0) for bucket in self.assignments)
        self.total_non_padding_entries = sum(
            sum(1 for x in bucket if x != _PADDING_NUM) for bucket in self.assignments
        )

        self.object_ids = object_ids
        if self.object_ids is not None and len(self.object_ids) != self.total_valid_entries:
            raise ValueError("`object_ids` must map 1:1 to the valid entries")

        seen_idxs = set()
        for bucket in self.assignments:
            valid_in_bucket = sum(1 for x in bucket if x != _PADDING_NUM)
            if valid_in_bucket > self.allowed_bucket_capacity:
                raise ValueError(
                    f"Bucket holds {valid_in_bucket} entries (> allowed_bucket_capacity="
                    f"{self.allowed_bucket_capacity})"
                )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    if obj_idx >= self.total_non_padding_entries:
                        raise ValueError(f"Object index {obj_idx} >= {self.total_non_padding_entries}")
                    if obj_idx in seen_idxs:
                        raise ValueError("Object indices must be unique")
                    seen_idxs.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        return self.num_buckets * self.allowed_bucket_capacity - self.total_non_padding_entries

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        if num_objects <= 0:
            raise ValueError(f"num_objects={num_objects} must be positive")
        if not allow_new_buckets and self.available_slots < num_objects:
            raise ValueError(f"not enough available slots {self.available_slots} < {num_objects}")
        return list(range(self.total_valid_entries, self.total_valid_entries + num_objects))

    def add_objects(
        self,
        object_indices: list[int],
        *,
        object_ids: list[int] | None = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        if len(object_indices) == 0:
            return
        object_indices = object_indices.copy()
        if (object_ids is None) != (self.object_ids is None):
            raise ValueError("`object_ids` must either be always given or always omitted")
        if object_ids is not None:
            if len(object_ids) != len(object_indices):
                raise ValueError("`object_ids` must have the same length as `object_indices`")
            object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        if object_indices != sorted(object_indices):
            raise ValueError("`object_indices` must be sorted")
        object_indices.reverse()
        if object_ids is not None:
            object_ids.reverse()
        if prefer_new_buckets and not allow_new_buckets:
            raise ValueError("prefer_new_buckets requires allow_new_buckets")

        def _pop_next():
            idx = object_indices.pop()
            if object_ids is not None and self.object_ids is not None:
                self.object_ids.append(object_ids.pop())
            return idx

        if not prefer_new_buckets:
            for bucket in self.assignments:
                for i in range(self.allowed_bucket_capacity):
                    if bucket[i] == _PADDING_NUM:
                        bucket[i] = _pop_next()
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(f"Cannot place objects {list(reversed(object_indices))} without creating new buckets")

        while len(object_indices) > 0:
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for i in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[i] = _pop_next()
            self.assignments.append(new_bucket)

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        if self.total_valid_entries != original_num_entries + num_new_objects:
            raise RuntimeError(f"{self.total_valid_entries=} != {original_num_entries=} + {num_new_objects=}")

    def remove_objects(self, object_indices: list[int], strict: bool = True):
        object_indices = object_indices.copy()
        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    self.assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)
        if strict and len(object_indices) > 0:
            raise ValueError(f"Failed to remove objects: {object_indices}")

        buckets_to_keep = []
        buckets_to_remove = []
        for bucket_idx, bucket in enumerate(self.assignments):
            if all(obj_id in (_PADDING_NUM, _REMOVED_NUM) for obj_id in bucket):
                buckets_to_remove.append(bucket_idx)
            else:
                buckets_to_keep.append(bucket_idx)
        for bucket_idx in reversed(buckets_to_remove):
            del self.assignments[bucket_idx]

        if len(buckets_to_keep) == 0:
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        # Remap surviving positive object ids to be consecutive.
        all_positive_ids = sorted({obj_id for bucket in self.assignments for obj_id in bucket if obj_id >= 0})
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_positive_ids)}
        for bucket in self.assignments:
            for i, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[i] = id_mapping[obj_id]
        if self.object_ids is not None:
            new_object_ids = [None] * len(all_positive_ids)
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]
            self.object_ids = new_object_ids

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def _precompute_transition_matrices(self, device: torch.device, dtype: torch.dtype):
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count,
            self.total_valid_entries,
            device=device,
            dtype=dtype,
        )
        self.demux_matrix = torch.zeros(
            self.total_valid_entries,
            self.num_buckets * self.multiplex_count,
            device=device,
            dtype=dtype,
        )
        for i in range(self.num_buckets):
            for j in range(self.multiplex_count):
                bucket_slot = i * self.multiplex_count + j
                object_idx = self.assignments[i][j]
                if object_idx >= 0:
                    self.mux_matrix[bucket_slot, object_idx] = 1.0
                    self.demux_matrix[object_idx, bucket_slot] = 1.0

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (total_valid_entries, ...) → (num_buckets, multiplex_count, ...)."""
        num_valid_entries = x.shape[0]
        if num_valid_entries != self.total_valid_entries:
            raise ValueError(f"{num_valid_entries=} != {self.total_valid_entries=}")
        output_shape = (self.num_buckets, self.multiplex_count) + tuple(x.shape[1:])
        result = self.mux_matrix @ x.reshape(num_valid_entries, -1)
        return result.view(output_shape)

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (num_buckets, multiplex_count, ...) → (total_valid_entries, ...)."""
        num_buckets, multiplex_count = x.shape[:2]
        if num_buckets != self.num_buckets or multiplex_count != self.multiplex_count:
            raise ValueError(f"{num_buckets=} or {multiplex_count=} mismatch")
        output_shape = (self.total_valid_entries,) + tuple(x.shape[2:])
        result = self.demux_matrix @ x.reshape(num_buckets * multiplex_count, -1)
        return result.view(output_shape)

    def get_valid_object_mask(self) -> torch.Tensor:
        """Return a `(num_buckets, multiplex_count)` boolean mask of valid (non-padding) slots."""
        valid_mask = self.mux_matrix.sum(dim=1) > 0
        return valid_mask.reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self) -> set[int]:
        return {obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0}


class Sam31MultiplexController(nn.Module):
    """Builds `Sam31MultiplexState`s that bucket up to `multiplex_count` objects per bucket.

    The controller carries no learnable parameters; it only owns the bucketing policy
    (capacity, optional full-shuffle vs in-bucket shuffle, eval-time capacity override).
    """

    def __init__(
        self,
        multiplex_count: int,
        full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ):
        super().__init__()
        if multiplex_count < 1:
            raise ValueError(f"multiplex_count must be >= 1, got {multiplex_count}")
        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        self.eval_multiplex_count = multiplex_count if eval_multiplex_count < 0 else eval_multiplex_count

    @property
    def allowed_bucket_capacity(self) -> int:
        return self.multiplex_count if self.training else self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        device: torch.device,
        dtype: torch.dtype,
        random: bool = True,
        *,
        object_ids: list[int] | None = None,
    ) -> Sam31MultiplexState:
        allowed_bucket_capacity = self.allowed_bucket_capacity
        true_bucket_capacity = self.multiplex_count
        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)

        if self.full_shuffle:
            ids = torch.cat(
                [
                    torch.arange(num_valid_entries, dtype=torch.long),
                    torch.tensor(
                        [_PADDING_NUM] * (num_buckets * true_bucket_capacity - num_valid_entries),
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            if random:
                ids = ids[torch.randperm(ids.shape[0], dtype=torch.long)]
            assignments = [
                ids[i * true_bucket_capacity : (i + 1) * true_bucket_capacity].tolist() for i in range(num_buckets)
            ]
        else:
            ids = torch.randperm(num_valid_entries, dtype=torch.int64) if random else torch.arange(num_valid_entries)
            total_elements = num_buckets * allowed_bucket_capacity
            if ids.shape[0] < total_elements:
                ids = torch.cat([ids, torch.tensor([_PADDING_NUM] * (total_elements - ids.shape[0]))])
            assignments = [
                ids[i * allowed_bucket_capacity : (i + 1) * allowed_bucket_capacity].tolist()
                + [_PADDING_NUM] * (true_bucket_capacity - allowed_bucket_capacity)
                for i in range(num_buckets)
            ]

        return Sam31MultiplexState(
            assignments,
            device,
            dtype,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids,
        )


class Sam31TrackerVideoInferenceCache(Sam3TrackerVideoInferenceCache):
    pass


class Sam31TrackerVideoInferenceSession(Sam3TrackerVideoInferenceSession):
    pass


class Sam31TrackerVideoLayerNorm(Sam3TrackerVideoLayerNorm):
    pass


class Sam31TrackerVideoPositionEmbeddingSine(Sam3TrackerVideoPositionEmbeddingSine):
    pass


class Sam31TrackerVideoAttention(Sam3TrackerVideoAttention):
    pass


class Sam31TrackerVideoTwoWayAttentionBlock(Sam3TrackerVideoTwoWayAttentionBlock):
    pass


class Sam31TrackerVideoFeedForward(Sam3TrackerVideoFeedForward):
    pass


class Sam31TrackerVideoImageSegmentationOutput(Sam3TrackerVideoImageSegmentationOutput):
    pass


class Sam31TrackerVideoSegmentationOutput(Sam3TrackerVideoSegmentationOutput):
    pass


class Sam31TrackerVideoPreTrainedModel(Sam3TrackerVideoPreTrainedModel):
    pass


class Sam31TrackerVideoVisionRotaryEmbedding(Sam3TrackerVideoVisionRotaryEmbedding):
    pass


class Sam31TrackerVideoRoPEAttention(Sam3TrackerVideoRoPEAttention):
    pass


class Sam31TrackerVideoSimpleRoPEAttention(nn.Module):
    """Pre-projected 2D-RoPE attention block.

    Takes already-projected q / k / v tensors, splits heads, applies axial 2D RoPE,
    runs scaled dot-product attention, and returns the merged-heads output. No
    q / k / v / out projections live on this module — they belong to the parent
    decoupled memory-attention layer, which needs them externalized in order to
    sum image-stream and token-stream contributions before RoPE.
    """

    def __init__(self, config, rope_k_repeat: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.memory_attention_hidden_size
        self.internal_dim = self.hidden_size // config.memory_attention_downsample_rate
        self.num_attention_heads = config.memory_attention_num_attention_heads
        self.head_dim = self.internal_dim // self.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.rope_k_repeat = rope_k_repeat
        self.dropout_p = config.memory_attention_rope_dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, q_len = query.shape[:2]
        head_shape = (batch_size, -1, self.num_attention_heads, self.head_dim)

        query = query.reshape(*head_shape).transpose(1, 2)
        key = key.reshape(*head_shape).transpose(1, 2)
        value = value.reshape(*head_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_2d(
            query, key, cos, sin, repeat_freqs_k=self.rope_k_repeat, num_k_exclude_rope=num_k_exclude_rope
        )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0 if not self.training else self.dropout_p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        return attn_output.reshape(batch_size, q_len, self.num_attention_heads * self.head_dim).contiguous()


class Sam31TrackerVideoMemoryAttentionLayer(nn.Module):
    """Single decoupled memory cross-attention layer.

    Performs self-attention over `tgt` (object tokens), then a *decoupled* cross
    attention whose queries combine `image` and `tgt` (`image_cross_attn_q_proj(image) +
    cross_attn_q_proj(tgt)`) and whose keys combine `memory_image` and `memory`
    (`image_cross_attn_k_proj(memory_image) + cross_attn_k_proj(memory)`), and finally
    a feed-forward block. Values come from the `memory` token stream alone. The `image`
    input flows through the layer unchanged — it is consumed only as a side input for
    the cross-attention queries.

    Positional injection follows SAM3.1's `_create_multiplex_transformer`:
    `pos_enc_at_attn=False`, `pos_enc_at_cross_attn_queries=False`,
    `pos_enc_at_cross_attn_keys=True`, `cross_attention_first=False`, `pre_norm=True`.
    These choices do not affect parameter shapes and are hard-wired here.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.memory_attention_hidden_size
        self.hidden_size = hidden_size

        self.self_attn_q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_out_proj = nn.Linear(hidden_size, hidden_size)

        self.cross_attn_q_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_k_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_v_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_out_proj = nn.Linear(hidden_size, hidden_size)

        self.image_cross_attn_q_proj = nn.Linear(hidden_size, hidden_size)
        self.image_cross_attn_k_proj = nn.Linear(hidden_size, hidden_size)

        self.self_attention_rope = Sam31TrackerVideoSimpleRoPEAttention(config, rope_k_repeat=False)
        self.cross_attention_rope = Sam31TrackerVideoSimpleRoPEAttention(config, rope_k_repeat=True)

        self.linear1 = nn.Linear(hidden_size, config.memory_attention_feed_forward_hidden_size)
        self.dropout = nn.Dropout(config.memory_attention_dropout)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(config.memory_attention_dropout)
        self.dropout2 = nn.Dropout(config.memory_attention_dropout)
        self.dropout3 = nn.Dropout(config.memory_attention_dropout)

        self.activation = ACT2FN[config.memory_attention_feed_forward_hidden_act]

    def _forward_self_attention(
        self, tgt: torch.Tensor, rope_position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        tgt2 = self.layer_norm1(tgt)
        q = self.self_attn_q_proj(tgt2)
        k = self.self_attn_k_proj(tgt2)
        v = self.self_attn_v_proj(tgt2)
        out = self.self_attention_rope(q, k, v, position_embeddings=rope_position_embeddings)
        return tgt + self.dropout1(self.self_attn_out_proj(out))

    def _forward_cross_attention(
        self,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        memory_image_pos: torch.Tensor,
        rope_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt2 = self.layer_norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        out = self.cross_attention_rope(
            q, k, v, position_embeddings=rope_position_embeddings, num_k_exclude_rope=num_k_exclude_rope
        )
        return tgt + self.dropout2(self.cross_attn_out_proj(out))

    def forward(
        self,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        memory_image_pos: torch.Tensor,
        rope_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt = self._forward_self_attention(tgt, rope_position_embeddings)
        tgt = self._forward_cross_attention(
            image=image,
            tgt=tgt,
            memory_image=memory_image,
            memory=memory,
            memory_image_pos=memory_image_pos,
            rope_position_embeddings=rope_position_embeddings,
            num_k_exclude_rope=num_k_exclude_rope,
        )

        tgt2 = self.layer_norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return image, tgt


class Sam31TrackerVideoMemoryAttention(nn.Module):
    """Decoupled memory cross-attention encoder for the SAM3.1 PVS tracker.

    Stacks `config.memory_attention_num_layers` decoupled layers. Inputs follow the
    SAM3.1 convention: `image` / `src` are the *current* frame's image and token
    features (with `src` typically `image` expanded along the multiplex-bucket
    dimension), and `memory_image` / `memory` are their *past* analogs (concatenated
    across selected memory frames). The `image` stream is consumed as a side-input
    inside each layer but is not propagated to the encoder's output; the returned
    tensor is the layer-normed updated `tgt`, in sequence-first layout.
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Sam31TrackerVideoMemoryAttentionLayer(config) for _ in range(config.memory_attention_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.memory_attention_hidden_size)
        self.rotary_emb = Sam31TrackerVideoVisionRotaryEmbedding(config=config)

    def forward(
        self,
        image: torch.Tensor,
        src: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor,
        src_pos: torch.Tensor,
        memory_image_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        num_object_pointer_tokens: int = 0,
    ) -> torch.Tensor:
        r"""
        Args:
            image (`torch.FloatTensor` of shape `(seq_len, batch_size, hidden)`):
                Current frame image features (shared across multiplex buckets).
            src (`torch.FloatTensor` of shape `(seq_len, batch_size * num_buckets, hidden)`):
                Current frame token features (image features expanded per bucket).
            memory_image (`torch.FloatTensor` of shape `(memory_len, batch_size, hidden)`):
                Past-frame image features (concatenated across memory frames).
            memory (`torch.FloatTensor` of shape `(memory_len, batch_size * num_buckets, hidden)`):
                Past-frame token features (concatenated across memory frames + obj-ptr tokens).
            image_pos (`torch.FloatTensor`): Position encoding for `image`.
            src_pos (`torch.FloatTensor`): Position encoding for `src`. Added to `src` as
                `pos_enc_at_input` (scaled by 0.1), following SAM3 / SAM3.1.
            memory_image_pos (`torch.FloatTensor`): Position encoding for `memory_image`.
                Injected into cross-attention keys.
            memory_pos (`torch.FloatTensor`): Position encoding for `memory` (currently
                unused — kept for signature symmetry with the original repo).
            num_object_pointer_tokens (`int`, *optional*, defaults to 0):
                Number of trailing object-pointer tokens in the memory stream that should
                be excluded from RoPE (their positional information comes from a separate
                temporal pos-enc).
        """
        # Inject positional encoding into the current-frame token features at the
        # encoder input, scaled by 0.1, matching `pos_enc_at_input=True` in the
        # original `TransformerEncoderDecoupledCrossAttention`.
        output = src + 0.1 * src_pos if src_pos is not None else src

        # Switch sequence-first → batch-first to match the layer's expected layout.
        image_bf = image.transpose(0, 1)
        output_bf = output.transpose(0, 1)
        memory_image_bf = memory_image.transpose(0, 1)
        memory_bf = memory.transpose(0, 1)
        memory_image_pos_bf = memory_image_pos.transpose(0, 1)
        memory_pos_bf = memory_pos.transpose(0, 1) if memory_pos is not None else None

        # `memory` carries `num_object_pointer_tokens` extra object-pointer tokens at the
        # tail that have no spatial / image counterpart. To keep the decoupled cross-attn
        # sum `image_cross_attn_k_proj(memory_image) + cross_attn_k_proj(memory)`
        # well-shaped, pad `memory_image` with zero tokens at the matching positions and
        # adopt the temporal positional encoding from the corresponding tail slice of
        # `memory_pos` (broadcast over the batch).
        if num_object_pointer_tokens > 0 and memory_image_bf.shape[1] != memory_bf.shape[1]:
            seq_diff = memory_bf.shape[1] - memory_image_bf.shape[1]
            if seq_diff != num_object_pointer_tokens:
                raise ValueError(
                    f"Expected memory to have {num_object_pointer_tokens} extra tokens vs "
                    f"memory_image, got a difference of {seq_diff}."
                )
            obj_ptr_pad = memory_image_bf.new_zeros(
                memory_image_bf.shape[0], num_object_pointer_tokens, memory_image_bf.shape[2]
            )
            memory_image_bf = torch.cat([memory_image_bf, obj_ptr_pad], dim=1)
            if memory_pos_bf is not None:
                # The original repo notes that `tpos is the same across the batch`, so we
                # pick a single batch element of `memory_pos` for the trailing obj-ptr
                # positions and broadcast it across all buckets.
                obj_ptr_pos = memory_pos_bf[0:1, -num_object_pointer_tokens:].expand(
                    memory_image_pos_bf.shape[0], -1, -1
                )
                memory_image_pos_bf = torch.cat([memory_image_pos_bf, obj_ptr_pos], dim=1)

        rope_position_embeddings = self.rotary_emb()

        for layer in self.layers:
            image_bf, output_bf = layer(
                image=image_bf,
                tgt=output_bf,
                memory_image=memory_image_bf,
                memory=memory_bf,
                memory_image_pos=memory_image_pos_bf,
                rope_position_embeddings=rope_position_embeddings,
                num_k_exclude_rope=num_object_pointer_tokens,
            )

        normed_output = self.layer_norm(output_bf).transpose(0, 1)
        return normed_output


class Sam31TrackerVideoMemoryFuserCXBlock(Sam3TrackerVideoMemoryFuserCXBlock):
    pass


class Sam31TrackerVideoMemoryFuser(Sam3TrackerVideoMemoryFuser):
    pass


class Sam31TrackerVideoMaskDownSamplerLayer(Sam3TrackerVideoMaskDownSamplerLayer):
    pass


class Sam31TrackerVideoMaskDownSampler(Sam3TrackerVideoMaskDownSampler):
    """Multiplex-aware mask downsampler.

    Differences from `Sam3TrackerVideoMaskDownSampler`:
      - Input channels = `multiplex_count * mask_downsampler_input_channel_multiplier`
        (the downsampler ingests one mask per slot per multiplexed batch).
      - First conv layer outputs `mask_downsampler_starting_out_chan` channels (was 1 in
        SAM3); subsequent layers grow by `stride ** 2`.
      - Optional bilinear interpolation of the input to `mask_downsampler_interpol_size`
        before any conv layer is applied.
    """

    def __init__(self, config):
        nn.Module.__init__(self)

        num_layers = int(math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride))
        in_channels = config.multiplex_count * config.mask_downsampler_input_channel_multiplier
        out_channels = config.mask_downsampler_starting_out_chan

        self.layers = nn.ModuleList()
        self.activation = ACT2FN[config.mask_downsampler_hidden_act]
        for _ in range(num_layers):
            out_channels = out_channels * (config.mask_downsampler_stride**2)
            self.layers.append(Sam31TrackerVideoMaskDownSamplerLayer(config, in_channels, out_channels))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(out_channels, config.mask_downsampler_embed_dim, kernel_size=1)

        self.interpol_size = (
            list(config.mask_downsampler_interpol_size) if config.mask_downsampler_interpol_size is not None else None
        )

    def forward(self, x):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class Sam31TrackerVideoMemoryEncoder(Sam3TrackerVideoMemoryEncoder):
    pass


class Sam31TrackerVideoVisionEncoderOutput(Sam3TrackerVideoVisionEncoderOutput):
    pass


class Sam31TrackerVideoPositionalEmbedding(Sam3TrackerVideoPositionalEmbedding):
    pass


class Sam31TrackerVideoMaskEmbedding(Sam3TrackerVideoMaskEmbedding):
    pass


class Sam31TrackerVideoPromptEncoder(Sam3TrackerVideoPromptEncoder):
    def get_dense_pe(self) -> torch.Tensor:
        """Dense PE for the interactive SAM mask decoder."""
        h, w = self.image_embedding_size
        device = self.shared_embedding.positional_embedding.device
        dtype = self.shared_embedding.positional_embedding.dtype
        grid = torch.ones((h, w), device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self.shared_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1).unsqueeze(0)


class Sam31TrackerVideoTwoWayTransformer(Sam3TrackerVideoTwoWayTransformer):
    pass


class Sam31TrackerVideoMaskDecoder(Sam3TrackerVideoMaskDecoder):
    def __init__(self, config: Sam31TrackerVideoMaskDecoderConfig):
        super().__init__(config)
        self.iou_prediction_head = Sam31TrackerVideoFeedForward(
            self.hidden_size,
            config.iou_head_hidden_dim,
            self.num_mask_tokens,
            config.iou_head_depth,
            sigmoid_output=config.iou_prediction_use_sigmoid,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        high_resolution_features: list[torch.Tensor],
        attention_similarity: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                The embeddings from the image encoder.
            image_positional_embeddings (`torch.Tensor`):
                Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes.
            dense_prompt_embeddings (`torch.Tensor`):
                The embeddings of the mask inputs.
            multimask_output (`bool`):
                Whether to return multiple masks or a single mask.
            high_resolution_features (`list[torch.Tensor]`, *optional*):
                The high-resolution features from the vision encoder.
            attention_similarity (`torch.Tensor`, *optional*):
                The attention similarity tensor.
            target_embedding (`torch.Tensor`, *optional*):
                The target embedding.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat(
            [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            dim=0,
        )
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.shape[0] != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Match Meta `sam3/sam/mask_decoder.py::predict_masks` with `repeat_image=True`:
        # `repeat_interleave(image_embeddings, …)` first, then add dense prompts.
        image_embeddings = torch.repeat_interleave(image_embeddings, point_batch_size, dim=0)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, dim=0)
        # Run the transformer
        point_embeddings, image_embeddings = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )
        iou_token_out = point_embeddings[:, :, 1, :]
        mask_tokens_out = point_embeddings[:, :, 2 : (2 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).view(
            batch_size * point_batch_size, num_channels, height, width
        )

        feat_s0, feat_s1 = high_resolution_features
        feat_s0 = feat_s0.repeat_interleave(point_batch_size, dim=0)
        feat_s1 = feat_s1.repeat_interleave(point_batch_size, dim=0)
        upscaled_embedding = self.upscale_conv1(image_embeddings) + feat_s1
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding) + feat_s0)

        hyper_in_list: list[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.view(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).view(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(point_embeddings[:, :, 0, :])

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
            masks = masks[:, :, mask_slice, :, :]
            iou_pred = iou_pred[:, :, mask_slice]
        elif self.dynamic_multimask_via_stability and not self.training:
            mask_slice = slice(0, 1)
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            mask_slice = slice(0, 1)
            masks = masks[:, :, mask_slice, :, :]
            iou_pred = iou_pred[:, :, mask_slice]

        sam_tokens_out = mask_tokens_out[:, :, mask_slice]  # [b, 3, c] shape

        return masks, iou_pred, sam_tokens_out, object_score_logits


class Sam31TrackerVideoMultiplexMaskDecoder(nn.Module):
    """Multiplex mask decoder used for tracker propagation.

    Predicts masks for `multiplex_count` independent objects per bucket in a single forward
    pass. Compared to the standard `Sam31TrackerVideoMaskDecoder` (used for interactive
    clicks), this decoder maintains separate iou / object-score / mask token banks for each
    multiplex slot and emits multi-object outputs of shape
    `(batch_size, multiplex_count, num_mask_output_per_object, height, width)`.

    The decoder takes already-conditioned image embeddings (the bucket-level memory) and an
    optional `extra_per_object_embeddings` tensor `(batch_size, multiplex_count, hidden_size)`
    that is added to the mask tokens so the network can identify which slot each prediction
    belongs to.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.multiplex_count = config.multiplex_count
        self.num_multimask_outputs = config.num_multimask_outputs
        self.multimask_outputs_only = config.multimask_outputs_only
        self.decode_mask_with_shared_tokens = config.decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = config.decode_mask_attribute_with_shared_tokens
        self.pred_obj_scores = config.pred_obj_scores
        self.use_multimask_token_for_obj_ptr = config.use_multimask_token_for_obj_ptr

        if self.decode_mask_with_shared_tokens and not self.multimask_outputs_only:
            raise ValueError("multimask_outputs_only must be True if decode_mask_with_shared_tokens")

        self.num_mask_output_per_object = (
            self.num_multimask_outputs if self.multimask_outputs_only else self.num_multimask_outputs + 1
        )
        self.num_mask_tokens = (
            self.multiplex_count
            if self.decode_mask_with_shared_tokens
            else self.multiplex_count * self.num_mask_output_per_object
        )

        if not self.decode_mask_attribute_with_shared_tokens:
            self.iou_token = nn.Embedding(self.multiplex_count, self.hidden_size)
            if self.pred_obj_scores:
                self.obj_score_token = nn.Embedding(self.multiplex_count, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = Sam31TrackerVideoTwoWayTransformer(config)

        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = Sam31TrackerVideoLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        if self.num_multimask_outputs == 0:
            self.output_hypernetworks_mlp = Sam31TrackerVideoFeedForward(
                self.hidden_size, self.hidden_size, self.hidden_size // 8, 3
            )
        else:
            self.output_hypernetworks_mlps = nn.ModuleList(
                [
                    Sam31TrackerVideoFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)
                    for _ in range(self.num_mask_output_per_object)
                ]
            )

        iou_output_dim = (
            1
            if (self.decode_mask_attribute_with_shared_tokens and not self.decode_mask_with_shared_tokens)
            else self.num_mask_output_per_object
        )
        self.iou_prediction_head = Sam31TrackerVideoFeedForward(
            self.hidden_size,
            config.iou_head_hidden_dim,
            iou_output_dim,
            config.iou_head_depth,
            sigmoid_output=config.iou_prediction_use_sigmoid,
        )

        self.conv_s0 = nn.Conv2d(self.hidden_size, self.hidden_size // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(self.hidden_size, self.hidden_size // 4, kernel_size=1, stride=1)

        if self.pred_obj_scores:
            if config.pred_obj_scores_mlp:
                self.pred_obj_score_head = Sam31TrackerVideoFeedForward(self.hidden_size, self.hidden_size, 1, 3)
            else:
                self.pred_obj_score_head = nn.Linear(self.hidden_size, 1)

        self.dynamic_multimask_via_stability = config.dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = config.dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = config.dynamic_multimask_stability_thresh

    def _predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        high_resolution_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = image_embeddings.shape[0]
        compute_dtype = self.mask_tokens.weight.dtype
        image_embeddings = image_embeddings.to(dtype=compute_dtype)
        image_positional_embeddings = image_positional_embeddings.to(dtype=compute_dtype)
        if high_resolution_features is not None:
            high_resolution_features = [feat.to(dtype=compute_dtype) for feat in high_resolution_features]
        if extra_per_object_embeddings is not None:
            extra_per_object_embeddings = extra_per_object_embeddings.to(dtype=compute_dtype)

        token_list = []
        if self.pred_obj_scores and not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.obj_score_token.weight)
        if not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.iou_token.weight)
        attribute_tokens = torch.cat(token_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1)

        if extra_per_object_embeddings is not None:
            mask_tokens = (
                self.mask_tokens.weight.view(1, self.multiplex_count, self.num_mask_output_per_object, -1)
                .expand(batch_size, -1, -1, -1)
                .clone()
            )
            mask_tokens = mask_tokens + extra_per_object_embeddings.unsqueeze(2)
            mask_tokens = mask_tokens.flatten(1, 2)
        else:
            mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

        tokens = torch.cat([attribute_tokens, mask_tokens], dim=1)

        if image_positional_embeddings.size(0) != 1:
            raise ValueError("`image_positional_embeddings` must have a batch size of 1.")
        pos_src = image_positional_embeddings.repeat_interleave(tokens.shape[0], dim=0)

        # `Sam31TrackerVideoTwoWayTransformer` carries an extra "point_batch" dim on the
        # token stream and expects 4D image inputs of shape `(B, C, H, W)`. We use
        # `point_batch=1` here since the multiplex decoder runs a single decode per bucket.
        point_embeddings = tokens.unsqueeze(1)
        point_embeddings, keys = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=pos_src,
            attention_similarity=None,
        )
        point_embeddings = point_embeddings.squeeze(1)
        keys = keys.squeeze(1)

        if self.decode_mask_attribute_with_shared_tokens:
            iou_token_out = mask_tokens_out_flat = point_embeddings[:, : self.num_mask_tokens]
            obj_score_token_out = mask_tokens_out_flat if self.pred_obj_scores else None
        else:
            offset = 0
            if self.pred_obj_scores:
                obj_score_token_out = point_embeddings[:, offset : offset + self.multiplex_count]
                offset += self.multiplex_count
            else:
                obj_score_token_out = None
            iou_token_out = point_embeddings[:, offset : offset + self.multiplex_count]
            offset += self.multiplex_count
            mask_tokens_out_flat = point_embeddings[:, offset : offset + self.num_mask_tokens]

        # `keys` exits the two-way transformer in `(B, num_tokens, C)` shape; reshape back to
        # spatial `(B, C, H, W)` for upscaling. We pick H = W from the input image embeddings.
        height, width = image_embeddings.shape[-2:]
        src_2d = keys.transpose(1, 2).reshape(batch_size, self.hidden_size, height, width)

        if high_resolution_features is not None:
            feat_s0, feat_s1 = high_resolution_features
            upscaled = self.activation(self.upscale_layer_norm(self.upscale_conv1(src_2d) + feat_s1))
            upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)
        else:
            upscaled = self.activation(self.upscale_layer_norm(self.upscale_conv1(src_2d)))
            upscaled = self.activation(self.upscale_conv2(upscaled))

        if self.decode_mask_with_shared_tokens:
            mask_tokens_out = mask_tokens_out_flat.view(batch_size, self.multiplex_count, 1, -1)
        else:
            mask_tokens_out = mask_tokens_out_flat.view(
                batch_size, self.multiplex_count, self.num_mask_output_per_object, -1
            )

        if self.num_multimask_outputs == 0:
            hyper_in = self.output_hypernetworks_mlp(mask_tokens_out[:, :, 0, :]).unsqueeze(2)
        else:
            hyper_in_list: list[torch.Tensor] = []
            for i in range(self.num_mask_output_per_object):
                source_slice = (
                    mask_tokens_out[:, :, 0, :] if self.decode_mask_with_shared_tokens else mask_tokens_out[:, :, i, :]
                )
                hyper_in_list.append(self.output_hypernetworks_mlps[i](source_slice))
            hyper_in = torch.stack(hyper_in_list, dim=2)

        _, channels, h_up, w_up = upscaled.shape
        masks = torch.bmm(hyper_in.flatten(1, 2), upscaled.view(batch_size, channels, h_up * w_up)).view(
            batch_size, self.multiplex_count, self.num_mask_output_per_object, h_up, w_up
        )
        iou_pred = self.iou_prediction_head(iou_token_out).view(batch_size, self.multiplex_count, -1)

        if self.pred_obj_scores:
            if self.decode_mask_attribute_with_shared_tokens and not self.decode_mask_with_shared_tokens:
                object_score_logits = (
                    self.pred_obj_score_head(obj_score_token_out)
                    .view(batch_size, self.multiplex_count, self.num_mask_output_per_object)
                    .sum(-1, keepdim=True)
                )
            else:
                object_score_logits = self.pred_obj_score_head(obj_score_token_out)
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], iou_pred.shape[1])

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        multimask_output: bool,
        high_resolution_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_multimask_outputs <= 0 and multimask_output:
            raise ValueError(f"multimask_output must be False with {self.num_multimask_outputs=}")
        if self.multimask_outputs_only and not multimask_output:
            raise ValueError(f"multimask_output must be True with {self.multimask_outputs_only=}")

        out = self._predict_masks(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_resolution_features=high_resolution_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )
        masks = out["masks"]
        iou_pred = out["iou_pred"]
        mask_tokens_out = out["mask_tokens_out"]
        object_score_logits = out["object_score_logits"]

        if multimask_output:
            if not self.multimask_outputs_only:
                masks = masks[:, :, 1:, :, :]
                iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1, :, :]
            iou_pred = iou_pred[:, :, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out if self.multimask_outputs_only else mask_tokens_out[:, :, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, :, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: torch.Tensor) -> torch.Tensor:
        mask_logits = mask_logits.flatten(-2)
        delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(
        self, all_mask_logits: torch.Tensor, all_iou_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, multiplex_count = all_mask_logits.shape[:2]
        all_mask_logits = all_mask_logits.flatten(0, 1)
        all_iou_scores = all_iou_scores.flatten(0, 1)

        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_idx = torch.argmax(multimask_iou_scores, dim=-1)
        batch_idx = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_idx, best_idx].unsqueeze(1)
        best_multimask_iou = multimask_iou_scores[batch_idx, best_idx].unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou = all_iou_scores[:, 0:1]
        is_stable = self._get_stability_scores(singlemask_logits) >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou),
            singlemask_iou,
            best_multimask_iou,
        )

        return (
            mask_logits_out.unflatten(0, (batch_size, multiplex_count)),
            iou_scores_out.unflatten(0, (batch_size, multiplex_count)),
        )


# `Sam31VisionModel` inherits from the locally defined `Sam31TrackerVideoPreTrainedModel`
# (rather than from `Sam3VisionModel`) so that the modular converter does not pull the
# entire SAM3 detector/ViT hierarchy into this file as a transitive dependency.
@auto_docstring(
    custom_intro="""
    The SAM3.1 vision encoder: a shared ViT backbone followed by three independent FPN
    necks (SAM3 detection, interactive, and propagation streams), each producing
    multi-scale feature maps. The interactive and propagation streams are consumed by
    the PVS tracker; the SAM3 stream is consumed by the PCS detector.
    """
)
class Sam31VisionModel(Sam31TrackerVideoPreTrainedModel):
    config_class = Sam31VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Sam31VisionConfig):
        super().__init__(config)
        self.config = config
        self.backbone = AutoModel.from_config(config.backbone_config)
        self.neck = Sam31VisionNeck(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam31VisionEncoderOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        backbone_output = self.backbone(pixel_values, **kwargs)
        hidden_states = backbone_output.last_hidden_state

        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone_config.patch_size
        width = pixel_values.shape[-1] // self.config.backbone_config.patch_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        (
            sam3_fpn_hidden_states,
            sam3_fpn_position_encoding,
            interactive_fpn_hidden_states,
            interactive_fpn_position_encoding,
            propagation_fpn_hidden_states,
            propagation_fpn_position_encoding,
        ) = self.neck(hidden_states_spatial)

        return Sam31VisionEncoderOutput(
            last_hidden_state=hidden_states,
            sam3_fpn_hidden_states=sam3_fpn_hidden_states,
            sam3_fpn_position_encoding=sam3_fpn_position_encoding,
            interactive_fpn_hidden_states=interactive_fpn_hidden_states,
            interactive_fpn_position_encoding=interactive_fpn_position_encoding,
            propagation_fpn_hidden_states=propagation_fpn_hidden_states,
            propagation_fpn_position_encoding=propagation_fpn_position_encoding,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )


class Sam31TrackerVideoModel(Sam3TrackerVideoModel):
    """SAM3.1 PVS-only video tracker.

    Extends `Sam3TrackerVideoModel` with three multiplex components used during memory-
    conditioned propagation:
      * `multiplex_controller` (`Sam31MultiplexController`) — stateless bucketing policy
        that maps propagating objects onto `(bucket, slot)` pairs.
      * `propagation_mask_decoder` (`Sam31TrackerVideoMultiplexMaskDecoder`) — predicts
        masks for `multiplex_count` slots per bucket in a single forward pass.
      * Multiplex-aware memory encoder (inherited via `self.memory_encoder` + the
        `Sam31TrackerVideoMaskDownSampler` override) — consumes `multiplex_count`-stacked
        masks per bucket and emits one bucket-level memory feature map.

    The standard `self.mask_decoder` (interactive variant inherited from SAM3) is still used
    for the per-object click / box / mask path on initial conditioning frames, mirroring
    the dual-decoder design of the original SAM3.1 multiplex tracker.
    """

    def __init__(self, config: Sam31TrackerVideoConfig, remove_vision_encoder: bool = False):
        r"""
        remove_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to skip allocating the vision encoder (useful when the encoder is shared
            externally, e.g. when SAM3.1 is wired into a `Sam31VideoModel`).
        """
        # The parent (`Sam3TrackerVideoModel`) defines the full tracker scaffolding (vision
        # encoder, prompt encoder, interactive mask decoder, memory encoder, etc.). We
        # inline its body here so we can register the SAM3.1-specific multiplex components
        # alongside it. All references below use `Sam31TrackerVideo*` directly because the
        # modular converter only renames classes in inheritance positions, not inside
        # method bodies.
        if hasattr(config, "tracker_config") and config.tracker_config is not None:
            tracker_config = config.tracker_config
            if isinstance(tracker_config, dict):
                tracker_config = Sam31TrackerVideoConfig(**tracker_config)
            config = tracker_config
        Sam31TrackerVideoPreTrainedModel.__init__(self, config)
        self.shared_image_embedding = Sam31TrackerVideoPositionalEmbedding(config.prompt_encoder_config)
        self.vision_encoder = AutoModel.from_config(config.vision_config) if not remove_vision_encoder else None
        self.prompt_encoder = Sam31TrackerVideoPromptEncoder(config.prompt_encoder_config)
        # The module using it is not a PreTrainedModel subclass so we need this
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = Sam31TrackerVideoMaskDecoder(config.mask_decoder_config)

        self.backbone_feature_sizes = config.vision_config.backbone_feature_sizes
        self.hidden_dim = config.vision_config.fpn_hidden_size
        self.no_memory_embedding = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.config = config
        self.image_size = config.image_size
        self.memory_attention = Sam31TrackerVideoMemoryAttention(config)
        self.memory_encoder = Sam31TrackerVideoMemoryEncoder(config)
        self.no_memory_positional_encoding = torch.nn.Parameter(
            torch.zeros(1, 1, config.vision_config.fpn_hidden_size)
        )
        self.mem_dim = config.memory_encoder_output_channels
        self.num_maskmem = config.num_maskmem
        self.memory_temporal_positional_encoding = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )

        # Meta SAM3.1 uses `use_linear_no_obj_ptr=True`, replacing the legacy constant
        # `no_obj_ptr` parameter with a learned `Linear(hidden_dim, hidden_dim)` applied to
        # the predicted object pointer when the slot is "not appearing".
        self.no_obj_ptr_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # Kept as a zero buffer for backward compat with older converted checkpoints.
        self.no_object_pointer = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        # Meta SAM3.1 keeps two distinct 3-layer FF projections for the object pointer:
        # one for multiplex propagation (`obj_ptr_proj` → `object_pointer_proj`) and one for
        # interactive conditioning frames (`interactive_obj_ptr_proj` →
        # `interactive_object_pointer_proj`). Earlier conversions discarded the interactive
        # head and reused the propagation projection at conditioning time, which feeds the
        # wrong cond-frame pointer into every subsequent propagation step's memory attention
        # and accumulates drift over long horizons.
        self.object_pointer_proj = Sam31TrackerVideoFeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.interactive_object_pointer_proj = Sam31TrackerVideoFeedForward(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
        )

        if self.config.enable_temporal_pos_encoding_for_object_pointers:
            self.temporal_positional_encoding_projection_layer = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.temporal_positional_encoding_projection_layer = torch.nn.Identity()

        # Meta SAM3.1 keeps a per-slot `no_obj_embed_spatial` of shape
        # `(multiplex_count, mem_dim)` and applies it slot-wise; see
        # `_encode_new_memory` for the full rationale.
        self.occlusion_spatial_embedding_parameter = None
        if config.enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = torch.nn.Parameter(
                torch.zeros(config.multiplex_count, self.mem_dim)
            )

        # SAM3.1-specific multiplex components.
        # `eval_multiplex_count=1` puts a single object per bucket at eval time, mirroring
        # Meta `Sam3VideoTrackingMultiplexDemo.add_new_points`, which always passes
        # `prefer_new_buckets=True` when handling a click on a new object (see
        # `facebook_sam3/sam3/model/video_tracking_multiplex_demo.py::add_new_points`,
        # `prefer_new_buckets_local = True`). Per-obj buckets give every object its own
        # memory column during propagation / consolidation: each object's cond-frame
        # `maskmem_features` are encoded from that single object's mask only (matching
        # Meta's preflight), and the propagation memory attention's per-bucket memory /
        # object-pointer lookup (which always reads the first valid slot in each bucket)
        # naturally picks up each object's own memory rather than a joint multi-object
        # encoding. With the default capacity HF packs multiple objects into a single
        # bucket and ends up encoding / reading a joint memory, drifting away from Meta
        # over long propagations (mismatches the per-object memory keys Meta's
        # `track_step` sees). `_prepare_memory_conditioned_features_batched_for_propagation`
        # processes all buckets in a single batched `memory_attention` call (matching Meta)
        # — with PyTorch's default SDPA backend (FlashAttention / memory-efficient) this is
        # equivalent in peak memory to per-bucket sequential calls and materially faster.
        self.multiplex_controller = Sam31MultiplexController(
            multiplex_count=config.multiplex_count, eval_multiplex_count=1
        )
        propagation_mask_decoder_config = copy.deepcopy(config.mask_decoder_config)
        # Match Meta `VideoTrackingMultiplex`: `sam_mask_decoder` runs with
        # `dynamic_multimask_via_stability=False` while `interactive_sam_mask_decoder` keeps it True.
        propagation_mask_decoder_config.dynamic_multimask_via_stability = False
        self.propagation_mask_decoder = Sam31TrackerVideoMultiplexMaskDecoder(propagation_mask_decoder_config)

        self.output_valid_embed = None
        self.output_invalid_embed = None
        if getattr(config, "add_output_suppression_embeddings", True):
            self.output_valid_embed = torch.nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
            self.output_invalid_embed = torch.nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
            torch.nn.init.trunc_normal_(self.output_valid_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.output_invalid_embed, std=0.02)

        self.post_init()

    def get_image_wide_positional_embeddings(self) -> torch.Tensor:
        """Propagation SAM dense PE (Meta `image_pe_layer` / `get_propagation_dense_pe`)."""
        size = self.prompt_encoder.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        grid = torch.ones(size, device=target_device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / float(size[0])
        x_embed = x_embed / float(size[1])
        coords = torch.stack([x_embed, y_embed], dim=-1).to(self.shared_image_embedding.positional_embedding.dtype)
        positional_embedding = self.shared_image_embedding(coords)
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _scalar_object_score_logits(scores: torch.Tensor) -> torch.Tensor:
        """One logit per object, shape ``(1,)``, for stacking across objects."""
        return scores.reshape(-1).amax(dim=0, keepdim=True)

    @staticmethod
    def _batch_object_score_logits(scores: torch.Tensor) -> torch.Tensor:
        """Shape ``(1, 1)`` for per-object rows in memory-encoding ``torch.cat``."""
        return scores.reshape(-1).amax(dim=0, keepdim=True).reshape(1, 1)

    def _prune_stale_tracker_outputs(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        reverse: bool = False,
    ) -> None:
        """Drop non-conditioning frame outputs outside the temporal memory window.

        Propagation reads at most ``num_maskmem - 1`` prior non-conditioning frames via
        ``_gather_memory_frame_outputs``; older ``non_cond_frame_outputs`` entries are never
        used again. Conditioning frames are left intact (Meta keeps them for fallback lookups).
        After memory encoding, ``high_res_masks`` are stripped from retained entries since only
        ``maskmem_features`` / ``pred_masks`` are needed going forward.
        """
        if self.num_maskmem <= 0:
            return

        mem_span = self.num_maskmem - 1

        for obj_idx in range(inference_session.get_obj_num()):
            output_dict = inference_session.output_dict_per_obj[obj_idx]
            non_cond = output_dict["non_cond_frame_outputs"]
            stale_non_cond = [
                f
                for f in non_cond
                if (not reverse and f < frame_idx - mem_span) or (reverse and f > frame_idx + mem_span)
            ]
            for f in stale_non_cond:
                non_cond.pop(f, None)

            for stored in list(non_cond.values()) + list(output_dict["cond_frame_outputs"].values()):
                if isinstance(stored, dict) and stored.get("maskmem_features") is not None:
                    stored.pop("high_res_masks", None)

    def _blend_no_object_pointer(
        self,
        object_pointer: torch.Tensor,
        lambda_is_obj_appearing: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the predicted pointer with the `no_obj_ptr_linear` fallback.

        Mirrors Meta SAM3.1's `use_linear_no_obj_ptr=True` blend:
        ``obj_ptr = lambda * obj_ptr + (1 - lambda) * no_obj_ptr_linear(obj_ptr)``.
        See `modeling_sam3_1_tracker_video.py::_blend_no_object_pointer` for the full
        explanation of why the constant-pointer fallback inherited from SAM3 is no longer
        used on the SAM3.1 path.
        """
        weight = lambda_is_obj_appearing.to(object_pointer.dtype)
        projected = self.no_obj_ptr_linear(object_pointer)
        return weight * object_pointer + (1.0 - weight) * projected

    def _use_mask_as_output(
        self,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        mask_inputs: torch.Tensor,
    ) -> Sam31TrackerVideoImageSegmentationOutput:
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in forward above).
        """
        # Use -10/+20 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.to(backbone_features[0].dtype)

        # Ensure mask is at self.image_size resolution for consistency
        if mask_inputs_float.shape[-2:] != (self.image_size, self.image_size):
            mask_inputs_float = F.interpolate(
                mask_inputs_float.float(),
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(mask_inputs.dtype)

        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks.float(),
            size=self.prompt_encoder.mask_input_size,
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        ).to(backbone_features[0].dtype)
        # a dummy IoU prediction of all 1's under mask input
        iou_scores = mask_inputs.new_ones(mask_inputs.size(0), 1).to(backbone_features[0].dtype)
        # produce an object pointer using the SAM decoder from the mask input
        object_pointer = self._single_frame_forward(
            input_masks=self.mask_downsample(mask_inputs_float.to(backbone_features[0].dtype)),
            image_embeddings=high_res_features + [backbone_features],
        ).object_pointer
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.to(backbone_features[0].dtype)
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        object_pointer = self._blend_no_object_pointer(object_pointer, lambda_is_obj_appearing)
        return Sam31TrackerVideoImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_masks,
            high_res_masks=high_res_masks,
            object_pointer=object_pointer,
            object_score_logits=object_score_logits.unsqueeze(-1),
            image_embeddings=high_res_features + [backbone_features],
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam31TrackerVideoVisionEncoderOutput:
        r"""
        Run the SAM3.1 TriNeck vision encoder and adapt the propagation stream into the
        tracker-format `Sam31TrackerVideoVisionEncoderOutput` consumed by the rest of the
        memory pipeline.

        Differences from `Sam3TrackerVideoModel.get_image_features`:
          - Reads `propagation_fpn_hidden_states` / `propagation_fpn_position_encoding`
            from `Sam31VisionEncoderOutput` (the TriNeck's memory branch) instead of the
            single `fpn_hidden_states` stream.
          - Keeps all three TriNeck levels (no `[:-1]` slice): the SAM3.1 TriNeck emits
            three pyramid levels (4x, 2x, 1x), all of which feed the tracker.
        """
        vision_outputs: Sam31VisionEncoderOutput = self.vision_encoder(pixel_values, return_dict=True, **kwargs)

        feature_maps = list(vision_outputs.propagation_fpn_hidden_states)
        feature_maps_position_embeddings = list(vision_outputs.propagation_fpn_position_encoding)

        feature_maps[0] = self.propagation_mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.propagation_mask_decoder.conv_s1(feature_maps[1])

        feature_maps = [feat.flatten(2).permute(2, 0, 1) for feat in feature_maps]
        feature_maps_position_embeddings = [pe.flatten(2).permute(2, 0, 1) for pe in feature_maps_position_embeddings]

        return Sam31TrackerVideoVisionEncoderOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            fpn_hidden_states=feature_maps,
            fpn_position_encoding=feature_maps_position_embeddings,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def _tri_neck_to_tracker_fpn(
        self, vision_outputs: Sam31VisionEncoderOutput, stream: str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert TriNeck `Sam31VisionEncoderOutput` to tracker-style flattened FPN lists."""
        if stream == "propagation":
            feature_maps = list(vision_outputs.propagation_fpn_hidden_states)
            feature_maps_position_embeddings = list(vision_outputs.propagation_fpn_position_encoding)
            feature_maps[0] = self.propagation_mask_decoder.conv_s0(feature_maps[0])
            feature_maps[1] = self.propagation_mask_decoder.conv_s1(feature_maps[1])
        elif stream == "interactive":
            feature_maps = list(vision_outputs.interactive_fpn_hidden_states)
            feature_maps_position_embeddings = list(vision_outputs.interactive_fpn_position_encoding)
            feature_maps[0] = self.mask_decoder.conv_s0(feature_maps[0])
            feature_maps[1] = self.mask_decoder.conv_s1(feature_maps[1])
        else:
            raise ValueError(f"stream must be 'propagation' or 'interactive', got {stream!r}")

        feature_maps = [feat.flatten(2).permute(2, 0, 1) for feat in feature_maps]
        feature_maps_position_embeddings = [pe.flatten(2).permute(2, 0, 1) for pe in feature_maps_position_embeddings]
        return feature_maps, feature_maps_position_embeddings

    def _prepare_vision_features(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Prepare propagation + interactive FPN lists for one frame (cached per frame_idx)."""
        cached_features = inference_session.cache.get_vision_features(frame_idx)
        if cached_features is not None and "interactive_vision_feats" not in cached_features:
            cached_features = None

        if cached_features is not None:
            vision_feats = cached_features["vision_feats"]
            vision_pos_embeds = cached_features["vision_pos_embeds"]
            interactive_vision_feats = cached_features["interactive_vision_feats"]
            interactive_vision_pos_embeds = cached_features["interactive_vision_pos_embeds"]
        else:
            image_batch = inference_session.get_frame(frame_idx).unsqueeze(0)
            vision_outputs: Sam31VisionEncoderOutput = self.vision_encoder(image_batch, return_dict=True)
            vision_feats, vision_pos_embeds = self._tri_neck_to_tracker_fpn(vision_outputs, "propagation")
            interactive_vision_feats, interactive_vision_pos_embeds = self._tri_neck_to_tracker_fpn(
                vision_outputs, "interactive"
            )
            inference_session.cache.cache_vision_features(
                frame_idx,
                {
                    "vision_feats": vision_feats,
                    "vision_pos_embeds": vision_pos_embeds,
                    "interactive_vision_feats": interactive_vision_feats,
                    "interactive_vision_pos_embeds": interactive_vision_pos_embeds,
                },
            )

        if batch_size > 1:
            vision_feats = [v.expand(batch_size, -1, -1) for v in vision_feats]
            vision_pos_embeds = [pe.expand(batch_size, -1, -1) for pe in vision_pos_embeds]
            interactive_vision_feats = [v.expand(batch_size, -1, -1) for v in interactive_vision_feats]
            interactive_vision_pos_embeds = [pe.expand(batch_size, -1, -1) for pe in interactive_vision_pos_embeds]

        return vision_feats, vision_pos_embeds, interactive_vision_feats, interactive_vision_pos_embeds

    def _single_frame_forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_points: torch.FloatTensor | None = None,
        input_labels: torch.LongTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_masks: torch.LongTensor | None = None,
        image_embeddings: torch.FloatTensor | None = None,
        multimask_output: bool = True,
        attention_similarity: torch.FloatTensor | None = None,
        target_embedding: torch.FloatTensor | None = None,
        use_propagation_dense_pe: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam31TrackerVideoImageSegmentationOutput:
        """
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box
        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decoder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        use_propagation_dense_pe (`bool`, *optional*, defaults to `False`):
            If `True`, positional encodings come from the propagation `image_pe_layer` weights. If `False`, they come
            from the interactive prompt encoder (Meta `interactive_sam_prompt_encoder.get_dense_pe()`), consistent
            with point-prompt embeddings.
        """
        if not ((pixel_values is None) ^ (image_embeddings is None)):
            raise ValueError("Exactly one of pixel_values or image_embeddings must be provided.")
        if input_points is not None and input_boxes is not None:
            if input_points.shape[1] != input_boxes.shape[1]:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {input_points.shape[1]} and {input_boxes.shape[1]}."
                )
        elif input_points is not None:
            num_objects = input_points.shape[1]
        elif input_boxes is not None:
            num_objects = input_boxes.shape[1]
        elif input_masks is not None:
            num_objects = input_masks.shape[1]
        else:
            num_objects = 1

        if use_propagation_dense_pe:
            image_positional_embeddings = self.get_image_wide_positional_embeddings()
        else:
            image_positional_embeddings = self.prompt_encoder.get_dense_pe()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings[-1].shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, return_dict=True, **kwargs)
            feature_maps = image_outputs.fpn_hidden_states
            vision_hidden_states = image_outputs.hidden_states
            vision_attentions = image_outputs.attentions

            # add no memory embedding to the last feature map
            feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding

            # reshape feature maps to the same shape as the backbone feature sizes
            image_embeddings = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(feature_maps, self.backbone_feature_sizes)
            ]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is None and input_boxes is None:
            # If no points are provide, pad with an empty point (with label -1)
            input_points = torch.zeros(
                batch_size, 1, 1, 2, dtype=image_embeddings[-1].dtype, device=image_embeddings[-1].device
            )
            input_labels = -torch.ones(batch_size, 1, 1, dtype=torch.int32, device=image_embeddings[-1].device)

        if input_masks is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            if input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
                input_masks = F.interpolate(
                    input_masks.float(),
                    size=self.prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                ).to(input_masks.dtype)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        low_res_multimasks, iou_scores, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )

        is_obj_appearing = object_score_logits > 0
        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
        # consistent with the actual mask prediction
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        high_res_multimasks = (
            F.interpolate(
                low_res_multimasks.squeeze(1).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            .unsqueeze(1)
            .to(low_res_multimasks.dtype)
        )
        sam_output_token = sam_output_tokens[:, :, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(iou_scores, dim=-1)
            batch_inds = torch.arange(batch_size, device=high_res_multimasks.device)
            object_batch_inds = torch.arange(num_objects, device=high_res_multimasks.device)
            low_res_masks = low_res_multimasks[batch_inds, object_batch_inds, best_iou_inds]
            high_res_masks = high_res_multimasks[batch_inds, object_batch_inds, best_iou_inds]
            if sam_output_tokens.size(2) > 1:
                sam_output_token = sam_output_tokens[batch_inds, object_batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks[:, :, 0], high_res_multimasks[:, :, 0]

        # Extract object pointer from the SAM output token (with occlusion handling).
        # Meta SAM3.1 uses `interactive_obj_ptr_proj` (a separate 3-layer FF) on interactive
        # frames and `obj_ptr_proj` on the multiplex propagation path. `_forward_sam_heads`
        # is the interactive path (initial click / box / mask conditioning frames), so we
        # use the interactive projection here. The lost-object fallback is the shared
        # `no_obj_ptr_linear` blend (Meta `use_linear_no_obj_ptr=True`).
        object_pointer = self.interactive_object_pointer_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.to(object_pointer.dtype)
        object_pointer = self._blend_no_object_pointer(object_pointer, lambda_is_obj_appearing)

        return Sam31TrackerVideoImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_masks,
            high_res_masks=high_res_masks,
            object_pointer=object_pointer,
            object_score_logits=object_score_logits,
            image_embeddings=image_embeddings,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
        )

    def _run_single_frame_inference(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: dict | None,
        mask_inputs: torch.Tensor | None,
        reverse: bool,
        prev_sam_mask_logits: torch.Tensor | None = None,
        streaming: bool = False,
        add_to_existing_state: bool = False,
    ) -> dict[str, Any]:
        """SAM3.1-aware override.

        On top of the base behavior, this routes "new object on existing state" point clicks
        through Meta's `_use_mask_as_output` semantics (rescale via `20*x - 10`, saturate the
        object score, re-derive the object pointer from the rescaled mask). See the matching
        docstring on the public override in `modeling_sam3_1_tracker_video.py` for the full
        rationale; the flag is set by `forward` when a click is being added to a conditioning
        frame that already has other conditioned objects, mirroring Meta's
        `add_new_masks_to_existing_state` branch.
        """
        (
            current_vision_feats,
            current_vision_pos_embeds,
            interactive_vision_feats,
            _interactive_vision_pos_embeds,
        ) = self._prepare_vision_features(inference_session, frame_idx, batch_size)
        if point_inputs is not None and mask_inputs is not None:
            raise ValueError(
                "point_inputs and mask_inputs should not appear as input simultaneously on the same frame"
            )

        def _interactive_high_res_features():
            if len(interactive_vision_feats) <= 1:
                return None
            return [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(interactive_vision_feats[:-1], self.backbone_feature_sizes[:-1])
            ]

        def _interactive_pix_feat_bchw():
            pix = (
                interactive_vision_feats[-1]
                .permute(1, 2, 0)
                .view(-1, self.hidden_dim, *self.backbone_feature_sizes[-1])
            )
            return pix + self.no_memory_embedding.to(dtype=pix.dtype, device=pix.device).view(1, -1, 1, 1)

        interactive_high_res = _interactive_high_res_features()
        if len(current_vision_feats) > 1:
            propagation_high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], self.backbone_feature_sizes[:-1])
            ]
        else:
            propagation_high_res_features = None

        if mask_inputs is not None:
            pix_feat = _interactive_pix_feat_bchw()
            sam_outputs = self._use_mask_as_output(pix_feat, interactive_high_res, mask_inputs)
        elif point_inputs is not None:
            pix_feat = _interactive_pix_feat_bchw()
            interaction_mask_low_res = None
            if prev_sam_mask_logits is not None:
                interaction_mask_low_res = prev_sam_mask_logits
            elif not is_init_cond_frame:
                prop_outs = self._run_multiplex_propagation(
                    inference_session=inference_session,
                    frame_idx=frame_idx,
                    obj_idxs=[obj_idx],
                    reverse=reverse,
                    streaming=streaming,
                )
                if prop_outs:
                    interaction_mask_low_res = prop_outs[0]["pred_masks"].to(
                        device=pix_feat.device, dtype=pix_feat.dtype
                    )
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._single_frame_forward(
                pixel_values=None,
                input_points=point_inputs["point_coords"] if point_inputs is not None else None,
                input_labels=point_inputs["point_labels"] if point_inputs is not None else None,
                input_masks=interaction_mask_low_res,
                image_embeddings=(interactive_high_res or []) + [pix_feat],
                multimask_output=multimask_output,
            )
            # Meta `add_new_masks_to_existing_state` → `_use_mask_as_output(mask_inputs=sam_low_res)`:
            # rescale SAM masks via `20*x - 10`, saturate object score, re-derive object pointer
            # by running SAM heads with the rescaled mask as the dense prompt. See the matching
            # block in `modeling_sam3_1_tracker_video.py::_run_single_frame_inference` for the
            # full per-step explanation. Implemented inline (rather than via `_use_mask_as_output`)
            # to keep the stored mask resolution aligned with the regular SAM path.
            if add_to_existing_state and not self.training:
                out_scale, out_bias = 20.0, -10.0
                orig_pred_masks = sam_outputs.pred_masks
                orig_high_res_masks = sam_outputs.high_res_masks
                dtype = orig_pred_masks.dtype
                refined = self._single_frame_forward(
                    pixel_values=None,
                    input_points=None,
                    input_labels=None,
                    input_masks=orig_pred_masks.float(),
                    image_embeddings=(interactive_high_res or []) + [pix_feat],
                    multimask_output=True,
                )
                is_obj_appearing = (orig_pred_masks.float().flatten(2) > 0.0).any(dim=-1)
                lambda_is_obj_appearing = is_obj_appearing.to(dtype).unsqueeze(-1)
                new_object_score_logits = (out_scale * lambda_is_obj_appearing + out_bias).to(
                    sam_outputs.object_score_logits.dtype
                )
                lambda_ptr = lambda_is_obj_appearing.to(refined.object_pointer.dtype)
                new_object_pointer = self._blend_no_object_pointer(refined.object_pointer, lambda_ptr)
                sam_outputs = Sam31TrackerVideoImageSegmentationOutput(
                    iou_scores=sam_outputs.iou_scores,
                    pred_masks=(orig_pred_masks * out_scale + out_bias).to(dtype),
                    high_res_masks=(orig_high_res_masks * out_scale + out_bias).to(dtype),
                    object_pointer=new_object_pointer,
                    object_score_logits=new_object_score_logits,
                    image_embeddings=sam_outputs.image_embeddings,
                    vision_hidden_states=sam_outputs.vision_hidden_states,
                    vision_attentions=sam_outputs.vision_attentions,
                )
        else:
            pix_feat = self._prepare_memory_conditioned_features(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idx=obj_idx,
                is_initial_conditioning_frame=is_init_cond_frame,
                current_vision_features=current_vision_feats[-1],
                current_vision_positional_embeddings=current_vision_pos_embeds[-1],
                num_total_frames=inference_session.num_frames,
                track_in_reverse_time=reverse,
                streaming=streaming,
            )
            sam_mask_prompt = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._single_frame_forward(
                pixel_values=None,
                input_points=None,
                input_labels=None,
                input_masks=sam_mask_prompt,
                image_embeddings=(propagation_high_res_features or []) + [pix_feat],
                multimask_output=multimask_output,
                use_propagation_dense_pe=True,
            )

        current_out = {
            "pred_masks": sam_outputs.pred_masks,
            "object_pointer": sam_outputs.object_pointer,
            "high_res_masks": sam_outputs.high_res_masks,
        }
        if not self.training:
            current_out["object_score_logits"] = self._batch_object_score_logits(sam_outputs.object_score_logits)

        return current_out

    def _process_object_pointers(
        self,
        temporal_offsets: list[int],
        pointer_tokens: list[torch.Tensor],
        max_object_pointers_to_use: int,
        batch_size: int,
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SAM3.1-aware override that keeps object pointers 3-dim after stacking.

        `Sam3TrackerVideoModel._run_single_frame_inference` stores each `object_pointer` with shape
        `(batch_size, num_objects, hidden)` (3D) because the multimask-output index path broadcasts to
        a 2D index tensor. The SAM3 base implementation then relies on its
        `mem_dim < num_channels` reshape branch to flatten that extra axis. SAM3.1 uses
        `mem_dim == hidden_dim`, so that branch is skipped and the stacked tensor stays 4D, which then
        clashes with the 3D `maskmem_features` stream inside `_prepare_memory_conditioned_features`.

        We collapse the trailing object axis here so the rest of the SAM3 pipeline sees a clean
        `(seq_len, batch, hidden)` tensor regardless of SAM3.1's dimension setup.
        """
        if pointer_tokens:
            pointer_tokens = [t.reshape(batch_size, -1) for t in pointer_tokens]
        return super()._process_object_pointers(
            temporal_offsets=temporal_offsets,
            pointer_tokens=pointer_tokens,
            max_object_pointers_to_use=max_object_pointers_to_use,
            batch_size=batch_size,
            num_channels=num_channels,
            device=device,
        )

    def _maskmem_temporal_positional_encoding_for_t_pos(self, t_pos: int) -> torch.Tensor:
        """Index `memory_temporal_positional_encoding` like Meta `maskmem_tpos_enc` with `use_maskmem_tpos_v2=True`."""
        if t_pos <= 0 or t_pos >= self.num_maskmem:
            return self.memory_temporal_positional_encoding[self.num_maskmem - 1]
        return self.memory_temporal_positional_encoding[self.num_maskmem - t_pos - 1]

    def _gather_memory_frame_outputs(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        obj_idx: int,
        frame_idx: int,
        track_in_reverse_time: bool = False,
    ) -> list[tuple[int, int, dict]]:
        """
        Get memory frames from conditioning and non-conditioning outputs.

        Returns:
            List of `(t_pos, previous_frame_idx, output_data)` tuples. `t_pos` matches Meta multiplex
            `t_pos` used when indexing `maskmem_tpos_enc`.
        """
        temporal_positions_and_previous_outputs: list[tuple[int, int, dict]] = []

        conditioning_outputs = inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
        if not conditioning_outputs:
            raise ValueError(
                "maskmem_features in conditioning outputs cannot be empty when not is_initial_conditioning_frame"
            )
        conditioning_outputs, unselected_conditioning_outputs = self._select_closest_cond_frames(
            frame_idx, conditioning_outputs, max_cond_frame_num=self.config.max_cond_frame_num
        )

        for cond_abs_idx, out in conditioning_outputs.items():
            if not track_in_reverse_time:
                t_pos = frame_idx - cond_abs_idx
                prev_idx = cond_abs_idx
            else:
                t_pos = cond_abs_idx - frame_idx
                prev_idx = cond_abs_idx
            temporal_positions_and_previous_outputs.append((t_pos, prev_idx, out))

        for relative_frame_distance in range(self.num_maskmem - 1, 0, -1):
            if not track_in_reverse_time:
                previous_frame_idx = frame_idx - relative_frame_distance
            else:
                previous_frame_idx = frame_idx + relative_frame_distance

            output_data = inference_session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].get(
                previous_frame_idx, unselected_conditioning_outputs.get(previous_frame_idx, None)
            )

            t_pos = self.num_maskmem - relative_frame_distance
            temporal_positions_and_previous_outputs.append((t_pos, previous_frame_idx, output_data))

        return temporal_positions_and_previous_outputs

    def _build_memory_attention_inputs(
        self,
        temporal_positions_and_previous_outputs: list[tuple[int, int, dict]],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Concatenate memory features and positional embeddings from previous frames.

        Returns:
            Tuple of (memories_to_concatenate, memory_positional_embeddings_to_concatenate).
        """
        memories_to_concatenate = []
        memory_positional_embeddings_to_concatenate = []

        for relative_temporal_offset, prev_output_data in temporal_positions_and_previous_outputs:
            if prev_output_data is None:
                continue  # Skip if no output data for this temporal position (e.g., padding frames)

            # Load memory features (potentially from CPU to GPU)
            # Features are flattened: (Batch, Channels, H, W) -> (H*W, Batch, Channels)
            memory_features = prev_output_data["maskmem_features"].to(device, non_blocking=True)
            memories_to_concatenate.append(memory_features)

            # Spatial positional encoding (potentially from CPU to GPU)
            spatial_memory_pos_embed = prev_output_data["maskmem_pos_enc"].to(device, non_blocking=True)

            # Add temporal positional encoding (Meta `use_maskmem_tpos_v2` indexing)
            combined_memory_pos_embed = (
                spatial_memory_pos_embed
                + self._maskmem_temporal_positional_encoding_for_t_pos(relative_temporal_offset)
            )
            memory_positional_embeddings_to_concatenate.append(combined_memory_pos_embed)

        return memories_to_concatenate, memory_positional_embeddings_to_concatenate

    def _get_stored_or_cached_propagation_image_features(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        previous_frame_idx: int,
        prev_output_for_image: dict | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Top-level propagation FPN + PE for the memory-image stream (Meta `save_image_features`).

        Prefer tensors stored with mask memory at encode time so cross-attention matches Meta
        even when the per-frame vision cache was evicted or no longer matches encode-time feats.
        """
        if getattr(self.config, "save_propagation_image_features", True) and prev_output_for_image is not None:
            pi = prev_output_for_image.get("propagation_image_features")
            pp = prev_output_for_image.get("propagation_image_pos_enc")
            if isinstance(pi, torch.Tensor) and isinstance(pp, torch.Tensor):
                return (
                    pi.to(device=device, dtype=dtype, non_blocking=True),
                    pp.to(device=device, dtype=dtype, non_blocking=True),
                )
        cached = inference_session.cache.get_vision_features(previous_frame_idx)
        if cached is None:
            return None
        vfe = cached.get("vision_feats")
        vpe = cached.get("vision_pos_embeds")
        if not vfe or not vpe:
            return None
        return (
            vfe[-1].to(device=device, dtype=dtype, non_blocking=True),
            vpe[-1].to(device=device, dtype=dtype, non_blocking=True),
        )

    def _prepare_memory_conditioned_features(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_idx: int,
        is_initial_conditioning_frame: bool,
        current_vision_features: torch.Tensor,
        current_vision_positional_embeddings: torch.Tensor,
        num_total_frames: int,
        track_in_reverse_time: bool = False,
        streaming: bool = False,
    ) -> torch.Tensor:
        r"""
        Memory cross-attention using SAM3.1's decoupled (image + token) encoder.

        Mirrors `Sam3TrackerVideoModel._prepare_memory_conditioned_features` but threads
        the current image features and the per-frame past image features as a separate
        stream into `self.memory_attention`, in addition to the token-level memory. Past
        image features prefer snapshots stored with maskmem (Meta `save_image_features`);
        otherwise the vision cache is used.

        When neither snapshot nor cache is available for a frame that has maskmem, that
        memory row is skipped so token and image streams stay aligned.
        """
        batch_size = current_vision_features.size(1)
        num_channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]
        device = current_vision_features.device

        if self.num_maskmem == 0:
            return current_vision_features.permute(1, 2, 0).view(batch_size, num_channels, height, width)

        if is_initial_conditioning_frame:
            conditioned_feature_map_flat = current_vision_features + self.no_memory_embedding
            return conditioned_feature_map_flat.permute(1, 2, 0).view(batch_size, num_channels, height, width)

        temporal_positions_and_previous_outputs = self._gather_memory_frame_outputs(
            inference_session, obj_idx, frame_idx, track_in_reverse_time
        )

        # Gather token-stream memory and the matching image-stream memory in a single
        # pass so the two stay length-aligned. Image stream prefers snapshots stored with
        # maskmem (Meta `save_image_features`); otherwise falls back to the vision cache.
        memories_to_concatenate = []
        memory_pos_to_concatenate = []
        memory_image_to_concatenate = []
        memory_image_pos_to_concatenate = []

        for t_pos, previous_frame_idx, prev_output_data in temporal_positions_and_previous_outputs:
            if prev_output_data is None:
                continue
            if prev_output_data.get("maskmem_features") is None or prev_output_data.get("maskmem_pos_enc") is None:
                continue

            img_pair = self._get_stored_or_cached_propagation_image_features(
                inference_session, previous_frame_idx, prev_output_data, device, inference_session.dtype
            )
            if img_pair is None:
                continue
            past_image_feat, past_image_pos = img_pair

            temporal_pos_enc = self._maskmem_temporal_positional_encoding_for_t_pos(t_pos)
            memory_features = prev_output_data["maskmem_features"].to(device, non_blocking=True)
            spatial_memory_pos_embed = prev_output_data["maskmem_pos_enc"].to(device, non_blocking=True)
            memories_to_concatenate.append(memory_features)
            memory_pos_to_concatenate.append(spatial_memory_pos_embed + temporal_pos_enc)
            memory_image_to_concatenate.append(past_image_feat)
            memory_image_pos_to_concatenate.append(past_image_pos + temporal_pos_enc)

        if not memory_image_to_concatenate:
            # No usable memory frames — fall back to the current-frame features so
            # propagation can proceed without raising. Skipping object-pointer
            # gathering here keeps the streams length-consistent (no `memory_image`
            # slots to pad against).
            return current_vision_features.permute(1, 2, 0).view(batch_size, num_channels, height, width)

        # Object pointers (token-only; the decoupled wrapper zero-pads the matching
        # memory_image slots internally).
        temporal_offsets, pointer_tokens, max_object_pointers = self._get_object_pointers(
            inference_session, obj_idx, frame_idx, num_total_frames, device, track_in_reverse_time, streaming
        )
        num_object_pointer_tokens = 0
        if pointer_tokens:
            object_pointers, object_pointers_pos = self._process_object_pointers(
                temporal_offsets, pointer_tokens, max_object_pointers, batch_size, num_channels, device
            )
            if object_pointers is not None:
                memories_to_concatenate.append(object_pointers)
                memory_pos_to_concatenate.append(object_pointers_pos)
                num_object_pointer_tokens = object_pointers.shape[0]

        combined_memory = torch.cat(memories_to_concatenate, dim=0).to(dtype=inference_session.dtype)
        combined_memory_pos = torch.cat(memory_pos_to_concatenate, dim=0)
        combined_memory_image = torch.cat(memory_image_to_concatenate, dim=0).to(dtype=inference_session.dtype)
        combined_memory_image_pos = torch.cat(memory_image_pos_to_concatenate, dim=0)

        # In the PVS-only (no multiplex bucketing) path, `image` and `src` are the same
        # current-frame stream — the decoupled attention learns to fuse them through
        # separate projections (`image_cross_attn_q_proj` vs `cross_attn_q_proj`).
        conditioned_feature_map_flat = self.memory_attention(
            image=current_vision_features,
            src=current_vision_features,
            memory_image=combined_memory_image,
            memory=combined_memory,
            image_pos=current_vision_positional_embeddings,
            src_pos=current_vision_positional_embeddings,
            memory_image_pos=combined_memory_image_pos,
            memory_pos=combined_memory_pos,
            num_object_pointer_tokens=num_object_pointer_tokens,
        )

        return conditioned_feature_map_flat.permute(1, 2, 0).view(batch_size, num_channels, height, width)

    def _lookup_prev_memory_output_for_propagation(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        obj_idx: int,
        frame_idx: int,
        prev_frame_idx: int,
    ) -> dict | None:
        """Resolve stored outputs for `prev_frame_idx` using the same rules as `_gather_memory_frame_outputs`."""
        od = inference_session.output_dict_per_obj[obj_idx]
        selected_cond, unselected_cond = self._select_closest_cond_frames(
            frame_idx, od["cond_frame_outputs"], max_cond_frame_num=self.config.max_cond_frame_num
        )
        if prev_frame_idx in selected_cond:
            return selected_cond[prev_frame_idx]
        out = od["non_cond_frame_outputs"].get(prev_frame_idx)
        if out is not None:
            return out
        return unselected_cond.get(prev_frame_idx, None)

    def _prepare_memory_conditioned_features_batched_for_propagation(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_idxs: list[int],
        multiplex_state: Sam31MultiplexState,
        current_vision_features: torch.Tensor,
        current_vision_positional_embeddings: torch.Tensor,
        track_in_reverse_time: bool = False,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Bucket-batched memory fusion for propagation, mirroring Meta `Sam3VideoTrackingMultiplexModel`
        `_prepare_memory_conditioned_features` + `TransformerEncoderDecoupledCrossAttention`:
        one `memory_attention` call with `src` expanded to `num_buckets` and per-bucket memory
        columns (instead of per-slot forwards followed by mean-pooling over slots).
        """
        num_buckets = multiplex_state.num_buckets
        num_channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]
        device = current_vision_features.device
        work_dtype = inference_session.dtype

        if self.num_maskmem == 0:
            flat = current_vision_features.expand(-1, num_buckets, -1)
            return flat.permute(1, 2, 0).view(num_buckets, num_channels, height, width)

        ref_obj_idx = obj_idxs[0]
        schedule = self._gather_memory_frame_outputs(inference_session, ref_obj_idx, frame_idx, track_in_reverse_time)

        memories_to_concatenate: list[torch.Tensor] = []
        memory_pos_to_concatenate: list[torch.Tensor] = []
        memory_image_to_concatenate: list[torch.Tensor] = []
        memory_image_pos_to_concatenate: list[torch.Tensor] = []

        for t_pos, previous_frame_idx, ref_out in schedule:
            if ref_out is None:
                continue

            token_cols: list[torch.Tensor] = []
            pos_cols: list[torch.Tensor] = []
            row_ok = True
            for bucket_idx in range(num_buckets):
                internal_idx = None
                for slot_assignment in multiplex_state.assignments[bucket_idx]:
                    if slot_assignment >= 0:
                        internal_idx = slot_assignment
                        break
                if internal_idx is None:
                    row_ok = False
                    break
                obj_k = obj_idxs[internal_idx]
                prev_k = self._lookup_prev_memory_output_for_propagation(
                    inference_session, obj_k, frame_idx, previous_frame_idx
                )
                if prev_k is None or prev_k.get("maskmem_features") is None or prev_k.get("maskmem_pos_enc") is None:
                    row_ok = False
                    break
                memory_features = prev_k["maskmem_features"].to(device=device, dtype=work_dtype, non_blocking=True)
                spatial_memory_pos_embed = prev_k["maskmem_pos_enc"].to(
                    device=device, dtype=work_dtype, non_blocking=True
                )
                temporal_pos_enc = self._maskmem_temporal_positional_encoding_for_t_pos(t_pos)
                token_cols.append(memory_features)
                pos_cols.append(spatial_memory_pos_embed + temporal_pos_enc)

            if not row_ok or len(token_cols) != num_buckets:
                continue

            img_pair = self._get_stored_or_cached_propagation_image_features(
                inference_session, previous_frame_idx, ref_out, device, work_dtype
            )
            if img_pair is None:
                continue
            past_image_feat, past_image_pos = img_pair
            temporal_pos_enc = self._maskmem_temporal_positional_encoding_for_t_pos(t_pos)

            memories_to_concatenate.append(torch.cat(token_cols, dim=1))
            memory_pos_to_concatenate.append(torch.cat(pos_cols, dim=1))
            memory_image_to_concatenate.append(past_image_feat)
            memory_image_pos_to_concatenate.append(past_image_pos + temporal_pos_enc)

        if not memory_image_to_concatenate:
            # Meta `VideoTrackingMultiplexModel._prepare_memory_conditioned_features`: when there is
            # no memory to fuse (`len(to_cat_prompt)==0` / cleared image stream), return the current
            # propagation FPN features only — no `no_memory_embedding` on this path (see same file
            # around the early `return pix_feat` branches before `transformer.encoder`).
            flat = current_vision_features.expand(-1, num_buckets, -1)
            return flat.permute(1, 2, 0).view(num_buckets, num_channels, height, width)

        num_object_pointer_tokens = 0
        ptr_cols: list[torch.Tensor | None] = []
        ptr_pos_cols: list[torch.Tensor | None] = []
        max_ptr_len = 0
        for bucket_idx in range(num_buckets):
            internal_idx = None
            for slot_assignment in multiplex_state.assignments[bucket_idx]:
                if slot_assignment >= 0:
                    internal_idx = slot_assignment
                    break
            if internal_idx is None:
                ptr_cols.append(None)
                ptr_pos_cols.append(None)
                continue
            obj_k = obj_idxs[internal_idx]
            temporal_offsets, pointer_tokens, max_object_pointers = self._get_object_pointers(
                inference_session,
                obj_k,
                frame_idx,
                inference_session.num_frames,
                device,
                track_in_reverse_time,
                streaming,
            )
            object_pointers, object_pointers_pos = self._process_object_pointers(
                temporal_offsets,
                pointer_tokens,
                max_object_pointers,
                1,
                num_channels,
                device,
            )
            ptr_cols.append(object_pointers)
            ptr_pos_cols.append(object_pointers_pos)
            if object_pointers is not None:
                max_ptr_len = max(max_ptr_len, int(object_pointers.shape[0]))

        if max_ptr_len > 0:
            padded_ptrs: list[torch.Tensor] = []
            padded_ptr_pos: list[torch.Tensor] = []
            zptr = current_vision_features.new_zeros(max_ptr_len, 1, num_channels).to(dtype=work_dtype)
            for p, ppos in zip(ptr_cols, ptr_pos_cols):
                if p is None or ppos is None:
                    padded_ptrs.append(zptr.clone())
                    padded_ptr_pos.append(zptr.clone())
                else:
                    seq_len = int(p.shape[0])
                    if seq_len < max_ptr_len:
                        pad_rows = max_ptr_len - seq_len
                        pad_t = p.new_zeros(pad_rows, 1, num_channels)
                        pad_p = ppos.new_zeros(pad_rows, 1, num_channels)
                        p = torch.cat([p, pad_t], dim=0)
                        ppos = torch.cat([ppos, pad_p], dim=0)
                    padded_ptrs.append(p)
                    padded_ptr_pos.append(ppos)
            memories_to_concatenate.append(torch.cat(padded_ptrs, dim=1))
            memory_pos_to_concatenate.append(torch.cat(padded_ptr_pos, dim=1))
            num_object_pointer_tokens = max_ptr_len

        combined_memory = torch.cat(memories_to_concatenate, dim=0).to(dtype=work_dtype)
        combined_memory_pos = torch.cat(memory_pos_to_concatenate, dim=0)
        combined_memory_image = torch.cat(memory_image_to_concatenate, dim=0).to(dtype=work_dtype)
        combined_memory_image_pos = torch.cat(memory_image_pos_to_concatenate, dim=0)

        # Single batched memory_attention over all `num_buckets`, matching Meta's
        # `_prepare_memory_conditioned_features` exactly. The `image` / `memory_image`
        # streams carry batch size 1 (shared across buckets); `src` and `memory` are
        # expanded / shaped to `num_buckets`. PyTorch broadcasting propagates the shared
        # image keys into every bucket's cross-attention. With PyTorch's default SDPA
        # backend (FlashAttention / memory-efficient) this batched call never materialises
        # the full O(seq²) attention matrix, so peak activation memory is the same as
        # processing buckets one at a time — without the `num_buckets`× compute overhead
        # of sequential calls.
        src = current_vision_features.expand(-1, num_buckets, -1)
        conditioned_feature_map_flat = self.memory_attention(
            image=current_vision_features,
            src=src,
            memory_image=combined_memory_image,
            memory=combined_memory,
            image_pos=current_vision_positional_embeddings,
            src_pos=current_vision_positional_embeddings,
            memory_image_pos=combined_memory_image_pos,
            memory_pos=combined_memory_pos,
            num_object_pointer_tokens=num_object_pointer_tokens,
        )
        return conditioned_feature_map_flat.permute(1, 2, 0).view(num_buckets, num_channels, height, width)

    def _run_multiplex_propagation(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        obj_idxs: list[int],
        reverse: bool = False,
        streaming: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        r"""
        Run bucket-batched memory-conditioned propagation for a set of objects.

        Uses `_prepare_memory_conditioned_features_batched_for_propagation` (single
        `memory_attention` over buckets, Meta-aligned) then `propagation_mask_decoder`.

        Returns a list of per-object output dicts (one per entry in `obj_idxs`) with the
        same schema as `_run_single_frame_inference` so callers can plug them straight
        into `inference_session.store_output`.
        """
        if not obj_idxs:
            return []

        num_objs = len(obj_idxs)
        device = inference_session.device if hasattr(inference_session, "device") else next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype

        multiplex_state = self.multiplex_controller.get_state(
            num_valid_entries=num_objs,
            device=device,
            dtype=param_dtype,
            random=False,
        )
        num_buckets = multiplex_state.num_buckets
        multiplex_count = multiplex_state.multiplex_count

        current_vision_feats, current_vision_pos_embeds, _, _ = self._prepare_vision_features(
            inference_session, frame_idx, batch_size=1
        )

        bucket_pix_feat = self._prepare_memory_conditioned_features_batched_for_propagation(
            inference_session=inference_session,
            frame_idx=frame_idx,
            obj_idxs=obj_idxs,
            multiplex_state=multiplex_state,
            current_vision_features=current_vision_feats[-1],
            current_vision_positional_embeddings=current_vision_pos_embeds[-1],
            track_in_reverse_time=reverse,
            streaming=streaming,
        ).to(param_dtype)

        high_res_features = None
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0)
                .view(x.size(1), x.size(2), *s)
                .expand(num_buckets, -1, -1, -1)
                .contiguous()
                .to(dtype=param_dtype)
                for x, s in zip(current_vision_feats[:-1], self.backbone_feature_sizes[:-1])
            ]

        image_pos = self.get_image_wide_positional_embeddings().to(dtype=param_dtype)

        extra_per_object_embeddings = None
        if self.output_valid_embed is not None:
            valid_m = multiplex_state.get_valid_object_mask().to(device=device, dtype=param_dtype).unsqueeze(-1)
            ov = self.output_valid_embed.unsqueeze(0)
            oi = self.output_invalid_embed.unsqueeze(0)
            extra_per_object_embeddings = valid_m * ov + (1.0 - valid_m) * oi

        # SAM3.1's propagation mask decoder uses `multimask_outputs_only=True` by default —
        # it always emits `num_mask_output_per_object` candidate masks per slot and we pick
        # the best (highest predicted IoU) per slot, mirroring the original repo.
        low_res_multimasks, ious, sam_tokens_out, object_score_logits = self.propagation_mask_decoder(
            image_embeddings=bucket_pix_feat,
            image_positional_embeddings=image_pos,
            multimask_output=True,
            high_resolution_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(
            is_obj_appearing[..., None, None],
            low_res_multimasks,
            torch.full_like(low_res_multimasks, NO_OBJ_SCORE),
        )

        # Pick the per-slot best mask via IoU. Reshape to `(num_buckets * multiplex_count, …)`
        # so we can gather along the candidate dim, then reshape back.
        num_candidates = low_res_multimasks.shape[2]
        flat_masks = low_res_multimasks.flatten(0, 1)
        flat_ious = ious.flatten(0, 1)
        flat_sam_tokens = sam_tokens_out.flatten(0, 1)
        if num_candidates > 1:
            best_idx = torch.argmax(flat_ious, dim=-1)
            batch_inds = torch.arange(flat_masks.shape[0], device=flat_masks.device)
            best_masks = flat_masks[batch_inds, best_idx]
            best_sam_tokens = flat_sam_tokens[batch_inds, best_idx]
        else:
            best_masks = flat_masks[:, 0]
            best_sam_tokens = flat_sam_tokens[:, 0]
        low_res_masks = best_masks.view(num_buckets, multiplex_count, *best_masks.shape[-2:])
        sam_output_token = best_sam_tokens.view(num_buckets, multiplex_count, -1)

        high_res_masks_flat = F.interpolate(
            low_res_masks.float().flatten(0, 1).unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(low_res_masks.dtype)
        high_res_masks = high_res_masks_flat.view(num_buckets, multiplex_count, self.image_size, self.image_size)

        object_pointers = self.object_pointer_proj(sam_output_token)
        lambda_obj_appearing = is_obj_appearing.to(object_pointers.dtype).squeeze(-1).unsqueeze(-1)
        object_pointers = self._blend_no_object_pointer(object_pointers, lambda_obj_appearing)

        low_res_masks_per_obj = multiplex_state.demux(low_res_masks)
        high_res_masks_per_obj = multiplex_state.demux(high_res_masks)
        object_pointers_per_obj = multiplex_state.demux(object_pointers)
        object_score_logits_per_obj = multiplex_state.demux(object_score_logits)

        per_obj_outputs: list[dict[str, torch.Tensor]] = []
        for i in range(num_objs):
            per_obj_outputs.append(
                {
                    "pred_masks": low_res_masks_per_obj[i : i + 1].unsqueeze(1),
                    "object_pointer": object_pointers_per_obj[i : i + 1],
                    "high_res_masks": high_res_masks_per_obj[i : i + 1].unsqueeze(1),
                    "object_score_logits": self._batch_object_score_logits(object_score_logits_per_obj[i : i + 1]),
                }
            )
        return per_obj_outputs

    def _encode_new_memory(
        self,
        current_vision_feats: torch.Tensor,
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: bool,
        conditioning_slots_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode bucket-level memory for SAM3.1.

        Unlike SAM3 (one mask per object in a single 1-channel input), SAM3.1's
        `Sam31TrackerVideoMaskDownSampler` expects
        `multiplex_count * mask_downsampler_input_channel_multiplier` channels — the
        `multiplex_count` mask channels followed by the same number of "is this slot a
        conditioning object on this frame?" indicator channels. Meta builds these in
        `Sam3VideoTrackingMultiplexModel._encode_new_memory` under the
        `condition_as_mask_input=True` flag (see
        `facebook_sam3/sam3/model/video_tracking_multiplex.py::_encode_new_memory`): for
        each slot, the indicator is `condition_as_mask_input_fg=1.0` when that slot's
        object is a conditioning object on this frame (clicked / mask-prompted /
        consolidated cond) and `condition_as_mask_input_bg=0.0` otherwise (padded slots
        or pure propagation outputs).

        We accept the already-bucketed input shape `(num_buckets, multiplex_count, H, W)`
        and concatenate `conditioning_slots_mask` of the same shape as the second half of
        the channel dimension. When `conditioning_slots_mask` is `None` we fall back to
        zero indicator channels — the correct value for the non-conditioning propagation
        path; pass an explicit tensor for cond / click frames.

        Returns the bucket-level `(maskmem_features, maskmem_pos_enc)` ready to be
        replicated per-object by `_batch_encode_memories`.
        """
        batch_size = current_vision_feats.size(1)
        channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]

        mask_input_size_h, mask_input_size_w = self.prompt_encoder.mask_input_size
        mask_mem_size_h = mask_input_size_h * 4
        mask_mem_size_w = mask_input_size_w * 4

        if pred_masks_high_res.shape[-2:] != (mask_mem_size_h, mask_mem_size_w):
            target_lead = pred_masks_high_res.shape[:-2]
            flat = pred_masks_high_res.reshape(-1, 1, *pred_masks_high_res.shape[-2:]).float()
            flat = F.interpolate(
                flat,
                size=(mask_mem_size_h, mask_mem_size_w),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(pred_masks_high_res.dtype)
            pred_masks_high_res = flat.view(*target_lead, mask_mem_size_h, mask_mem_size_w)

        pix_feat = current_vision_feats.permute(1, 2, 0).view(batch_size, channels, height, width)

        # SAM3.1 always uses the continuous sigmoid path (Meta:
        # `apply_sigmoid_to_mask_logits_for_mem_enc=True` with
        # `binarize_mask_from_pts_for_mem_enc=False`, asserted in
        # `Sam3VideoTrackingMultiplexModel._encode_new_memory`). The base SAM3 class
        # binarizes mask-from-points inputs to {0, 1} before scaling, which together with
        # SAM3's scale=20 / bias=-10 produces hard {-10, +10} mask channels; SAM3.1 instead
        # uses scale=2 / bias=-1 to land in a continuous (-1, +1) range. The downstream
        # mask downsampler's first conv was trained against this continuous range, so we
        # never binarize here regardless of `is_mask_from_pts`.
        del is_mask_from_pts
        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * self.config.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.config.sigmoid_bias_for_mem_enc

        expected_channels = self.config.multiplex_count * self.config.mask_downsampler_input_channel_multiplier
        current_channels = mask_for_mem.shape[1]
        if current_channels < expected_channels:
            cond_channel_count = expected_channels - current_channels
            if conditioning_slots_mask is not None:
                if conditioning_slots_mask.shape[1] != cond_channel_count:
                    raise ValueError(
                        f"conditioning_slots_mask has {conditioning_slots_mask.shape[1]} channels but "
                        f"{cond_channel_count} are expected (mask channels={current_channels}, "
                        f"expected_total={expected_channels})."
                    )
                cond_channels = conditioning_slots_mask.to(device=mask_for_mem.device, dtype=mask_for_mem.dtype)
                if cond_channels.shape[-2:] != mask_for_mem.shape[-2:]:
                    cond_channels = cond_channels.expand(
                        cond_channels.shape[0], cond_channels.shape[1], *mask_for_mem.shape[-2:]
                    )
            else:
                cond_channels = mask_for_mem.new_zeros(
                    mask_for_mem.shape[0], cond_channel_count, *mask_for_mem.shape[2:]
                )
            mask_for_mem = torch.cat([mask_for_mem, cond_channels], dim=1)

        if pix_feat.shape[0] != mask_for_mem.shape[0]:
            pix_feat = pix_feat.expand(mask_for_mem.shape[0], -1, -1, -1).contiguous()

        mem_dtype = next(self.memory_encoder.parameters()).dtype
        pix_feat = pix_feat.to(dtype=mem_dtype)
        mask_for_mem = mask_for_mem.to(dtype=mem_dtype)

        maskmem_features, maskmem_pos_enc = self.memory_encoder(pix_feat, mask_for_mem)

        # Per-slot Meta `no_obj_embed_spatial` application. See the modeling file for the
        # full reasoning; in short, Meta sums the per-slot `(1 - is_obj_appearing_slot) *
        # no_obj_embed_spatial[slot]` across all `multiplex_count` slots in each bucket
        # (padding slots always contribute, valid slots contribute when their score <= 0).
        if self.occlusion_spatial_embedding_parameter is not None and object_score_logits is not None:
            emb = self.occlusion_spatial_embedding_parameter
            mp_count = emb.shape[0]
            mem_features_batch = maskmem_features.shape[0]
            score = object_score_logits.to(device=emb.device, dtype=emb.dtype)
            score_flat = score.reshape(-1)
            expected_numel = mem_features_batch * mp_count
            if score_flat.numel() != expected_numel:
                if score_flat.numel() < expected_numel:
                    pad = score_flat.new_zeros(expected_numel - score_flat.numel())
                    score_flat = torch.cat([score_flat, pad], dim=0)
                else:
                    score_flat = score_flat[:expected_numel]
            score_per_slot = score_flat.view(mem_features_batch, mp_count, 1)
            is_obj_appearing = (score_per_slot > 0).to(emb.dtype)
            no_obj_embed = ((1.0 - is_obj_appearing) * emb.unsqueeze(0)).sum(dim=1)
            maskmem_features = maskmem_features + no_obj_embed[..., None, None].to(maskmem_features.dtype)

        maskmem_features = maskmem_features.to(torch.bfloat16).flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc.to(pred_masks_high_res.dtype).flatten(2).permute(2, 0, 1)

        return maskmem_features, maskmem_pos_enc

    def _apply_non_overlapping_constraints(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """Mirror Meta `Sam3VideoTrackingMultiplexModel._apply_non_overlapping_constraints`.

        For each spatial location, only the object with the highest logit keeps its raw
        value; the other objects' logits are clamped to `max=-10` (so `sigmoid(-10)=4.5e-5`
        in the memory encoder). Used by `_consolidate_conditioning_frame_memories` before
        the memory encoder so multi-object cond-frame memories match Meta's preflight.
        """
        if pred_masks.size(0) <= 1:
            return pred_masks
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(pred_masks.size(0), device=pred_masks.device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))

    def _consolidate_conditioning_frame_memories(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
    ) -> None:
        """Re-encode `maskmem_features` on every multi-object conditioning frame using
        per-object high-res masks consolidated via `_apply_non_overlapping_constraints`.

        Mirrors Meta `Sam3VideoTrackingMultiplexDemo.propagate_in_video_preflight` →
        `_consolidate_temp_output_across_obj(run_mem_encoder=True)`. HF's per-click forward
        runs the memory encoder for one object at a time, so on multi-object conditioning
        frames each object's maskmem is encoded without seeing other objects' masks. Meta
        instead stacks all conditioning objects' masks, suppresses overlapping pixels with
        non-overlap, then runs the memory encoder once. Without this consolidation the
        cond-frame `maskmem_features` diverge between HF and Meta on multi-object scenes,
        which in turn shifts every propagation step. Called once at the start of
        `propagate_in_video_iterator`.

        Meta's path-dependent `20*x - 10` rescaling that
        `add_new_masks_to_existing_state` → `_use_mask_as_output` applies to later-added
        objects is already baked into the stored `high_res_masks` here (see the
        `add_to_existing_state` branch of `_run_single_frame_inference`). This
        consolidation only stacks the stored masks and applies non-overlap; the
        already-rescaled later-added masks naturally dominate the per-pixel argmax inside
        `_apply_non_overlapping_constraints` because their magnitudes are ~10x the raw
        SAM-decoded masks of the first-added object.
        """
        if self.num_maskmem == 0:
            return
        num_objs = inference_session.get_obj_num()
        if num_objs <= 1:
            return

        cond_frame_indices: set[int] = set()
        for obj_idx in range(num_objs):
            cond_frame_indices.update(inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"].keys())

        for frame_idx in sorted(cond_frame_indices):
            self._consolidate_conditioning_memories_at_frame(inference_session, frame_idx)

    def _consolidate_conditioning_memories_at_frame(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
    ) -> None:
        """Re-encode conditioning memories on one frame when multiple objects are conditioned."""
        if self.num_maskmem == 0:
            return
        num_objs = inference_session.get_obj_num()
        if num_objs <= 1:
            return

        obj_idxs_on_frame: list[int] = []
        high_res_masks: list[torch.Tensor] = []
        object_score_logits: list[torch.Tensor] = []
        for obj_idx in range(num_objs):
            stored = inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"].get(frame_idx)
            if stored is None:
                continue
            hr = stored.get("high_res_masks")
            osl = stored.get("object_score_logits")
            if hr is None or osl is None:
                continue
            obj_idxs_on_frame.append(obj_idx)
            high_res_masks.append(hr)
            object_score_logits.append(osl)

        if len(obj_idxs_on_frame) < 2:
            return

        stacked = torch.cat(high_res_masks, dim=0)
        stacked = self._apply_non_overlapping_constraints(stacked)
        masks_per_obj = [stacked[i : i + 1] for i in range(len(obj_idxs_on_frame))]

        self._batch_encode_memories(
            inference_session=inference_session,
            frame_idx=frame_idx,
            objects_needing_memory_encoding=obj_idxs_on_frame,
            high_res_masks_for_memory=masks_per_obj,
            object_score_logits_for_memory=object_score_logits,
            is_mask_from_pts_per_obj=[True] * len(obj_idxs_on_frame),
        )

    def propagate_in_video_iterator(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        show_progress_bar: bool = False,
    ) -> Iterator[Sam31TrackerVideoSegmentationOutput]:
        # Mirrors Meta `propagate_in_video_preflight(run_mem_encoder=True)`: re-encode
        # `maskmem_features` on every multi-object conditioning frame after applying
        # non-overlap suppression across objects. Without this, HF's per-click memory
        # encoder runs on each object in isolation while Meta runs on the consolidated
        # multi-object mask stack, which diverges the cond-frame maskmem (and therefore
        # every subsequent propagated frame).
        self._consolidate_conditioning_frame_memories(inference_session)
        return super().propagate_in_video_iterator(
            inference_session, start_frame_idx, max_frame_num_to_track, reverse, show_progress_bar
        )

    def _batch_encode_memories(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int,
        objects_needing_memory_encoding: list[int],
        high_res_masks_for_memory: list[torch.Tensor],
        object_score_logits_for_memory: list[torch.Tensor],
        is_mask_from_pts_per_obj: list[bool],
    ):
        r"""
        Bucketed memory encoding for SAM3.1.

        Groups `objects_needing_memory_encoding` into multiplex buckets, stacks each
        bucket's masks along the `multiplex_count` channel dim and runs the multiplex
        memory encoder once per bucket. Resulting bucket-level memory features are
        replicated across each object in that bucket so per-object
        `_prepare_memory_conditioned_features` lookups keep working unchanged.
        """
        if not objects_needing_memory_encoding:
            return

        encode_device = inference_session.inference_device
        param_dtype = next(self.parameters()).dtype

        current_vision_feats, current_vision_pos_embeds, _, _ = self._prepare_vision_features(
            inference_session, frame_idx, batch_size=1
        )
        pix_feat_stream = current_vision_feats[-1].to(device=encode_device, non_blocking=True)
        pix_pos_stream = current_vision_pos_embeds[-1].to(device=encode_device, non_blocking=True)
        save_pi = getattr(self.config, "save_propagation_image_features", True)

        # Masks/scores are often on `inference_state_device` (CPU) after `store_output`;
        # the memory encoder and vision features must run on `inference_device` (GPU).
        high_res_masks_batched = torch.cat(high_res_masks_for_memory, dim=0).float().to(
            device=encode_device, non_blocking=True
        )
        object_score_logits_batched = torch.cat(
            [self._batch_object_score_logits(s) for s in object_score_logits_for_memory], dim=0
        ).to(device=encode_device, non_blocking=True)

        num_objs = len(objects_needing_memory_encoding)
        device = encode_device

        multiplex_state = self.multiplex_controller.get_state(
            num_valid_entries=num_objs,
            device=device,
            dtype=param_dtype,
            random=False,
        )
        multiplex_count = multiplex_state.multiplex_count
        mask_h, mask_w = high_res_masks_batched.shape[-2:]
        mask_dtype = high_res_masks_batched.dtype

        for bucket_idx in range(multiplex_state.num_buckets):
            bucket_assignment = multiplex_state.assignments[bucket_idx]
            slot_masks: list[torch.Tensor] = []
            slot_scores: list[torch.Tensor] = []
            slot_cond_flags: list[float] = []
            score_template = object_score_logits_batched[:1]
            for internal_idx in bucket_assignment:
                if internal_idx < 0:
                    slot_masks.append(torch.full((1, mask_h, mask_w), NO_OBJ_SCORE, device=device, dtype=mask_dtype))
                    slot_scores.append(torch.zeros_like(score_template))
                    slot_cond_flags.append(0.0)
                else:
                    slot_masks.append(high_res_masks_batched[internal_idx].view(1, mask_h, mask_w))
                    slot_scores.append(object_score_logits_batched[internal_idx : internal_idx + 1])
                    # Meta's `condition_as_mask_input` semantics: a slot's conditioning channel
                    # is `condition_as_mask_input_fg=1.0` iff the slot's object is a
                    # conditioning object on this frame (clicked / mask-prompted, or being
                    # consolidated from a cond frame). `is_mask_from_pts_per_obj[i]` is True
                    # exactly when object `i`'s mask in `high_res_masks_for_memory` came from
                    # a click / mask prompt (HF `forward` sets this from `point_inputs is not
                    # None or mask_inputs is not None`; `_consolidate_conditioning_frame_memories`
                    # sets it to `True` for every object on the cond frame), so we can reuse
                    # it directly as the per-slot conditioning indicator.
                    slot_cond_flags.append(1.0 if is_mask_from_pts_per_obj[internal_idx] else 0.0)
            bucket_masks = torch.stack(slot_masks, dim=0).view(1, multiplex_count, mask_h, mask_w)
            bucket_scores = torch.cat(slot_scores, dim=0).reshape(1, -1)
            bucket_cond_mask = torch.tensor(slot_cond_flags, device=device, dtype=mask_dtype).view(
                1, multiplex_count, 1, 1
            )

            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=pix_feat_stream,
                pred_masks_high_res=bucket_masks,
                object_score_logits=bucket_scores,
                is_mask_from_pts=any(is_mask_from_pts_per_obj[idx] for idx in bucket_assignment if idx >= 0),
                conditioning_slots_mask=bucket_cond_mask,
            )

            for internal_idx in bucket_assignment:
                if internal_idx < 0:
                    continue
                obj_idx = objects_needing_memory_encoding[internal_idx]
                output_dict = inference_session.output_dict_per_obj[obj_idx]
                stored = None
                for store_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    if frame_idx in output_dict[store_key]:
                        stored = output_dict[store_key][frame_idx]
                        break
                if stored is None:
                    continue
                stored["maskmem_features"] = maskmem_features
                stored["maskmem_pos_enc"] = maskmem_pos_enc
                if save_pi:
                    snapshots = getattr(inference_session, "propagation_image_snapshots_per_frame", None)
                    if snapshots is None:
                        inference_session.propagation_image_snapshots_per_frame = {}
                        snapshots = inference_session.propagation_image_snapshots_per_frame
                    if frame_idx not in snapshots:
                        snapshots[frame_idx] = (
                            pix_feat_stream.detach().to(inference_session.inference_state_device),
                            pix_pos_stream.detach().to(inference_session.inference_state_device),
                        )
                    snap_pi, snap_pp = snapshots[frame_idx]
                    stored["propagation_image_features"] = snap_pi
                    stored["propagation_image_pos_enc"] = snap_pp

    def forward(
        self,
        inference_session: Sam31TrackerVideoInferenceSession,
        frame_idx: int | None = None,
        frame: torch.Tensor | None = None,
        reverse: bool = False,
        run_mem_encoder: bool = True,
        **kwargs,
    ) -> Sam31TrackerVideoSegmentationOutput:
        r"""
        SAM3.1 PVS forward pass with multiplex-batched propagation.

        Behaves like `Sam3TrackerVideoModel.forward` for objects that are either cached
        at `frame_idx` or have fresh interactive inputs — those still run through the
        per-object `_run_single_frame_inference` path so prompt encoding stays intact.
        Pure propagation objects (no new inputs, no cached output) are bucketed and
        forwarded through `_run_multiplex_propagation` in a single call to the multiplex
        mask decoder per bucket; memory encoding is likewise bucketed so the multiplex
        memory encoder receives `multiplex_count`-stacked masks per call.

        inference_session (`Sam31TrackerVideoInferenceSession`):
            The video inference session object.
        frame_idx (`int`, *optional*):
            The index of the frame on which to run inference. Not needed for streamed
            frames.
        frame (`torch.Tensor`, *optional*):
            The frame to process. Provide when streaming.
        reverse (`bool`, *optional*, defaults to `False`):
            Whether to propagate in reverse.
        run_mem_encoder (`bool`, *optional*, defaults to `True`):
            Whether to run the memory encoder on predicted masks.
        """
        if frame is not None:
            frame_idx = inference_session.add_new_frame(frame, frame_idx)

        if frame is not None and inference_session.get_obj_num() == 0:
            raise ValueError("No objects are provided for tracking; please add inputs first.")

        num_objects = inference_session.get_obj_num()
        pred_masks_per_obj: list[torch.Tensor | None] = [None] * num_objects
        object_score_logits_per_obj: list[torch.Tensor | None] = [None] * num_objects

        objects_needing_memory_encoding: list[int] = []
        high_res_masks_for_memory: list[torch.Tensor] = []
        object_score_logits_for_memory: list[torch.Tensor] = []
        is_mask_from_pts_per_obj: list[bool] = []

        cached_obj_idxs: list[int] = []
        interactive_obj_idxs: list[int] = []
        propagation_obj_idxs: list[int] = []
        # Map each object to the storage key that already holds a stored output for this
        # frame (cond_frame_outputs has priority). `cached_storage_key_per_obj` lets us
        # short-circuit the second tracker forward triggered by
        # `Sam3VideoModel._tracker_add_new_objects` (PCS new-object admission): when that
        # path re-enters this forward to encode the new detection masks, existing objects
        # already have pred_masks / maskmem_features stored at `frame_idx` (from the
        # planning-phase propagation + `_tracker_update_memories`). Re-propagating them
        # would overwrite those stored memories with a second forward that reads from a
        # different memory bank (the planning phase may have just promoted reconditioned
        # frames into `cond_frame_outputs` and encoded new mem features), drifting
        # existing masklets right when a new detection joins the session. Meta's
        # `add_new_masks_to_existing_state` only encodes the new object's mask in place
        # and never touches existing objects' memory; the per-frame `has_*_output` cached
        # branch reproduces that behaviour. On the first forward of a frame (called from
        # `run_tracker_propagation`) neither storage key holds `frame_idx` yet, so
        # existing objects still fall through to `propagation_obj_idxs` as before.
        cached_storage_key_per_obj: dict[int, str] = {}
        for obj_idx in range(num_objects):
            obj_id = inference_session.obj_idx_to_id(obj_idx)
            has_new_inputs = obj_id in inference_session.obj_with_new_inputs
            output_dict = inference_session.output_dict_per_obj[obj_idx]
            has_cond_output = frame_idx in output_dict["cond_frame_outputs"]
            has_non_cond_output = frame_idx in output_dict["non_cond_frame_outputs"]
            if (not has_new_inputs) and (has_cond_output or has_non_cond_output):
                cached_obj_idxs.append(obj_idx)
                cached_storage_key_per_obj[obj_idx] = (
                    "cond_frame_outputs" if has_cond_output else "non_cond_frame_outputs"
                )
            elif has_new_inputs:
                interactive_obj_idxs.append(obj_idx)
            else:
                propagation_obj_idxs.append(obj_idx)

        for obj_idx in cached_obj_idxs:
            is_cond = cached_storage_key_per_obj[obj_idx] == "cond_frame_outputs"
            pred_masks_per_obj[obj_idx] = inference_session.get_output(
                obj_idx, frame_idx, "pred_masks", is_conditioning_frame=is_cond
            )
            object_score_logits_per_obj[obj_idx] = inference_session.get_output(
                obj_idx, frame_idx, "object_score_logits", is_conditioning_frame=is_cond
            )

        for obj_idx in interactive_obj_idxs:
            obj_id = inference_session.obj_idx_to_id(obj_idx)
            is_init_cond_frame = frame_idx not in inference_session.frames_tracked_per_obj[obj_idx]
            if is_init_cond_frame:
                reverse = False
            point_inputs = inference_session.point_inputs_per_obj[obj_idx].get(frame_idx, None)
            mask_inputs = inference_session.mask_inputs_per_obj[obj_idx].get(frame_idx, None)
            if (
                point_inputs is not None or mask_inputs is not None
            ) and obj_id in inference_session.obj_with_new_inputs:
                inference_session.obj_with_new_inputs.remove(obj_id)

            # Detect Meta's `add_to_existing_state` case for point clicks: a brand-new object
            # being added on a conditioning frame that already holds other conditioned objects.
            # Refinement clicks (this `obj_idx` already has a cond output for `frame_idx`) and
            # first-on-frame additions (no other obj has cond output for `frame_idx`) keep the
            # plain SAM path. Mask-driven inputs already invoke `_use_mask_as_output` inside
            # `_run_single_frame_inference`, so we restrict the trigger to point clicks.
            already_my_cond = frame_idx in inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
            others_have_cond = any(
                other_idx != obj_idx
                and frame_idx in inference_session.output_dict_per_obj[other_idx]["cond_frame_outputs"]
                for other_idx in range(num_objects)
            )
            add_to_existing_state = (
                is_init_cond_frame
                and point_inputs is not None
                and mask_inputs is None
                and not already_my_cond
                and others_have_cond
            )

            current_out = self._run_single_frame_inference(
                inference_session=inference_session,
                obj_idx=obj_idx,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                reverse=reverse,
                streaming=frame is not None,
                add_to_existing_state=add_to_existing_state,
            )
            inference_session.store_output(
                obj_idx,
                frame_idx,
                output_value=current_out,
                is_conditioning_frame=is_init_cond_frame,
            )
            pred_masks_per_obj[obj_idx] = current_out["pred_masks"]
            object_score_logits_per_obj[obj_idx] = current_out["object_score_logits"]
            if run_mem_encoder and self.num_maskmem > 0:
                objects_needing_memory_encoding.append(obj_idx)
                high_res_masks_for_memory.append(current_out["high_res_masks"])
                object_score_logits_for_memory.append(
                    self._batch_object_score_logits(current_out["object_score_logits"])
                )
                is_mask_from_pts_per_obj.append(point_inputs is not None or mask_inputs is not None)
            if not is_init_cond_frame:
                inference_session.frames_tracked_per_obj[obj_idx][frame_idx] = {"reverse": reverse}

        if propagation_obj_idxs:
            prop_outputs = self._run_multiplex_propagation(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idxs=propagation_obj_idxs,
                reverse=reverse,
                streaming=frame is not None,
            )
            for obj_idx, current_out in zip(propagation_obj_idxs, prop_outputs):
                inference_session.store_output(
                    obj_idx,
                    frame_idx,
                    output_value=current_out,
                    is_conditioning_frame=False,
                )
                pred_masks_per_obj[obj_idx] = current_out["pred_masks"]
                object_score_logits_per_obj[obj_idx] = current_out["object_score_logits"]
                if run_mem_encoder and self.num_maskmem > 0:
                    objects_needing_memory_encoding.append(obj_idx)
                    high_res_masks_for_memory.append(current_out["high_res_masks"])
                    object_score_logits_for_memory.append(
                        self._batch_object_score_logits(current_out["object_score_logits"])
                    )
                    is_mask_from_pts_per_obj.append(False)
                inference_session.frames_tracked_per_obj[obj_idx][frame_idx] = {"reverse": reverse}

        self._batch_encode_memories(
            inference_session=inference_session,
            frame_idx=frame_idx,
            objects_needing_memory_encoding=objects_needing_memory_encoding,
            high_res_masks_for_memory=high_res_masks_for_memory,
            object_score_logits_for_memory=object_score_logits_for_memory,
            is_mask_from_pts_per_obj=is_mask_from_pts_per_obj,
        )

        if run_mem_encoder and frame_idx is not None:
            self._prune_stale_tracker_outputs(inference_session, frame_idx, reverse=reverse)

        squeezed_scores: list[torch.Tensor] = []
        for score in object_score_logits_per_obj:
            if score is not None:
                squeezed_scores.append(self._scalar_object_score_logits(score))
        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            all_object_score_logits = torch.cat(squeezed_scores, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
            all_object_score_logits = squeezed_scores[0]

        return Sam31TrackerVideoSegmentationOutput(
            object_ids=inference_session.obj_ids.copy(),
            pred_masks=all_pred_masks,
            object_score_logits=all_object_score_logits,
            frame_idx=frame_idx,
        )


__all__ = [
    "Sam31VisionConfig",
    "Sam31VisionModel",
    "Sam31VisionEncoderOutput",
    "Sam31TrackerVideoConfig",
    "Sam31TrackerVideoPromptEncoderConfig",
    "Sam31TrackerVideoMaskDecoderConfig",
    "Sam31TrackerVideoModel",
    "Sam31TrackerVideoMultiplexMaskDecoder",
    "Sam31TrackerVideoInferenceSession",
    "Sam31TrackerVideoPreTrainedModel",
    "Sam31TrackerVideoProcessor",
    "Sam31MultiplexController",
    "Sam31MultiplexState",
]
