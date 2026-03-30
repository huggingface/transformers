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
"""PyTorch SAM 3.1 video model."""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ..auto import CONFIG_MAPPING
from ..sam2.modeling_sam2 import Sam2LayerNorm
from ..sam2_video.modeling_sam2_video import Sam2VideoTwoWayTransformer
from ..sam3.configuration_sam3 import Sam3Config, Sam3MaskDecoderConfig, Sam3ViTConfig
from ..sam3.modeling_sam3 import Sam3FPNLayer, Sam3SinePositionEmbedding, Sam3ViTModel
from ..sam3_tracker_video.configuration_sam3_tracker_video import (
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoMaskDecoderConfig,
    Sam3TrackerVideoPromptEncoderConfig,
)
from ..sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoMaskDecoder, Sam3TrackerVideoPromptEncoder


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam3_1VideoPromptEncoderConfig(Sam3TrackerVideoPromptEncoderConfig):
    pass


@strict
class Sam3_1VideoMaskDecoderConfig(Sam3TrackerVideoMaskDecoderConfig):
    r"""
    iou_prediction_use_sigmoid (`bool`, *optional*, defaults to `False`):
        Whether the interactive mask decoder's IoU head applies a sigmoid to its outputs.
    """

    iou_prediction_use_sigmoid: bool = False


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31ViTConfig(Sam3ViTConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31MaskDecoderConfig(Sam3MaskDecoderConfig):
    pass


@strict
class Sam31TrackerPromptEncoderConfig(Sam3_1VideoPromptEncoderConfig):
    pass


@strict
class Sam31TrackerMaskDecoderConfig(Sam3_1VideoMaskDecoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam31Config(Sam3Config):
    pass


class Sam3_1LayerNorm(Sam2LayerNorm):
    pass


class Sam3_1TwoWayTransformer(Sam2VideoTwoWayTransformer):
    pass


class Sam3_1FPNLayer(Sam3FPNLayer):
    pass


class Sam3_1SinePositionEmbedding(Sam3SinePositionEmbedding):
    pass


class Sam3_1ViTModel(Sam3ViTModel):
    pass


class Sam3_1TrackerPromptEncoder(Sam3TrackerVideoPromptEncoder):
    pass


class Sam3_1TrackerMaskDecoder(Sam3TrackerVideoMaskDecoder):
    def __init__(self, config: Sam31TrackerMaskDecoderConfig):
        super().__init__(config)
        self.iou_prediction_head = self.output_hypernetworks_mlps[0].__class__(
            self.hidden_size,
            config.iou_head_hidden_dim,
            self.num_mask_tokens,
            config.iou_head_depth,
            sigmoid_output=config.iou_prediction_use_sigmoid,
        )


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam3_1VideoConfig(Sam3TrackerVideoConfig):
    r"""
    prompt_encoder_config (Union[`dict`, `Sam3_1VideoPromptEncoderConfig`], *optional*):
        Configuration of the interactive SAM prompt encoder.
    mask_decoder_config (Union[`dict`, `Sam3_1VideoMaskDecoderConfig`], *optional*):
        Configuration of the interactive SAM mask decoder.
    num_maskmem (`int`, *optional*, defaults to 7):
        Number of temporal memory slots tracked by the video model.
    sigmoid_scale_for_mem_enc (`float`, *optional*, defaults to 2.0):
        Scale applied to mask logits before they are encoded into memory features.
    sigmoid_bias_for_mem_enc (`float`, *optional*, defaults to -1.0):
        Bias applied to mask logits before they are encoded into memory features.
    enable_occlusion_spatial_embedding (`bool`, *optional*, defaults to `True`):
        Whether to add an occlusion-aware spatial embedding in the memory path.
    multimask_output_in_sam (`bool`, *optional*, defaults to `True`):
        Whether the interactive SAM decoder can emit multiple candidate masks.
    multimask_min_pt_num (`int`, *optional*, defaults to 0):
        Minimum number of prompt points that enables multimask decoding.
    multimask_max_pt_num (`int`, *optional*, defaults to 1):
        Maximum number of prompt points that keeps multimask decoding enabled.
    multimask_output_for_tracking (`bool`, *optional*, defaults to `True`):
        Whether multimask decoding is also enabled during tracking.
    max_object_pointers_in_encoder (`int`, *optional*, defaults to 16):
        Maximum number of object pointers passed to the memory encoder.
    max_cond_frame_num (`int`, *optional*, defaults to 4):
        Maximum number of conditioning frames used by memory attention.
    enable_temporal_pos_encoding_for_object_pointers (`bool`, *optional*, defaults to `True`):
        Whether to apply temporal positional encoding to object pointers.
    memory_attention_hidden_size (`int`, *optional*, defaults to 256):
        Hidden size of the temporal memory attention layers.
    memory_attention_num_layers (`int`, *optional*, defaults to 4):
        Number of layers in the temporal memory attention stack.
    memory_attention_num_attention_heads (`int`, *optional*, defaults to 1):
        Number of attention heads in each memory attention layer.
    memory_attention_downsample_rate (`int`, *optional*, defaults to 1):
        Spatial downsampling rate used inside the memory attention layers.
    memory_attention_feed_forward_hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size of the feed-forward block inside memory attention.
    memory_attention_feed_forward_hidden_act (`str`, *optional*, defaults to `"relu"`):
        Activation used in the memory attention feed-forward block.
    memory_attention_dropout (`float`, *optional*, defaults to 0.1):
        Dropout probability used in the memory attention stack.
    memory_attention_rope_theta (`float`, *optional*, defaults to 10000):
        Base theta value for rotary positional embeddings in memory attention.
    memory_attention_rope_feat_sizes (`list[int]`, *optional*, defaults to `[72, 72]`):
        Feature-map size assumed by the rotary positional embedding helper.
    memory_attention_rope_dropout (`float`, *optional*, defaults to 0.1):
        Dropout probability used around rotary positional embeddings in memory attention.
    memory_encoder_hidden_size (`int`, *optional*, defaults to 256):
        Hidden size of the memory encoder.
    memory_encoder_output_channels (`int`, *optional*, defaults to 64):
        Output channel count of the memory encoder.
    mask_downsampler_embed_dim (`int`, *optional*, defaults to 256):
        Embedding dimension used by the mask downsampler before memory encoding.
    mask_downsampler_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size used by the mask downsampler.
    mask_downsampler_stride (`int`, *optional*, defaults to 2):
        Stride used by the mask downsampler.
    mask_downsampler_padding (`int`, *optional*, defaults to 1):
        Padding used by the mask downsampler.
    mask_downsampler_total_stride (`int`, *optional*, defaults to 16):
        Effective total stride of the mask downsampler.
    mask_downsampler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation used in the mask downsampler.
    memory_fuser_num_layers (`int`, *optional*, defaults to 2):
        Number of layers in the memory fuser.
    memory_fuser_embed_dim (`int`, *optional*, defaults to 256):
        Embedding dimension used by the memory fuser.
    memory_fuser_intermediate_dim (`int`, *optional*, defaults to 1024):
        Intermediate hidden size used by the memory fuser.
    memory_fuser_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size used by the memory fuser.
    memory_fuser_padding (`int`, *optional*, defaults to 3):
        Padding used by the memory fuser.
    memory_fuser_layer_scale_init_value (`float`, *optional*, defaults to 1e-6):
        Initial layer-scale value used in the memory fuser blocks.
    memory_fuser_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation used in the memory fuser.
    multiplex_count (`int`, *optional*, defaults to 16):
        Number of object slots allocated in each multiplex bucket.
    eval_multiplex_count (`int`, *optional*, defaults to 16):
        Number of active object slots used per bucket at evaluation time.
    use_high_res_features_in_sam (`bool`, *optional*, defaults to `True`):
        Whether to use high-resolution FPN features in the interactive and propagation SAM decoders.
    use_obj_ptrs_in_encoder (`bool`, *optional*, defaults to `True`):
        Whether the memory transformer consumes object pointers.
    pred_obj_scores (`bool`, *optional*, defaults to `True`):
        Whether the mask decoders predict object existence logits.
    pred_obj_scores_mlp (`bool`, *optional*, defaults to `True`):
        Whether object score prediction uses an MLP head.
    fixed_no_obj_ptr (`bool`, *optional*, defaults to `True`):
        Whether the no-object pointer is kept fixed instead of being predicted dynamically.
    use_no_obj_ptr (`bool`, *optional*, defaults to `True`):
        Whether a dedicated no-object pointer embedding is used.
    use_linear_no_obj_ptr (`bool`, *optional*, defaults to `True`):
        Whether the no-object pointer uses a linear projection path.
    no_obj_embed_spatial (`bool`, *optional*, defaults to `True`):
        Whether to inject the no-object embedding into the spatial memory map.
    use_mlp_for_obj_ptr_proj (`bool`, *optional*, defaults to `True`):
        Whether object pointers are projected through an MLP.
    use_multimask_token_for_obj_ptr (`bool`, *optional*, defaults to `True`):
        Whether the multimask token also feeds the object-pointer projection.
    num_multimask_outputs (`int`, *optional*, defaults to 3):
        Number of interactive mask candidates predicted by the decoder.
    decode_mask_with_shared_tokens (`bool`, *optional*, defaults to `False`):
        Whether mask decoding reuses shared token embeddings.
    decode_mask_attribute_with_shared_tokens (`bool`, *optional*, defaults to `False`):
        Whether mask-attribute decoding reuses shared token embeddings.
    add_output_suppression_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to add embeddings that explicitly suppress inactive multiplex slots.
    add_object_conditional_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to add object-conditional embeddings to multiplex slots.
    add_object_unconditional_embeddings (`bool`, *optional*):
        Whether to add learned unconditional object embeddings. `None` keeps the upstream default behavior.
    condition_as_mask_input (`bool`, *optional*, defaults to `True`):
        Whether conditioning masks are fed back as mask prompts.
    condition_as_mask_input_fg (`float`, *optional*, defaults to 1.0):
        Foreground value used when turning conditioning masks into prompt inputs.
    condition_as_mask_input_bg (`float`, *optional*, defaults to 0.0):
        Background value used when turning conditioning masks into prompt inputs.
    use_maskmem_tpos_v2 (`bool`, *optional*, defaults to `True`):
        Whether to use the v2 temporal positional encoding for mask memory.
    directly_add_no_mem_embed (`bool`, *optional*, defaults to `True`):
        Whether the no-memory embedding is added directly to the backbone features.
    forward_backbone_per_frame_for_eval (`bool`, *optional*, defaults to `True`):
        Whether evaluation runs the backbone frame by frame, matching the upstream inference path.
    non_overlap_masks_for_mem_enc (`bool`, *optional*, defaults to `False`):
        Whether masks are forced not to overlap before memory encoding.
    apply_sigmoid_to_mask_logits_for_mem_enc (`bool`, *optional*, defaults to `True`):
        Whether mask logits are passed through sigmoid before memory encoding.
    iou_prediction_use_sigmoid (`bool`, *optional*, defaults to `False`):
        Whether the interactive mask decoder's IoU prediction head applies a sigmoid.
    use_memory_selection (`bool`, *optional*, defaults to `False`):
        Whether to enable memory-frame selection.
    memory_selection_threshold (`float`, *optional*, defaults to 0.01):
        Minimum score required for a frame to stay in the selected memory bank.
    compile_all_components (`bool`, *optional*, defaults to `False`):
        Whether to compile all major submodules for inference.
    is_dynamic_model (`bool`, *optional*, defaults to `True`):
        Whether the exported model should be treated as dynamically shaped.
    low_res_mask_size (`int`, *optional*, defaults to 288):
        Spatial size of low-resolution masks before they are upsampled to the input frame resolution.
    image_mean (`tuple[float, float, float]` or `list[float]`, *optional*, defaults to `(0.5, 0.5, 0.5)`):
        Mean used by the upstream SAM 3.1 video preprocessing pipeline.
    image_std (`tuple[float, float, float]` or `list[float]`, *optional*, defaults to `(0.5, 0.5, 0.5)`):
        Standard deviation used by the upstream SAM 3.1 video preprocessing pipeline.
    """

    model_type = "sam3_1_video"

    prompt_encoder_config: dict | Sam3_1VideoPromptEncoderConfig | None = None
    mask_decoder_config: dict | Sam3_1VideoMaskDecoderConfig | None = None

    multiplex_count: int = 16
    eval_multiplex_count: int = 16

    use_high_res_features_in_sam: bool = True
    use_obj_ptrs_in_encoder: bool = True
    pred_obj_scores: bool = True
    pred_obj_scores_mlp: bool = True
    fixed_no_obj_ptr: bool = True
    use_no_obj_ptr: bool = True
    use_linear_no_obj_ptr: bool = True
    no_obj_embed_spatial: bool = True
    use_mlp_for_obj_ptr_proj: bool = True
    use_multimask_token_for_obj_ptr: bool = True
    num_multimask_outputs: int = 3
    decode_mask_with_shared_tokens: bool = False
    decode_mask_attribute_with_shared_tokens: bool = False
    add_output_suppression_embeddings: bool = True
    add_object_conditional_embeddings: bool = False
    add_object_unconditional_embeddings: bool | None = None
    condition_as_mask_input: bool = True
    condition_as_mask_input_fg: float = 1.0
    condition_as_mask_input_bg: float = 0.0
    use_maskmem_tpos_v2: bool = True
    directly_add_no_mem_embed: bool = True
    forward_backbone_per_frame_for_eval: bool = True
    non_overlap_masks_for_mem_enc: bool = False
    apply_sigmoid_to_mask_logits_for_mem_enc: bool = True
    sigmoid_scale_for_mem_enc: float = 2.0
    sigmoid_bias_for_mem_enc: float = -1.0
    iou_prediction_use_sigmoid: bool = False
    use_memory_selection: bool = False
    memory_selection_threshold: float = 0.01
    compile_all_components: bool = False
    is_dynamic_model: bool = True

    # Runtime defaults used by the upstream 3.1 builder.
    low_res_mask_size: int = 288
    image_mean: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5)

    def __post_init__(self, **kwargs):
        if isinstance(self.image_mean, list):
            self.image_mean = tuple(self.image_mean)
        if isinstance(self.image_std, list):
            self.image_std = tuple(self.image_std)

        if self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"](
                backbone_feature_sizes=[[288, 288], [144, 144], [72, 72]],
                scale_factors=[4.0, 2.0, 1.0],
            )
        elif isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config.setdefault("backbone_feature_sizes", [[288, 288], [144, 144], [72, 72]])
            self.vision_config.setdefault("scale_factors", [4.0, 2.0, 1.0])
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        if self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam3_1VideoPromptEncoderConfig()
        elif isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam3_1VideoPromptEncoderConfig(**self.prompt_encoder_config)

        if self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3_1VideoMaskDecoderConfig()
        elif isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3_1VideoMaskDecoderConfig(**self.mask_decoder_config)

        super().__post_init__(**kwargs)


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(num_layers)])


@dataclass
class Sam3_1VideoBackboneOutput(ModelOutput):
    vision_features: torch.FloatTensor | None = None
    vision_pos_enc: tuple[torch.FloatTensor, ...] | None = None
    backbone_fpn: tuple[torch.FloatTensor, ...] | None = None
    interactive: dict[str, Any] | None = None
    sam2_backbone_out: dict[str, Any] | None = None


@dataclass
class Sam3_1VideoOutput(ModelOutput):
    interactive_iou_scores: torch.FloatTensor | None = None
    interactive_pred_masks: torch.FloatTensor | None = None
    interactive_high_res_masks: torch.FloatTensor | None = None
    interactive_object_score_logits: torch.FloatTensor | None = None
    interactive_object_pointer: torch.FloatTensor | None = None
    propagation_masks: torch.FloatTensor | None = None
    propagation_iou_scores: torch.FloatTensor | None = None
    propagation_object_score_logits: torch.FloatTensor | None = None
    propagation_sam_tokens: torch.FloatTensor | None = None
    image_embeddings: tuple[torch.FloatTensor, ...] | None = None
    backbone_outputs: Sam3_1VideoBackboneOutput | None = None


class Sam3_1RandomPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
            persistent=True,
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def get_dense_pe(
        self,
        image_embedding_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = image_embedding_size
        y_embed = torch.arange(height, device=device, dtype=dtype) + 0.5
        x_embed = torch.arange(width, device=device, dtype=dtype) + 0.5
        y_embed = y_embed / height
        x_embed = x_embed / width

        grid_y, grid_x = torch.meshgrid(y_embed, x_embed, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)
        pe = self._pe_encoding(coords)
        return pe.permute(2, 0, 1).unsqueeze(0)


class Sam3_1TriVisionNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trunk = Sam3_1ViTModel(config.backbone_config)
        self.position_encoding = Sam3_1SinePositionEmbedding(num_pos_feats=config.fpn_hidden_size // 2, normalize=True)

        self.convs = nn.ModuleList(
            [
                Sam3_1FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale_factor,
                )
                for scale_factor in config.scale_factors
            ]
        )
        self.interactive_convs = deepcopy(self.convs)
        self.propagation_convs = deepcopy(self.convs)
        self.patch_size = config.backbone_config.patch_size

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        trunk_outputs = self.trunk(pixel_values, return_dict=True)
        hidden_states = trunk_outputs.last_hidden_state
        batch_size, num_patches, hidden_size = hidden_states.shape
        spatial_size = int(math.sqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Expected square patch grid, got {num_patches} patches.")
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, hidden_size, spatial_size, spatial_size)

        sam3_hidden_states = ()
        interactive_hidden_states = ()
        propagation_hidden_states = ()
        sam3_pos = ()
        interactive_pos = ()
        propagation_pos = ()

        for conv, interactive_conv, propagation_conv in zip(
            self.convs, self.interactive_convs, self.propagation_convs, strict=True
        ):
            if need_sam3_out:
                current_hidden_state = conv(hidden_states)
                sam3_hidden_states += (current_hidden_state,)
                sam3_pos += (
                    self.position_encoding(
                        current_hidden_state.shape, current_hidden_state.device, current_hidden_state.dtype
                    ),
                )
            if need_interactive_out:
                current_hidden_state = interactive_conv(hidden_states)
                interactive_hidden_states += (current_hidden_state,)
                interactive_pos += (
                    self.position_encoding(
                        current_hidden_state.shape, current_hidden_state.device, current_hidden_state.dtype
                    ),
                )
            if need_propagation_out:
                current_hidden_state = propagation_conv(hidden_states)
                propagation_hidden_states += (current_hidden_state,)
                propagation_pos += (
                    self.position_encoding(
                        current_hidden_state.shape, current_hidden_state.device, current_hidden_state.dtype
                    ),
                )

        return (
            sam3_hidden_states,
            sam3_pos,
            interactive_hidden_states,
            interactive_pos,
            propagation_hidden_states,
            propagation_pos,
        )


class Sam3_1TriHeadVisionOnly(nn.Module):
    def __init__(self, config: Sam3_1VideoConfig):
        super().__init__()
        self.vision_backbone = Sam3_1TriVisionNeck(config.vision_config)

    def forward_image(
        self,
        pixel_values: torch.FloatTensor,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> Sam3_1VideoBackboneOutput:
        (
            sam3_hidden_states,
            sam3_pos,
            interactive_hidden_states,
            interactive_pos,
            propagation_hidden_states,
            propagation_pos,
        ) = self.vision_backbone(
            pixel_values,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

        outputs = {}
        if need_sam3_out:
            outputs.update(
                {
                    "vision_features": sam3_hidden_states[-1],
                    "vision_pos_enc": sam3_pos,
                    "backbone_fpn": sam3_hidden_states,
                }
            )
        if need_interactive_out:
            outputs["interactive"] = {
                "vision_features": interactive_hidden_states[-1],
                "vision_pos_enc": interactive_pos,
                "backbone_fpn": interactive_hidden_states,
            }
        if need_propagation_out:
            outputs["sam2_backbone_out"] = {
                "vision_features": propagation_hidden_states[-1],
                "vision_pos_enc": propagation_pos,
                "backbone_fpn": propagation_hidden_states,
            }

        return Sam3_1VideoBackboneOutput(**outputs)


class Sam3_1SimpleRoPEAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (72, 72),
        use_fa3: bool = False,
        use_rope_real: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat
        self.feat_sizes = feat_sizes
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = tensor.shape
        head_dim = hidden_size // self.num_heads
        tensor = tensor.view(batch_size, seq_len, self.num_heads, head_dim)
        return tensor.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        del num_k_exclude_rope
        q = self._shape(q)
        k = self._shape(k)
        v = self._shape(v)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, q.shape[1] * q.shape[-1])
        return output


class Sam3_1DecoupledTransformerDecoderLayerv2(nn.Module):
    def __init__(
        self,
        *,
        activation: str,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        cross_attention_first: bool = False,
        self_attention_rope: Sam3_1SimpleRoPEAttention,
        cross_attention_rope: Sam3_1SimpleRoPEAttention,
    ):
        super().__init__()
        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)

        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)

        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)

        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn and query_pos is not None else tgt2
        q = self.self_attn_q_proj(q)
        k = self.self_attn_k_proj(k)
        v = self.self_attn_v_proj(tgt2)
        tgt2 = self.self_attn_out_proj(self.self_attention_rope(q, k, v))
        return tgt + self.dropout1(tgt2)

    def _forward_ca(
        self,
        *,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        memory_image_pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        if self.pos_enc_at_cross_attn_queries and query_pos is not None:
            q = q + query_pos
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if self.pos_enc_at_cross_attn_keys and memory_image_pos is not None:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        tgt2 = self.cross_attn_out_proj(self.cross_attention_rope(q, k, v, num_k_exclude_rope=num_k_exclude_rope))
        return tgt + self.dropout2(tgt2)

    def forward(
        self,
        *,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        memory_image_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ):
        del image_pos, memory_pos
        if not self.pre_norm:
            raise NotImplementedError("Only pre-norm is supported in the current SAM 3.1 prototype.")

        if self.cross_attention_first:
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return image, tgt


class Sam3_1TransformerEncoderDecoupledCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer: Sam3_1DecoupledTransformerDecoderLayerv2,
        num_layers: int,
        batch_first: bool = True,
        use_image_in_output: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = _get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.use_image_in_output = use_image_in_output

        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        image: torch.Tensor,
        src: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor | None = None,
        src_pos: torch.Tensor | None = None,
        memory_image_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_obj_ptr_tokens: int = 0,
    ):
        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if not self.batch_first:
            output = output.transpose(0, 1)
            image = image.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_image = memory_image.transpose(0, 1)
            src_pos = None if src_pos is None else src_pos.transpose(0, 1)
            memory_image_pos = None if memory_image_pos is None else memory_image_pos.transpose(0, 1)

        if memory_image.shape[1] != memory.shape[1]:
            padding = memory.shape[1] - memory_image.shape[1]
            if padding != num_obj_ptr_tokens:
                raise ValueError("Unexpected object-pointer padding while running the decoupled transformer.")
            memory_image = torch.cat(
                [
                    memory_image,
                    torch.zeros(
                        memory_image.shape[0],
                        padding,
                        memory_image.shape[2],
                        device=memory_image.device,
                        dtype=memory_image.dtype,
                    ),
                ],
                dim=1,
            )

        for layer in self.layers:
            image, output = layer(
                image=image,
                tgt=output,
                memory_image=memory_image,
                memory=memory,
                image_pos=image_pos,
                query_pos=src_pos,
                memory_image_pos=memory_image_pos,
                memory_pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        output = self.norm(output + image) if self.use_image_in_output else self.norm(output)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return {"memory": output, "pos_embed": src_pos}


class Sam3_1TransformerWrapper(nn.Module):
    def __init__(self, config: Sam3_1VideoConfig):
        super().__init__()
        hidden_size = config.prompt_encoder_config.hidden_size
        self_attention_rope = Sam3_1SimpleRoPEAttention(
            d_model=hidden_size,
            num_heads=config.mask_decoder_config.num_attention_heads,
            dropout_p=config.memory_attention_dropout,
            rope_theta=config.memory_attention_rope_theta,
            feat_sizes=tuple(config.memory_attention_rope_feat_sizes),
        )
        cross_attention_rope = Sam3_1SimpleRoPEAttention(
            d_model=hidden_size,
            num_heads=config.mask_decoder_config.num_attention_heads,
            dropout_p=config.memory_attention_dropout,
            rope_theta=config.memory_attention_rope_theta,
            rope_k_repeat=True,
            feat_sizes=tuple(config.memory_attention_rope_feat_sizes),
        )
        encoder_layer = Sam3_1DecoupledTransformerDecoderLayerv2(
            activation="gelu",
            d_model=hidden_size,
            num_heads=config.mask_decoder_config.num_attention_heads,
            dim_feedforward=config.memory_attention_feed_forward_hidden_size,
            dropout=config.memory_attention_dropout,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention_rope=self_attention_rope,
            cross_attention_rope=cross_attention_rope,
        )
        self.encoder = Sam3_1TransformerEncoderDecoupledCrossAttention(
            d_model=hidden_size,
            frozen=False,
            pos_enc_at_input=True,
            layer=encoder_layer,
            num_layers=config.memory_attention_num_layers,
            batch_first=True,
            use_image_in_output=False,
        )
        self.decoder = None
        self.d_model = hidden_size


class Sam3_1SimpleMaskDownSampler(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        total_stride: int = 16,
        interpol_size: tuple[int, int] | list[int] | None = None,
        multiplex_count: int = 1,
        starting_out_chan: int = 4,
        input_channel_multiplier: int = 1,
    ):
        super().__init__()
        num_layers = int(math.log(total_stride, stride))
        if stride**num_layers != total_stride:
            raise ValueError("total_stride must be a power of stride.")
        self.encoder = nn.Sequential()
        mask_in_chans = multiplex_count * input_channel_multiplier
        mask_out_chans = starting_out_chan
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(mask_in_chans, mask_out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.encoder.append(Sam3_1LayerNorm(mask_out_chans, data_format="channels_first"))
            self.encoder.append(nn.GELU())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.interpol_size = None if interpol_size is None else list(interpol_size)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if self.interpol_size is not None and self.interpol_size != list(masks.shape[-2:]):
            masks = F.interpolate(
                masks.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(masks.dtype)
        return self.encoder(masks)


class Sam3_1CXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        super().__init__()
        groups = dim if use_dwconv else 1
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = Sam3_1LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        return residual + self.drop_path(hidden_states)


class Sam3_1SimpleFuser(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Sam3_1SimpleMaskEncoder(nn.Module):
    def __init__(self, out_dim: int, mask_downsampler: nn.Module, fuser: nn.Module, position_encoding: nn.Module):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()

    def forward(self, pix_feat: torch.Tensor, masks: torch.Tensor, skip_mask_sigmoid: bool = False):
        if not skip_mask_sigmoid:
            masks = masks.sigmoid()
        masks = self.mask_downsampler(masks)
        hidden_states = self.pix_feat_proj(pix_feat.to(masks.device)) + masks
        hidden_states = self.fuser(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        pos = self.position_encoding(hidden_states.shape, hidden_states.device, hidden_states.dtype)
        return {"vision_features": hidden_states, "vision_pos_enc": [pos]}


class Sam3_1MultiplexState:
    def __init__(
        self, assignments: list[list[int]], device: torch.device, dtype: torch.dtype, allowed_bucket_capacity: int
    ):
        self.assignments = assignments
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self.num_buckets = len(assignments)
        self.multiplex_count = len(assignments[0])
        self.total_valid_entries = sum(sum(1 for idx in bucket if idx >= 0) for bucket in assignments)
        self._precompute_transition_matrices()

    def _precompute_transition_matrices(self):
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count,
            self.total_valid_entries,
            device=self.device,
            dtype=self.dtype,
        )
        self.demux_matrix = torch.zeros(
            self.total_valid_entries,
            self.num_buckets * self.multiplex_count,
            device=self.device,
            dtype=self.dtype,
        )
        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, object_idx in enumerate(bucket):
                flat_idx = bucket_idx * self.multiplex_count + slot_idx
                if object_idx >= 0:
                    self.mux_matrix[flat_idx, object_idx] = 1.0
                    self.demux_matrix[object_idx, flat_idx] = 1.0

    def mux(self, tensor: torch.Tensor) -> torch.Tensor:
        output_shape = (self.num_buckets, self.multiplex_count) + tensor.shape[1:]
        return (self.mux_matrix @ tensor.reshape(tensor.shape[0], -1)).view(output_shape)

    def demux(self, tensor: torch.Tensor) -> torch.Tensor:
        output_shape = (self.total_valid_entries,) + tensor.shape[2:]
        return (self.demux_matrix @ tensor.reshape(self.num_buckets * self.multiplex_count, -1)).view(output_shape)

    def get_valid_object_mask(self) -> torch.Tensor:
        return (self.mux_matrix.sum(dim=1) > 0).reshape(self.num_buckets, self.multiplex_count)


class Sam3_1MultiplexController(nn.Module):
    def __init__(self, multiplex_count: int, eval_multiplex_count: int = -1):
        super().__init__()
        self.multiplex_count = multiplex_count
        self.eval_multiplex_count = multiplex_count if eval_multiplex_count < 0 else eval_multiplex_count

    @property
    def allowed_bucket_capacity(self) -> int:
        return self.multiplex_count if self.training else self.eval_multiplex_count

    def get_state(self, num_valid_entries: int, device: torch.device, dtype: torch.dtype) -> Sam3_1MultiplexState:
        allowed_bucket_capacity = self.allowed_bucket_capacity
        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)
        ids = torch.arange(num_valid_entries, dtype=torch.int64)
        total_elements = num_buckets * allowed_bucket_capacity
        if ids.shape[0] < total_elements:
            ids = torch.cat([ids, torch.full((total_elements - ids.shape[0],), -1, dtype=torch.int64)])
        assignments = []
        for bucket_idx in range(num_buckets):
            bucket_ids = ids[
                bucket_idx * allowed_bucket_capacity : (bucket_idx + 1) * allowed_bucket_capacity
            ].tolist()
            bucket_ids = bucket_ids + [-1] * (self.multiplex_count - allowed_bucket_capacity)
            assignments.append(bucket_ids)
        return Sam3_1MultiplexState(
            assignments, device=device, dtype=dtype, allowed_bucket_capacity=allowed_bucket_capacity
        )


class Sam3_1MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        super().__init__()
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            hidden_states = F.relu(layer(hidden_states)) if idx < len(self.layers) - 1 else layer(hidden_states)
        if self.sigmoid_output:
            hidden_states = hidden_states.sigmoid()
        return hidden_states


class Sam3_1MaskDecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = Sam3_1TwoWayTransformer(config)

    def forward(self, src: torch.Tensor, pos_src: torch.Tensor, tokens: torch.Tensor):
        queries, image_tokens = self.inner(
            point_embeddings=tokens.unsqueeze(1),
            image_embeddings=src,
            image_positional_embeddings=pos_src,
            attention_similarity=None,
            target_embedding=None,
        )
        return queries.squeeze(1), image_tokens.squeeze(1)


class Sam3_1MultiplexMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        multiplex_count: int,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        decode_mask_with_shared_tokens: bool = False,
        decode_mask_attribute_with_shared_tokens: bool = False,
        multimask_outputs_only: bool = False,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_outputs_only = multimask_outputs_only
        self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = decode_mask_attribute_with_shared_tokens
        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.num_mask_output_per_object = (
            num_multimask_outputs if multimask_outputs_only else num_multimask_outputs + 1
        )
        self.num_mask_tokens = (
            multiplex_count if decode_mask_with_shared_tokens else multiplex_count * self.num_mask_output_per_object
        )

        if not self.decode_mask_attribute_with_shared_tokens:
            self.iou_token = nn.Embedding(multiplex_count, transformer_dim)
            if self.pred_obj_scores:
                self.obj_score_token = nn.Embedding(multiplex_count, transformer_dim)

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            Sam3_1LayerNorm(transformer_dim // 4, data_format="channels_first"),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        if self.num_multimask_outputs == 0:
            self.output_hypernetworks_mlp = Sam3_1MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        else:
            self.output_hypernetworks_mlps = nn.ModuleList(
                [
                    Sam3_1MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for _ in range(self.num_mask_output_per_object)
                ]
            )

        iou_output_dim = (
            1
            if (decode_mask_attribute_with_shared_tokens and not decode_mask_with_shared_tokens)
            else self.num_mask_output_per_object
        )
        self.iou_prediction_head = Sam3_1MLP(
            transformer_dim,
            iou_head_hidden_dim,
            iou_output_dim,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = Sam3_1MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = image_embeddings.shape[0]
        token_list = []
        if self.pred_obj_scores and not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.obj_score_token.weight)
        if not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.iou_token.weight)
        tokens = torch.cat(token_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1)

        if extra_per_object_embeddings is not None:
            mask_tokens = self.mask_tokens.weight.view(
                1, self.multiplex_count, self.num_mask_output_per_object, -1
            ).expand(batch_size, -1, -1, -1)
            mask_tokens = (mask_tokens + extra_per_object_embeddings.unsqueeze(2)).flatten(1, 2)
        else:
            mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([tokens, mask_tokens], dim=1)

        pos_src = image_pe.repeat(batch_size, 1, 1, 1)
        hidden_states, src = self.transformer(image_embeddings, pos_src, tokens)
        if self.decode_mask_attribute_with_shared_tokens:
            iou_token_out = hidden_states[:, 0 : self.num_mask_tokens]
            mask_tokens_out = hidden_states[:, 0 : self.num_mask_tokens]
            if self.pred_obj_scores:
                obj_score_token_out = mask_tokens_out
        else:
            start = 0
            if self.pred_obj_scores:
                obj_score_token_out = hidden_states[:, start : start + self.multiplex_count]
                start += self.multiplex_count
            iou_token_out = hidden_states[:, start : start + self.multiplex_count]
            start += self.multiplex_count
            mask_tokens_out = hidden_states[:, start : start + self.num_mask_tokens]

        channels = image_embeddings.shape[1]
        height, width = image_embeddings.shape[-2:]
        src = src.transpose(1, 2).view(batch_size, channels, height, width)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        if self.decode_mask_with_shared_tokens:
            mask_tokens_out = mask_tokens_out.view(batch_size, self.multiplex_count, 1, -1)
        else:
            mask_tokens_out = mask_tokens_out.view(
                batch_size, self.multiplex_count, self.num_mask_output_per_object, -1
            )

        if self.num_multimask_outputs == 0:
            hyper_in = self.output_hypernetworks_mlp(mask_tokens_out[:, :, 0, :]).unsqueeze(2)
        else:
            hyper_in = []
            for idx in range(self.num_mask_output_per_object):
                current_tokens = mask_tokens_out[:, :, 0 if self.decode_mask_with_shared_tokens else idx, :]
                hyper_in.append(self.output_hypernetworks_mlps[idx](current_tokens))
            hyper_in = torch.stack(hyper_in, dim=2)

        _, output_channels, output_height, output_width = upscaled_embedding.shape
        masks = torch.bmm(
            hyper_in.flatten(1, 2),
            upscaled_embedding.view(batch_size, output_channels, output_height * output_width),
        ).view(batch_size, self.multiplex_count, self.num_mask_output_per_object, output_height, output_width)
        iou_pred = self.iou_prediction_head(iou_token_out).view(
            batch_size, self.multiplex_count, self.num_mask_output_per_object
        )

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
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], iou_pred.shape[1], 1)

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def _get_stability_scores(self, mask_logits: torch.Tensor) -> torch.Tensor:
        mask_logits = mask_logits.flatten(-2)
        area_i = (mask_logits > self.dynamic_multimask_stability_delta).sum(dim=-1).float()
        area_u = (mask_logits > -self.dynamic_multimask_stability_delta).sum(dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits: torch.Tensor, all_iou_scores: torch.Tensor):
        batch_size, multiplex_count = all_mask_logits.shape[:2]
        all_mask_logits = all_mask_logits.flatten(0, 1)
        all_iou_scores = all_iou_scores.flatten(0, 1)

        multimask_logits = all_mask_logits[:, 1:]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_indices = torch.argmax(multimask_iou_scores, dim=-1)
        batch_indices = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_indices, best_indices].unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_indices, best_indices].unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        is_stable = self._get_stability_scores(singlemask_logits) >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits), singlemask_logits, best_multimask_logits
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores), singlemask_iou_scores, best_multimask_iou_scores
        )
        return mask_logits_out.unflatten(0, (batch_size, multiplex_count)), iou_scores_out.unflatten(
            0, (batch_size, multiplex_count)
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        high_res_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )
        masks = outputs["masks"]
        iou_pred = outputs["iou_pred"]
        mask_tokens_out = outputs["mask_tokens_out"]

        if multimask_output:
            if not self.multimask_outputs_only:
                masks = masks[:, :, 1:]
                iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1]
            iou_pred = iou_pred[:, :, 0:1]

        sam_tokens_out = mask_tokens_out[:, :, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out if self.multimask_outputs_only else mask_tokens_out[:, :, 1:]

        outputs["masks"] = masks
        outputs["iou_pred"] = iou_pred
        outputs["sam_tokens_out"] = sam_tokens_out
        return outputs


class Sam3_1VideoPreTrainedModel(PreTrainedModel):
    config_class = Sam3_1VideoConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"


@auto_docstring(custom_intro="SAM 3.1 multiplex video model with a tri-head video backbone.")
class Sam3_1VideoModel(Sam3_1VideoPreTrainedModel):
    def __init__(self, config: Sam3_1VideoConfig):
        super().__init__(config)
        self.config = config
        self.hidden_dim = config.prompt_encoder_config.hidden_size
        self.image_size = config.image_size
        self.backbone_stride = config.vision_config.backbone_config.patch_size
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.low_res_mask_size = config.low_res_mask_size

        self.backbone = Sam3_1TriHeadVisionOnly(config)
        self.transformer = Sam3_1TransformerWrapper(config)

        self.maskmem_backbone = Sam3_1SimpleMaskEncoder(
            out_dim=self.hidden_dim,
            mask_downsampler=Sam3_1SimpleMaskDownSampler(
                embed_dim=self.hidden_dim,
                kernel_size=config.mask_downsampler_kernel_size,
                stride=config.mask_downsampler_stride,
                padding=config.mask_downsampler_padding,
                total_stride=config.mask_downsampler_total_stride,
                interpol_size=(1152, 1152),
                multiplex_count=config.multiplex_count,
                starting_out_chan=4,
                input_channel_multiplier=2,
            ),
            fuser=Sam3_1SimpleFuser(
                layer=Sam3_1CXBlock(
                    dim=self.hidden_dim,
                    kernel_size=config.memory_fuser_kernel_size,
                    padding=config.memory_fuser_padding,
                    layer_scale_init_value=config.memory_fuser_layer_scale_init_value,
                    use_dwconv=True,
                ),
                num_layers=config.memory_fuser_num_layers,
            ),
            position_encoding=Sam3_1SinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True),
        )
        self.multiplex_controller = Sam3_1MultiplexController(
            multiplex_count=config.multiplex_count, eval_multiplex_count=config.eval_multiplex_count
        )

        self.image_pe_layer = Sam3_1RandomPositionEmbedding(self.hidden_dim // 2)
        config.mask_decoder_config.iou_prediction_use_sigmoid = config.iou_prediction_use_sigmoid
        self.interactive_sam_prompt_encoder = Sam3_1TrackerPromptEncoder(config.prompt_encoder_config)
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.interactive_sam_mask_decoder = Sam3_1TrackerMaskDecoder(config.mask_decoder_config)
        self.sam_mask_decoder = Sam3_1MultiplexMaskDecoder(
            transformer_dim=self.hidden_dim,
            transformer=Sam3_1MaskDecoderTransformer(config.mask_decoder_config),
            multiplex_count=config.multiplex_count,
            num_multimask_outputs=config.num_multimask_outputs,
            iou_head_depth=config.mask_decoder_config.iou_head_depth,
            iou_head_hidden_dim=config.mask_decoder_config.iou_head_hidden_dim,
            use_high_res_features=config.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=config.iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=False,
            dynamic_multimask_stability_delta=config.mask_decoder_config.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=config.mask_decoder_config.dynamic_multimask_stability_thresh,
            pred_obj_scores=config.pred_obj_scores,
            pred_obj_scores_mlp=config.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=config.use_multimask_token_for_obj_ptr,
            decode_mask_with_shared_tokens=config.decode_mask_with_shared_tokens,
            decode_mask_attribute_with_shared_tokens=config.decode_mask_attribute_with_shared_tokens,
            multimask_outputs_only=config.num_multimask_outputs > 0 and config.multimask_output_in_sam,
        )

        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(config.num_maskmem, 1, 1, self.hidden_dim))
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        self.no_obj_ptr = None
        self.no_obj_ptr_linear = None
        if config.use_no_obj_ptr:
            if config.use_linear_no_obj_ptr:
                self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                self.no_obj_ptr = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
        self.output_valid_embed = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
        self.output_invalid_embed = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))

        self.obj_cond_embed = None
        self.obj_non_cond_embed = None
        if config.add_object_conditional_embeddings:
            self.obj_cond_embed = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))
            if config.add_object_unconditional_embeddings or config.add_object_unconditional_embeddings is None:
                self.obj_non_cond_embed = nn.Parameter(torch.zeros(config.multiplex_count, self.hidden_dim))

        if config.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.interactive_obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            if config.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = Sam3_1MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
                self.interactive_obj_ptr_proj = Sam3_1MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = nn.Identity()
            self.interactive_obj_ptr_proj = nn.Identity()

        if config.enable_temporal_pos_encoding_for_object_pointers:
            self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.obj_ptr_tpos_proj = nn.Identity()

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.vision_backbone.trunk.get_input_embeddings()

    def get_image_wide_positional_embeddings(self, batch_size: int = 1, device=None, dtype=None) -> torch.Tensor:
        size = (self.sam_image_embedding_size, self.sam_image_embedding_size)
        prompt_pe_layer = self.interactive_sam_prompt_encoder.shared_embedding
        device = prompt_pe_layer.positional_embedding.device if device is None else device
        dtype = prompt_pe_layer.positional_embedding.dtype if dtype is None else dtype
        grid = torch.ones(size, device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]
        dense_pe = prompt_pe_layer(torch.stack([x_embed, y_embed], dim=-1)).permute(2, 0, 1).unsqueeze(0)
        return dense_pe.repeat(batch_size, 1, 1, 1)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> Sam3_1VideoBackboneOutput:
        outputs = self.backbone.forward_image(
            pixel_values,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

        if need_interactive_out and outputs.interactive is not None:
            interactive_fpn = list(outputs.interactive["backbone_fpn"])
            interactive_fpn[0] = self.interactive_sam_mask_decoder.conv_s0(interactive_fpn[0])
            interactive_fpn[1] = self.interactive_sam_mask_decoder.conv_s1(interactive_fpn[1])
            outputs.interactive["backbone_fpn"] = tuple(interactive_fpn)

        if need_propagation_out and outputs.sam2_backbone_out is not None:
            propagation_fpn = list(outputs.sam2_backbone_out["backbone_fpn"])
            propagation_fpn[0] = self.sam_mask_decoder.conv_s0(propagation_fpn[0])
            propagation_fpn[1] = self.sam_mask_decoder.conv_s1(propagation_fpn[1])
            outputs.sam2_backbone_out["backbone_fpn"] = tuple(propagation_fpn)

        return outputs

    def _select_best_interactive_masks(
        self,
        low_res_multimasks: torch.Tensor,
        iou_scores: torch.Tensor,
        sam_output_tokens: torch.Tensor,
        high_res_multimasks: torch.Tensor,
        multimask_output: bool,
    ):
        if multimask_output:
            best_iou_indices = torch.argmax(iou_scores, dim=-1)
            batch_indices = torch.arange(low_res_multimasks.shape[0], device=low_res_multimasks.device)[:, None]
            object_indices = torch.arange(low_res_multimasks.shape[1], device=low_res_multimasks.device)[None, :]
            low_res_masks = low_res_multimasks[batch_indices, object_indices, best_iou_indices]
            high_res_masks = high_res_multimasks[batch_indices, object_indices, best_iou_indices]
            sam_output_token = sam_output_tokens[batch_indices, object_indices, best_iou_indices]
        else:
            low_res_masks = low_res_multimasks[:, :, 0]
            high_res_masks = high_res_multimasks[:, :, 0]
            sam_output_token = sam_output_tokens[:, :, 0]
        return low_res_masks, high_res_masks, sam_output_token

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_points: torch.FloatTensor | None = None,
        input_labels: torch.LongTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_masks: torch.FloatTensor | None = None,
        multimask_output: bool = True,
        run_propagation_head: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam3_1VideoOutput:
        _ = kwargs
        if input_points is None and input_boxes is None and input_masks is None:
            raise ValueError("At least one prompt input is required for the current SAM 3.1 prototype forward path.")

        backbone_outputs = self.get_image_features(
            pixel_values,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=run_propagation_head,
        )
        interactive_outputs = backbone_outputs.interactive
        image_positional_embeddings = self.get_image_wide_positional_embeddings(
            batch_size=pixel_values.shape[0],
            device=pixel_values.device,
            dtype=interactive_outputs["vision_features"].dtype,
        )

        sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        low_res_multimasks, iou_scores, sam_output_tokens, object_score_logits = self.interactive_sam_mask_decoder(
            image_embeddings=interactive_outputs["vision_features"],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=list(interactive_outputs["backbone_fpn"][:-1]),
        )

        high_res_multimasks = (
            F.interpolate(
                low_res_multimasks.flatten(0, 2).unsqueeze(1).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .view(*low_res_multimasks.shape[:3], self.image_size, self.image_size)
            .to(low_res_multimasks.dtype)
        )

        interactive_pred_masks, interactive_high_res_masks, sam_output_token = self._select_best_interactive_masks(
            low_res_multimasks,
            iou_scores,
            sam_output_tokens,
            high_res_multimasks,
            multimask_output=multimask_output,
        )
        interactive_object_pointer = self.interactive_obj_ptr_proj(sam_output_token)

        propagation_masks = None
        propagation_iou_scores = None
        propagation_object_score_logits = None
        propagation_sam_tokens = None
        if run_propagation_head:
            if pixel_values.shape[0] != 1:
                raise ValueError("The current SAM 3.1 propagation prototype only supports batch_size=1.")

            num_objects = 1
            if input_points is not None:
                num_objects = input_points.shape[1]
            elif input_boxes is not None:
                num_objects = input_boxes.shape[1]
            elif input_masks is not None:
                num_objects = input_masks.shape[1] if input_masks.ndim == 4 else 1

            if num_objects > self.config.multiplex_count:
                raise ValueError(
                    f"num_objects={num_objects} exceeds multiplex_count={self.config.multiplex_count} in the current prototype."
                )

            propagation_outputs = backbone_outputs.sam2_backbone_out
            propagation_decoder_outputs = self.sam_mask_decoder(
                image_embeddings=propagation_outputs["vision_features"],
                image_pe=self.get_image_wide_positional_embeddings(
                    batch_size=1,
                    device=pixel_values.device,
                    dtype=propagation_outputs["vision_features"].dtype,
                )[:1],
                multimask_output=multimask_output,
                high_res_features=list(propagation_outputs["backbone_fpn"][:-1]),
                extra_per_object_embeddings=None,
            )
            propagation_masks = propagation_decoder_outputs["masks"][:, :num_objects]
            propagation_iou_scores = propagation_decoder_outputs["iou_pred"][:, :num_objects]
            propagation_object_score_logits = propagation_decoder_outputs["object_score_logits"][:, :num_objects]
            propagation_sam_tokens = propagation_decoder_outputs["sam_tokens_out"][:, :num_objects]

        return Sam3_1VideoOutput(
            interactive_iou_scores=iou_scores,
            interactive_pred_masks=interactive_pred_masks,
            interactive_high_res_masks=interactive_high_res_masks,
            interactive_object_score_logits=object_score_logits,
            interactive_object_pointer=interactive_object_pointer,
            propagation_masks=propagation_masks,
            propagation_iou_scores=propagation_iou_scores,
            propagation_object_score_logits=propagation_object_score_logits,
            propagation_sam_tokens=propagation_sam_tokens,
            image_embeddings=interactive_outputs["backbone_fpn"],
            backbone_outputs=backbone_outputs,
        )


__all__ = ["Sam3_1VideoConfig", "Sam3_1VideoModel", "Sam3_1VideoPreTrainedModel", "Sam3_1ViTModel"]
