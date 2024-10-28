# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""SAM2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Sam2PromptEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2PromptEncoder`]. The [`Sam2PromptEncoder`]
    module is used to encode the input 2D points and bounding boxes.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        layer_norm_eps (`<fill_type>`, *optional*, defaults to 1e-06): <fill_docstring>
        scale (`<fill_type>`, *optional*, defaults to 1): <fill_docstring>
    """

    def __init__(
        self,
        hidden_size=256,
        image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_embedding_size = image_size // patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale


class Sam2MemoryAttentionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryAttention`]. It is used to instantiate a SAM 2
    memory attention module according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        d_model (`int`, *optional*, defaults to 256):
            The dimension of the model in the memory attention module.
        pos_enc_at_input (`bool`, *optional*, defaults to `True`):
            Whether to apply positional encoding at the input.
        num_layers (`int`, *optional*, defaults to 4):
            The number of layers in the memory attention module.
        batch_first (`bool`, *optional*, defaults to `True`):
            Whether the input and output tensors are provided in batch-first format.

    """

    def __init__(
        self,
        d_model=256,
        pos_enc_at_input=True,
        num_layers=4,
        batch_first=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.pos_enc_at_input = pos_enc_at_input
        self.num_layers = num_layers
        self.batch_first = batch_first


class Sam2MemoryEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryEncoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        in_dim (`int`, *optional*, defaults to 256):
            Input dimension of the memory encoder.
        out_dim (`int`, *optional*, defaults to 64):
            Output dimension of the memory encoder.

    """

    def __init__(
        self,
        in_dim=256,
        out_dim=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim


class Sam2MaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MaskDecoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`<fill_type>`, *optional*, defaults to 256): <fill_docstring>
        num_multimask_outputs (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        hidden_act (`<fill_type>`, *optional*, defaults to `"gelu"`): <fill_docstring>
        iou_head_depth (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        iou_head_hidden_dim (`<fill_type>`, *optional*, defaults to 256): <fill_docstring>
        use_high_res_features (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        iou_prediction_use_sigmoid (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        dynamic_multimask_via_stability (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        dynamic_multimask_stability_delta (`<fill_type>`, *optional*, defaults to 0.05): <fill_docstring>
        dynamic_multimask_stability_thresh (`<fill_type>`, *optional*, defaults to 0.98): <fill_docstring>
        pred_obj_scores (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        pred_obj_scores_mlp (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        use_multimask_token_for_obj_ptr (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        two_way_transformer_depth (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>
        two_way_transformer_embedding_dim (`<fill_type>`, *optional*, defaults to 256): <fill_docstring>
        two_way_transformer_num_heads (`<fill_type>`, *optional*, defaults to 8): <fill_docstring>
        two_way_transformer_mlp_dim (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
        two_way_transformer_activation (`<fill_type>`, *optional*, defaults to `"relu"`): <fill_docstring>
        two_way_transformer_attention_downsample_rate (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>

    """

    def __init__(
        self,
        hidden_size=256,
        num_multimask_outputs=3,
        hidden_act="gelu",
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
        feed_forward_hidden_act="relu",
        two_way_transformer_depth=2,
        two_way_transformer_embedding_dim=256,
        two_way_transformer_num_heads=8,
        two_way_transformer_mlp_dim=2048,
        two_way_transformer_activation="relu",
        two_way_transformer_attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert hidden_size == two_way_transformer_embedding_dim

        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.use_high_res_features = use_high_res_features
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.feed_forward_hidden_act = feed_forward_hidden_act

        # TwoWayTransformer configuration
        self.two_way_transformer_depth = two_way_transformer_depth
        self.two_way_transformer_embedding_dim = two_way_transformer_embedding_dim
        self.two_way_transformer_num_heads = two_way_transformer_num_heads
        self.two_way_transformer_mlp_dim = two_way_transformer_mlp_dim
        self.two_way_transformer_activation = two_way_transformer_activation
        self.two_way_transformer_attention_downsample_rate = two_way_transformer_attention_downsample_rate


class Sam2ImageEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2ImageEncoder`]. It is used to instantiate a SAM
    image encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM 2 Hiera-B+
    [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        skip_lowest_resolutions (`int`, *optional*, defaults to 1):
            The skip_lowest_resolutions parameter for the image encoder.
        hidden_size (`<fill_type>`, *optional*, defaults to 96): <fill_docstring>
        num_heads (`int`, *optional*, defaults to 1):
            Initial number of attention heads.
        num_channels (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        image_size (`<fill_type>`, *optional*, defaults to 1024): <fill_docstring>
        patch_size (`<fill_type>`, *optional*, defaults to 7): <fill_docstring>
        patch_stride (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
        patch_padding (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
        q_pool (`int`, *optional*, defaults to 3):
            Number of q_pool stages.
        q_stride (`Tuple[int, int]`, *optional*, defaults to `(2, 2)`):
            Downsample stride between stages.
        stages (`Tuple[int, ...]`, *optional*, defaults to `(1, 2, 7, 2)`):
            Number of blocks per stage.
        dim_mul (`float`, *optional*, defaults to 2.0):
            Dimension multiplier factor at stage shift.
        head_mul (`float`, *optional*, defaults to 2.0):
            Head multiplier factor at stage shift.
        window_positional_embedding_background_size (`Tuple[int, int]`, *optional*, defaults to `(7, 7)`):
            Window size per stage when not using global attention.
        window_spec (`Tuple[int, ...]`, *optional*, defaults to `(8, 4, 14, 7)`):
            Window specifications for each stage.
        global_attention_blocks (`Tuple[int, ...]`, *optional*, defaults to `(5, 7, 9)`):
            Blocks where global attention is used.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the model in the neck.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`):
            List of channel dimensions for the backbone.
        fpn_kernel_size (`int`, *optional*, defaults to 1):
            Kernel size for convolutions in the neck.
        stride (`int`, *optional*, defaults to 1):
            Stride for convolutions in the neck.
        padding (`int`, *optional*, defaults to 0):
            Padding for convolutions in the neck.
        fpn_top_down_levels (`List[int]`, *optional*, defaults to `[2, 3]`):
            Levels for top-down FPN connections.
        fpn_interpolation_mode (`str`, *optional*, defaults to `"nearest"`):
            Interpolation model for FPN.
        fuse_type (`str`, *optional*, defaults to `"sum"`):
            Type of fusion to use in the neck.
        hidden_act (`<fill_type>`, *optional*, defaults to `"gelu"`): <fill_docstring>
        layer_norm_eps (`<fill_type>`, *optional*, defaults to 1e-06): <fill_docstring>

    """

    def __init__(
        self,
        hidden_size=96,
        num_heads=1,
        num_channels=3,
        image_size=1024,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(1, 2, 7, 2),
        dim_mul=2.0,
        head_mul=2.0,
        window_positional_embedding_background_size=(7, 7),
        window_spec=(8, 4, 14, 7),
        global_attention_blocks=(5, 7, 9),
        skip_lowest_resolutions=1,
        backbone_channel_list=[768, 384, 192, 96],
        fpn_hidden_size=256,
        fpn_kernel_size=1,
        fpn_stride=1,
        fpn_padding=0,
        fpn_top_down_levels=[2, 3],
        fpn_interpolation_mode="nearest",
        fuse_type="sum",
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(stages) == len(window_spec) == len(backbone_channel_list)
        assert fuse_type in ["sum", "avg"]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_positional_embedding_background_size = window_positional_embedding_background_size
        self.window_spec = window_spec
        self.global_attention_blocks = global_attention_blocks
        self.skip_lowest_resolutions = skip_lowest_resolutions

        # Neck
        self.backbone_channel_list = backbone_channel_list
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_kernel_size = fpn_kernel_size
        self.fpn_stride = fpn_stride
        self.fpn_padding = fpn_padding
        self.fpn_top_down_levels = fpn_top_down_levels
        self.fpn_interpolation_mode = fpn_interpolation_mode
        self.fuse_type = fuse_type

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


class Sam2Config(PretrainedConfig):
    r"""
    [`Sam2Config`] is the configuration class to store the configuration of a [`Sam2Model`]. It is used to instantiate a
    SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2 Hiera-B+
    [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_encoder_config (Union[`dict`, `Sam2ImageEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2ImageEncoderConfig`].
        prompt_encoder_config (`<fill_type>`, *optional*): <fill_docstring>
        mask_decoder_config (`<fill_type>`, *optional*): <fill_docstring>
        memory_attention_config (Union[`dict`, `Sam2MemoryAttentionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryAttentionConfig`].
        memory_encoder_config (Union[`dict`, `Sam2MemoryEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryEncoderConfig`].

        initializer_range (`float`, *optional*, defaults to 0.02): std for parameter initialization
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam2ImageEncoderConfig,
    ...     Sam2PromptEncoderConfig,
    ...     Sam2MaskDecoderConfig,
    ...     Sam2MemoryAttentionConfig,
    ...     Sam2MemoryEncoderConfig,
    ...     Sam2Model,
    ... )

    >>> # Initializing a Sam2Config with `"facebook/hiera-base-plus"` style configuration
    >>> configuration = Sam2config()

    >>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = Sam2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam2Config from a Sam2ImageEncoderConfig, Sam2MemoryAttentionConfig, and Sam2MemoryEncoderConfig

    >>> # Initializing SAM2 image encoder, memory attention, and memory encoder configurations
    >>> image_encoder_config = Sam2ImageEncoderConfig()
    >>> prompt_encoder_config = Sam2PromptEncoderConfig()
    >>> mask_decoder_config = Sam2MaskDecoderConfig()
    >>> memory_attention_config = Sam2MemoryAttentionConfig()
    >>> memory_encoder_config = Sam2MemoryEncoderConfig()

    >>> config = Sam2Config(image_encoder_config, prompt_encoder_config, mask_decoder_config, memory_attention_config, memory_encoder_config)
    ```"""

    model_type = "sam2"

    def __init__(
        self,
        image_encoder_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        memory_attention_config=None,
        memory_encoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        image_encoder_config = image_encoder_config if image_encoder_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        memory_attention_config = memory_attention_config if memory_attention_config is not None else {}
        memory_encoder_config = memory_encoder_config if memory_encoder_config is not None else {}

        self.image_encoder_config = Sam2ImageEncoderConfig(**image_encoder_config)
        self.prompt_encoder_config = Sam2PromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam2MaskDecoderConfig(**mask_decoder_config)
        self.memory_attention_config = Sam2MemoryAttentionConfig(**memory_attention_config)
        self.memory_encoder_config = Sam2MemoryEncoderConfig(**memory_encoder_config)

        self.initializer_range = initializer_range
        self.num_maskmem = 7  # default 1 input frame + 6 previous frames
        self.image_size = 1024
        self.backbone_stride = 16  # stride of the image backbone output
        self.sigmoid_scale_for_mem_enc = 20  # scale factor for mask sigmoid prob
        self.sigmoid_bias_for_mem_enc = -10  # bias factor for mask sigmoid prob
        # During evaluation whether to binarize the sigmoid mask logits on interacted frames with clicks
        self.binarize_mask_from_pts_for_mem_enc = False
        self.use_mask_input_as_output_without_sam = True  # on frames with mask input whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        self.max_cond_frames_in_attn = -1
        # on the first frame whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        self.directly_add_no_mem_embed = True
        # whether to use high-resolution feature maps in the SAM mask decoder
        self.use_high_res_features_in_sam = True
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        self.multimask_output_in_sam = True
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both meaning that only the first click gives multimask output; also note that a box counts as two points)
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        self.multimask_output_for_tracking = True
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        self.use_multimask_token_for_obj_ptr = True
        # whether to use sigmoid to restrict ious prediction to [0-1]
        self.iou_prediction_use_sigmoid = True
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1 the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames plus the last frame.
        self.memory_temporal_stride_for_eval = 1
        # if `add_all_frames_to_correct_as_cond` is True we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False we conditioning frame list to only use those initial conditioning frames
        self.add_all_frames_to_correct_as_cond = False
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        self.non_overlap_masks_for_mem_enc = False
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        self.use_obj_ptrs_in_encoder = True
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        self.max_obj_ptrs_in_encoder = 16
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        self.add_tpos_enc_to_obj_ptrs = False
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        self.proj_tpos_enc_in_obj_ptrs = False
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        self.only_obj_ptrs_in_the_past_for_eval = True
        # Whether to predict if there is an object in the frame
        self.pred_obj_scores = True
        # Whether to use an MLP to predict object scores
        self.pred_obj_scores_mlp = True
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        self.fixed_no_obj_ptr = True
        # Soft no object i.e. mix in no_obj_ptr softly
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        self.soft_no_obj_ptr = False
        self.use_mlp_for_obj_ptr_proj = True
        # extra arguments used to construct the SAM mask decoder; if not None it should be a dict of kwargs to be passed into `MaskDecoder` class.
        self.sam_mask_decoder_extra_args = None
        self.compile_image_encoder = False

        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
