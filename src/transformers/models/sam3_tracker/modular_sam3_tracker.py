# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import torch

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoModel
from ..sam2.configuration_sam2 import (
    Sam2Config,
    Sam2MaskDecoderConfig,
    Sam2PromptEncoderConfig,
)
from ..sam2.modeling_sam2 import (
    Sam2Attention,
    Sam2FeedForward,
    Sam2ImageSegmentationOutput,
    Sam2LayerNorm,
    Sam2MaskDecoder,
    Sam2MaskEmbedding,
    Sam2Model,
    Sam2PositionalEmbedding,
    Sam2PreTrainedModel,
    Sam2PromptEncoder,
    Sam2TwoWayAttentionBlock,
    Sam2TwoWayTransformer,
)
from ..sam2.processing_sam2 import Sam2Processor


class Sam3TrackerPromptEncoderConfig(Sam2PromptEncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam3TrackerPromptEncoder`]. The [`Sam3TrackerPromptEncoder`]
    module is used to encode the input 2D points and bounding boxes.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1008):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        scale (`float`, *optional*, defaults to 1):
            The scale factor for the prompt encoder.
    """

    base_config_key = "prompt_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        image_size=1008,
        patch_size=14,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)


class Sam3TrackerProcessor(Sam2Processor):
    pass


class Sam3TrackerMaskDecoderConfig(Sam2MaskDecoderConfig):
    pass


class Sam3TrackerConfig(Sam2Config):
    r"""
    [`Sam3TrackerConfig`] is the configuration class to store the configuration of a [`Sam3TrackerModel`]. It is used to instantiate a
    SAM3_TRACKER model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2.1 Hiera-tiny
    [facebook/sam3_tracker.1-hiera-tiny](https://huggingface.co/facebook/sam3_tracker.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    <Tip>

    SAM3 Tracker checkpoints with `model_type="sam3_tracker_video"` are compatible with `Sam3TrackerModel` since the
    video variant weights are a superset of the image-only model weights. You may see a warning about model type
    mismatch when loading such checkpoints, which can be safely ignored in this case.

    </Tip>

    Args:
        vision_config (Union[`dict`, `Sam3TrackerVisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam3TrackerVisionConfig`].
        prompt_encoder_config (Union[`dict`, `Sam3TrackerPromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam3TrackerPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `Sam3TrackerMaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam3TrackerMaskDecoderConfig`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for parameter initialization.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam3TrackerVisionConfig,
    ...     Sam3TrackerPromptEncoderConfig,
    ...     Sam3TrackerMaskDecoderConfig,
    ...     Sam3TrackerModel,
    ... )

    >>> # Initializing a Sam3TrackerConfig with `"facebook/sam3_tracker.1_hiera_tiny"` style configuration
    >>> configuration = Sam3TrackerConfig()

    >>> # Initializing a Sam3TrackerModel (with random weights) from the `"facebook/sam3_tracker.1_hiera_tiny"` style configuration
    >>> model = Sam3TrackerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam3TrackerConfig from a Sam3TrackerVisionConfig, Sam3TrackerPromptEncoderConfig, and Sam3TrackerMaskDecoderConfig
    >>> # Initializing SAM3_TRACKER vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = Sam3TrackerVisionConfig()
    >>> prompt_encoder_config = Sam3TrackerPromptEncoderConfig()
    >>> mask_decoder_config = Sam3TrackerMaskDecoderConfig()

    >>> config = Sam3TrackerConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    ```
    """

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        vision_config = (
            vision_config
            if vision_config is not None
            else {"backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]]}
        )
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "sam3_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        if isinstance(prompt_encoder_config, Sam3TrackerPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam3TrackerMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = vision_config
        self.prompt_encoder_config = Sam3TrackerPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam3TrackerMaskDecoderConfig(**mask_decoder_config)

        self.initializer_range = initializer_range
        PreTrainedConfig.__init__(**kwargs)


class Sam3TrackerImageSegmentationOutput(Sam2ImageSegmentationOutput):
    pass


class Sam3TrackerFeedForward(Sam2FeedForward):
    pass


@auto_docstring(
    custom_intro="""
    Segment Anything Model 3 (SAM 3) for generating segmentation masks, given an input image and
    input points and labels, boxes, or masks.
    """
)
class Sam3TrackerPreTrainedModel(Sam2PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, Sam3TrackerModel):
            if module.no_memory_embedding is not None:
                init.zeros_(module.no_memory_embedding)
        elif isinstance(module, Sam3TrackerPositionalEmbedding):
            init.normal_(module.positional_embedding, std=module.scale)


class Sam3TrackerPositionalEmbedding(Sam2PositionalEmbedding):
    pass


class Sam3TrackerMaskEmbedding(Sam2MaskEmbedding):
    pass


class Sam3TrackerPromptEncoder(Sam2PromptEncoder):
    pass


class Sam3TrackerAttention(Sam2Attention):
    pass


class Sam3TrackerTwoWayAttentionBlock(Sam2TwoWayAttentionBlock):
    pass


class Sam3TrackerTwoWayTransformer(Sam2TwoWayTransformer):
    pass


class Sam3TrackerLayerNorm(Sam2LayerNorm):
    pass


class Sam3TrackerMaskDecoder(Sam2MaskDecoder):
    pass


class Sam3TrackerModel(Sam2Model):
    _checkpoint_conversion_mapping = {
        r"tracker_model.(.+)": r"\1",  # the regex allows to remove the prefix, and add it back in revert mode
        "detector_model.vision_encoder.backbone.": "vision_encoder.backbone.",
        "tracker_neck.": "vision_encoder.neck.",
    }
    _keys_to_ignore_on_load_unexpected = [
        r"^detector_model.",
        r"^memory_.*",
        r"^mask_downsample.*",
        r"^object_pointer_proj.*",
        r"^temporal_positional_encoding_projection_layer.*",
        "no_memory_positional_encoding",
        "no_object_pointer",
        "occlusion_spatial_embedding_parameter",
    ]

    def __init__(self, config: Sam3TrackerConfig):
        # loading from a sam3_video config
        if hasattr(config, "tracker_config") and config.tracker_config is not None:
            if isinstance(config.tracker_config, dict):
                config.tracker_config = Sam3TrackerConfig(**config.tracker_config)
            config = config.tracker_config
        Sam3TrackerPreTrainedModel.__init__(config)
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(config.prompt_encoder_config)
        self.vision_encoder = AutoModel.from_config(config.vision_config)
        self.prompt_encoder = Sam3TrackerPromptEncoder(config.prompt_encoder_config)
        # The module using it is not a PreTrainedModel subclass so we need this
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = Sam3TrackerMaskDecoder(config.mask_decoder_config)

        self.backbone_feature_sizes = config.vision_config.backbone_feature_sizes
        # a single token to indicate no memory embedding from previous frames
        self.hidden_dim = config.vision_config.fpn_hidden_size
        self.no_memory_embedding = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.post_init()


__all__ = [
    "Sam3TrackerConfig",
    "Sam3TrackerPromptEncoderConfig",
    "Sam3TrackerMaskDecoderConfig",
    "Sam3TrackerProcessor",
    "Sam3TrackerModel",
    "Sam3TrackerPreTrainedModel",
]
