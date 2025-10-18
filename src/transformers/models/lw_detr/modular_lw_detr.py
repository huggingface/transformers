import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import meshgrid
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..conditional_detr.modeling_conditional_detr import (
    ConditionalDetrConvEncoder,
    ConditionalDetrConvModel,
    ConditionalDetrSinePositionEmbedding,
)
from ..convnext.modeling_convnext import ConvNextLayerNorm
from ..dab_detr.modeling_dab_detr import gen_sine_position_embeddings
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoderOutput,
    DeformableDetrForObjectDetection,
    DeformableDetrLearnedPositionEmbedding,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrMultiscaleDeformableAttention,
    _get_clones,
)
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from ..rt_detr.configuration_rt_detr import CONFIG_MAPPING, verify_backbone_config_arguments
from ..rt_detr.modeling_rt_detr import RTDetrConvNormLayer
from .configuration_lw_detr_vit import LwDetrViTConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the DeformableDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    """
)
class LwDetrDecoderOutput(DeformableDetrDecoderOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the LWDETR backbone-decoder model.
    """
)
class LwDetrModelOutput(ModelOutput):
    r"""
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    """

    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`LwDetrForObjectDetection`].
    """
)
class LwDetrObjectDetectionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~DeformableDetrProcessor.post_process_object_detection`] to retrieve the
        unnormalized bounding boxes.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[list[dict]] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    enc_outputs_class: Any = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


class LwDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LwDetrModel`]. It is used to instantiate
    a LW-DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LW-DETR
    [stevenbucaille/lwdetr_small_60e_coco](https://huggingface.co/stevenbucaille/lwdetr_small_60e_coco) architecture.

    LW-DETR (Lightweight Detection Transformer) is a transformer-based object detection model designed for real-time
    detection tasks. It replaces traditional CNN-based detectors like YOLO with a more efficient transformer architecture
    that achieves competitive performance while being computationally lightweight.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. If not provided, will default to `LwDetrViTConfig` with
            a small ViT architecture optimized for detection tasks.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. Only used when `use_timm_backbone` is `True`.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`] API.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint.
        projector_scale_factors (`list[float]`, *optional*, defaults to `[]`):
            Scale factors for the feature pyramid network. Each scale factor determines the resolution of features
            at different levels. Supported values are 0.5, 1.0, and 2.0.
        hidden_expansion (`float`, *optional*, defaults to 0.5):
            Expansion factor for hidden dimensions in the projector layers.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the projector. Supported values are `"silu"`, `"relu"`, `"gelu"`.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization layers.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the model layers and the number of expected features in the decoder inputs.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers in the transformer.
        decoder_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the decoder self-attention.
        decoder_cross_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder cross-attention.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the decoder. Supported values are `"relu"`, `"silu"`, `"gelu"`.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`LwDetrModel`] can detect in a single image.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        group_detr (`int`, *optional*, defaults to 13):
            Number of groups for Group DETR attention mechanism, which helps reduce computational complexity.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.

    Examples:

    ```python
    >>> from transformers import LwDetrConfig, LwDetrModel

    >>> # Initializing a LW-DETR stevenbucaille/lwdetr_small_60e_coco style configuration
    >>> configuration = LwDetrConfig()

    >>> # Initializing a model (with random weights) from the stevenbucaille/lwdetr_small_60e_coco style configuration
    >>> model = LwDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lw_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "decoder_self_attention_heads",
        "num_key_value_heads": "decoder_self_attention_heads",
    }

    def __init__(
        self,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        # projector
        projector_scale_factors: list[float] = [],
        hidden_expansion=0.5,
        activation_function="silu",
        batch_norm_eps=1e-5,
        # decoder
        d_model=256,
        dropout=0.1,
        decoder_ffn_dim=2048,
        decoder_n_points=4,
        decoder_layers: int = 3,
        decoder_self_attention_heads: int = 8,
        decoder_cross_attention_heads: int = 16,
        decoder_activation_function="relu",
        # model
        num_queries=300,
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        position_embedding_type="sine",
        group_detr: int = 13,
        init_std=0.02,
        disable_custom_kernels=True,
        # loss
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        auxiliary_loss=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_norm_eps = batch_norm_eps

        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `LwDetrViT` backbone."
            )
            backbone_config = LwDetrViTConfig(
                image_size=1024,
                hidden_size=192,
                num_hidden_layers=10,
                num_attention_heads=12,
                window_block_indices=[0, 1, 3, 6, 7, 9],
                out_indices=[2, 4, 5, 9],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        # projector
        self.projector_scale_factors = projector_scale_factors
        for scale in projector_scale_factors:
            if scale not in [0.5, 1.0, 2.0]:
                raise ValueError(f"Unsupported scale factor: {scale}")
        self.projector_in_channels = [d_model] * len(projector_scale_factors)
        self.projector_out_channels = d_model
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion
        # decoder
        self.d_model = d_model
        self.dropout = dropout
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = len(self.projector_scale_factors)
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_activation_function = decoder_activation_function
        self.decoder_self_attention_heads = decoder_self_attention_heads
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        # model
        self.position_embedding_type = position_embedding_type
        self.init_std = init_std
        self.group_detr = group_detr
        # Loss
        self.auxiliary_loss = auxiliary_loss
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @property
    def num_attention_heads(self) -> int:
        return self.decoder_self_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.decoder_self_attention_heads

    @property
    def sub_configs(self):
        return (
            {"backbone_config": type(self.backbone_config)}
            if getattr(self, "backbone_config", None) is not None
            else {}
        )


class LwDetrConvNormLayer(RTDetrConvNormLayer):
    def __init__(
        self,
        config: LwDetrConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: Optional[str] = None,
    ):
        super().__init__(config, in_channels, out_channels, kernel_size, stride, activation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        )


class LwDetrRepVggBlock(nn.Module):
    def __init__(self, config: LwDetrConfig, hidden_channels: int):
        super().__init__()
        self.conv1 = LwDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, activation="silu")
        self.conv2 = LwDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, activation="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class LwDetrC2FLayer(nn.Module):
    # Inspired by RTDetrCSPRepLayer
    def __init__(self, config: LwDetrConfig, in_channels: int, out_channels: int):
        super().__init__()
        num_blocks = 3
        activation = config.activation_function

        self.hidden_channels = int(out_channels * config.hidden_expansion)
        conv1_out_channels = 2 * self.hidden_channels
        conv2_in_channels = (2 + num_blocks) * self.hidden_channels
        self.conv1 = LwDetrConvNormLayer(config, in_channels, conv1_out_channels, 1, 1, activation=activation)
        self.conv2 = LwDetrConvNormLayer(config, conv2_in_channels, out_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.ModuleList(LwDetrRepVggBlock(config, self.hidden_channels) for _ in range(num_blocks))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        all_hidden_states = list(hidden_states.split(self.hidden_channels, 1))
        hidden_states = all_hidden_states[-1]

        for bottleneck in self.bottlenecks:
            hidden_states = bottleneck(hidden_states)
            all_hidden_states.append(hidden_states)

        hidden_states = torch.cat(all_hidden_states, 1)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class LwDetrLayerNorm(ConvNextLayerNorm):
    pass


class LwDetrSamplingLayer(nn.Module):
    def __init__(self, config: LwDetrConfig, channel_size: int, scale: float):
        super().__init__()

        self.scale = scale
        self.channel_size = channel_size

        layers = []
        if scale == 2.0:
            if channel_size > 512:
                layers.append(LwDetrConvNormLayer(config, channel_size, channel_size // 2, 1, 1, activation="relu"))
                layers.append(nn.ConvTranspose2d(channel_size // 2, channel_size // 4, kernel_size=2, stride=2))
            else:
                layers.append(nn.ConvTranspose2d(channel_size, channel_size // 2, 2, 2))
        elif scale == 0.5:
            layers.append(LwDetrConvNormLayer(config, channel_size, channel_size, 3, 2, activation="relu"))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class LwDetrScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig, intermediate_dims: list[int], scale: float, output_dim: int):
        super().__init__()

        sampling_layers = []
        for channel_size in intermediate_dims:
            sampling_layers.append(LwDetrSamplingLayer(config, channel_size, scale))
        self.sampling_layers = nn.ModuleList(sampling_layers)

        intermediate_dim = intermediate_dims[-1]
        if scale == 2.0:
            if intermediate_dim > 512:
                intermediate_dim = intermediate_dim // 4
            else:
                intermediate_dim = intermediate_dim // 2
        projector_input_dim = intermediate_dim * len(intermediate_dims)

        self.projector_layer = LwDetrC2FLayer(config, projector_input_dim, output_dim)
        self.layer_norm = LwDetrLayerNorm(output_dim, data_format="channels_first")

    def forward(self, hidden_states_tuple: tuple[torch.Tensor]) -> torch.Tensor:
        sampled_hidden_states = []
        for sampling_layer, hidden_states in zip(self.sampling_layers, hidden_states_tuple):
            hidden_states = sampling_layer(hidden_states)
            sampled_hidden_states.append(hidden_states)
        hidden_states = torch.cat(sampled_hidden_states, dim=1)
        hidden_states = self.projector_layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LwDetrMultiScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig, intermediate_channel_sizes: list[int]):
        super().__init__()

        self.config = config
        scale_factors = config.projector_scale_factors
        output_channels = config.d_model

        self.scale_layers = nn.ModuleList(
            [
                LwDetrScaleProjector(config, intermediate_channel_sizes, scale, output_channels)
                for scale in scale_factors
            ]
        )

    def forward(self, hidden_states: tuple[torch.Tensor]) -> list[torch.Tensor]:
        output_hidden_states = []
        for scale_layer in self.scale_layers:
            output_hidden_states.append(scale_layer(hidden_states))
        return output_hidden_states


class LwDetrConvEncoder(ConditionalDetrConvEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.projector = LwDetrMultiScaleProjector(config, self.intermediate_channel_sizes)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps
        features = self.projector(features)
        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class LwDetrConvModel(ConditionalDetrConvModel):
    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class LwDetrAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_len, embed_dim = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = hidden_states if position_embeddings is None else hidden_states + position_embeddings

        if self.training:
            hidden_states_original = torch.cat(
                hidden_states_original.split(seq_len // self.config.group_detr, dim=1), dim=0
            )
            hidden_states = torch.cat(hidden_states.split(seq_len // self.config.group_detr, dim=1), dim=0)

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states_original).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            attn_output = torch.cat(torch.split(attn_output, batch_size, dim=0), dim=1)

        return attn_output, attn_weights


class LwDetrMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )


class LwDetrMLP(nn.Module):
    def __init__(self, config: LwDetrConfig):
        super().__init__()
        # feedforward neural networks
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LwDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LwDetrConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)

        # self-attention
        self.self_attn = LwDetrAttention(config, layer_idx=layer_idx)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)

        # cross-attention
        self.cross_attn = LwDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_cross_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)

        # ffn
        self.ffn = LwDetrMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attention_output, self_attn_weights = self.self_attn(
            hidden_states, position_embeddings=position_embeddings, **kwargs
        )

        self_attention_output = nn.functional.dropout(self_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + self_attention_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attention_output, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )
        cross_attention_output = nn.functional.dropout(cross_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + cross_attention_output
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)

        return hidden_states


class LwDetrLearnedPositionEmbedding(DeformableDetrLearnedPositionEmbedding):
    pass


class LwDetrPreTrainedModel(PreTrainedModel):
    config: LwDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [
        r"LwDetrConvEncoder",
        r"LwDetrDecoderLayer",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": [LwDetrAttention, LwDetrMultiscaleDeformableAttention],
        "hidden_states": [LwDetrDecoderLayer],
    }

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, LwDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, LwDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)
        if hasattr(module, "refpoint_embed") and module.refpoint_embed is not None:
            nn.init.constant_(module.refpoint_embed.weight.data, 0)
        if hasattr(module, "class_embed") and module.class_embed is not None:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(self.config.num_labels) * bias_value
        if hasattr(module, "bbox_embed") and module.bbox_embed is not None:
            nn.init.constant_(module.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(module.bbox_embed.layers[-1].bias.data, 0)


def refine_bboxes(reference_points, deltas):
    new_reference_points_cxcy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
    new_reference_points_wh = deltas[..., 2:].exp() * reference_points[..., 2:]
    new_reference_points = torch.cat((new_reference_points_cxcy, new_reference_points_wh), -1)
    return new_reference_points


class LwDetrDecoder(LwDetrPreTrainedModel):
    def __init__(self, config: LwDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([LwDetrDecoderLayer(config, i) for i in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        self.ref_point_head = LwDetrMLPPredictionHead(2 * config.d_model, config.d_model, config.d_model, num_layers=2)

        self.post_init()

    def get_reference(self, reference_points, valid_ratios):
        # batch_size, num_queries, batch_size, 4
        obj_center = reference_points[..., :4]

        # batch_size, num_queries, num_levels, 4
        reference_points_inputs = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        # batch_size, num_queries, d_model * 2
        query_sine_embed = gen_sine_position_embeddings(reference_points_inputs[:, :, 0, :], self.config.d_model)

        # batch_size, num_queries, d_model
        query_pos = self.ref_point_head(query_sine_embed)
        return reference_points_inputs, query_pos

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        spatial_shapes_list: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        valid_ratios: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        intermediate = ()
        intermediate_reference_points = (reference_points,)

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        reference_points_inputs, query_pos = self.get_reference(reference_points, valid_ratios)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=query_pos,
                reference_points=reference_points_inputs,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                **kwargs,
            )
            intermediate_hidden_states = self.layernorm(hidden_states)
            intermediate += (intermediate_hidden_states,)

        intermediate = torch.stack(intermediate)
        intermediate_reference_points = torch.stack(intermediate_reference_points)

        return LwDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )


class LwDetrSinePositionEmbedding(ConditionalDetrSinePositionEmbedding):
    pass


class LwDetrModel(DeformableDetrModel):
    def __init__(self, config: LwDetrConfig):
        LwDetrPreTrainedModel.__init__(config)

        # Create backbone + positional encoding
        backbone = LwDetrConvEncoder(config)
        position_embeddings = LwDetrSinePositionEmbedding(config.d_model // 2, normalize=True)
        self.backbone = LwDetrConvModel(backbone, position_embeddings)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = LwDetrDecoder(config)

        self.enc_output = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)])
        # Should normally be None and then instantiated in the ForObjectDetection class
        self.enc_out_bbox_embed = nn.ModuleList(
            [LwDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(self.group_detr)]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(self.group_detr)]
        )

        self.post_init()

    # Copied from modeling_detr.DeformableDetrModel.gen_encoder_output_proposals
    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (list[tuple[int, int]]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_height), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        return object_query, output_proposals

    @auto_docstring
    @check_model_inputs
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LwDetrModelOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/lwdetr_small_60e_coco")
        >>> model = DeformableDetrModel.from_pretrained("stevenbucaille/lwdetr_small_60e_coco")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(source)
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        if self.training:
            reference_points = self.reference_point_embed.weight
            query_feat = self.query_feat.weight
        else:
            # only use one group in inference
            reference_points = self.reference_point_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        spatial_shapes_list = []
        for source, mask, pos_embed in zip(sources, masks, position_embeddings_list):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
            source_flatten, ~mask_flatten, spatial_shapes_list
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries
        topk_coords_logits = []
        topk_coords_logits_undetach = []
        object_query_undetach = []

        for group_id in range(group_detr):
            group_object_query = self.enc_output[group_id](object_query_embedding)
            group_object_query = self.enc_output_norm[group_id](group_object_query)

            group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)
            group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
            group_enc_outputs_coord = refine_bboxes(output_proposals, group_delta_bbox)

            group_topk_proposals = torch.topk(group_enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            group_topk_coords_logits_undetach = torch.gather(
                group_enc_outputs_coord,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            group_topk_coords_logits = group_topk_coords_logits_undetach.detach()
            group_object_query_undetach = torch.gather(
                group_object_query, 1, group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.config.d_model)
            )

            topk_coords_logits.append(group_topk_coords_logits)
            topk_coords_logits_undetach.append(group_topk_coords_logits_undetach)
            object_query_undetach.append(group_object_query_undetach)

        topk_coords_logits = torch.cat(topk_coords_logits, 1)
        topk_coords_logits_undetach = torch.cat(topk_coords_logits_undetach, 1)
        object_query_undetach = torch.cat(object_query_undetach, 1)

        topk_coords_logits = topk_coords_logits.sigmoid()
        enc_outputs_class = object_query_undetach
        enc_outputs_coord_logits = topk_coords_logits

        reference_points = refine_bboxes(topk_coords_logits_undetach, reference_points)

        init_reference_points = reference_points
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return LwDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class LwDetrMLPPredictionHead(DeformableDetrMLPPredictionHead):
    pass


class LwDetrForObjectDetection(DeformableDetrForObjectDetection):
    _tied_weights_keys = None

    def __init__(self, config: LwDetrConfig):
        LwDetrPreTrainedModel.__init__(config)
        self.model = LwDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = LwDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        self.model.enc_out_bbox_embed = _get_clones(self.bbox_embed, config.group_detr)
        self.model.enc_out_class_embed = _get_clones(self.class_embed, config.group_detr)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[list[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LwDetrObjectDetectionOutput:
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            **kwargs,
        )

        hidden_states = outputs.intermediate_hidden_states
        reference_points = outputs.intermediate_reference_points
        enc_outputs_class_logits = outputs.enc_outputs_class
        enc_outputs_boxes_logits = outputs.enc_outputs_coord_logits

        outputs_coord_delta = self.bbox_embed(hidden_states)
        outputs_coord = refine_bboxes(reference_points, outputs_coord_delta)
        outputs_coord = outputs_coord.sigmoid()

        outputs_class = self.class_embed(hidden_states)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        enc_outputs_class_logits_list = enc_outputs_class_logits.split(self.config.num_queries, dim=1)
        pred_class = []
        group_detr = self.config.group_detr if self.training else 1
        for group_index in range(group_detr):
            group_pred_class = self.model.enc_out_class_embed[group_index](enc_outputs_class_logits_list[group_index])
            pred_class.append(group_pred_class)
        enc_outputs_class_logits = torch.cat(pred_class, dim=1)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_outputs_class_logits,
                enc_outputs_boxes_logits,
            )

        return LwDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord_logits=enc_outputs_boxes_logits,
        )


__all__ = [
    "LwDetrConfig",
    "LwDetrPreTrainedModel",
    "LwDetrModel",
    "LwDetrForObjectDetection",
]
