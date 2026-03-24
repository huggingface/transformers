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
"""RT-DETR model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PekingU/rtdetr_r50vd")
@strict
class RTDetrConfig(PreTrainedConfig):
    r"""
    initializer_bias_prior_prob (`float`, *optional*):
        The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
        If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
    freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
        Whether to freeze the batch normalization layers in the backbone.
    encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
        Multi level features input for encoder.
    feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`):
        Strides used in each feature map.
    encode_proj_layers (`list[int]`, *optional*, defaults to `[2]`):
        Indexes of the projected layers to be used in the encoder.
    positional_encoding_temperature (`int`, *optional*, defaults to 10000):
        The temperature parameter used to create the positional encodings.
    encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    eval_size (`tuple[int, int]`, *optional*):
        Height and width used to computes the effective height and width of the position embeddings after taking
        into account the stride.
    normalize_before (`bool`, *optional*, defaults to `False`):
        Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
        feed-forward modules.
    hidden_expansion (`float`, *optional*, defaults to 1.0):
        Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
    num_queries (`int`, *optional*, defaults to 300):
        Number of object queries.
    decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
        Multi level features dimension for decoder
    num_feature_levels (`int`, *optional*, defaults to 3):
        The number of input feature levels.
    decoder_n_points (`int`, *optional*, defaults to 4):
        The number of sampled keys in each feature level for each attention head in the decoder.
    decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    num_denoising (`int`, *optional*, defaults to 100):
        The total number of denoising tasks or queries to be used for contrastive denoising.
    label_noise_ratio (`float`, *optional*, defaults to 0.5):
        The fraction of denoising labels to which random noise should be added.
    box_noise_scale (`float`, *optional*, defaults to 1.0):
        Scale or magnitude of noise to be added to the bounding boxes.
    learn_initial_query (`bool`, *optional*, defaults to `False`):
        Indicates whether the initial query embeddings for the decoder should be learned during training
    anchor_image_size (`tuple[int, int]`, *optional*):
        Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
    disable_custom_kernels (`bool`, *optional*, defaults to `True`):
        Whether to disable custom kernels.
    with_box_refine (`bool`, *optional*, defaults to `True`):
        Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
        based on the predictions from the previous layer.
    matcher_alpha (`float`, *optional*, defaults to 0.25):
        Parameter alpha used by the Hungarian Matcher.
    matcher_gamma (`float`, *optional*, defaults to 2.0):
        Parameter gamma used by the Hungarian Matcher.
    matcher_class_cost (`float`, *optional*, defaults to 2.0):
        The relative weight of the class loss used by the Hungarian Matcher.
    matcher_bbox_cost (`float`, *optional*, defaults to 5.0):
        The relative weight of the bounding box loss used by the Hungarian Matcher.
    matcher_giou_cost (`float`, *optional*, defaults to 2.0):
        The relative weight of the giou loss of used by the Hungarian Matcher.
    use_focal_loss (`bool`, *optional*, defaults to `True`):
        Parameter informing if focal focal should be used.
    focal_loss_alpha (`float`, *optional*, defaults to 0.75):
        Parameter alpha used to compute the focal loss.
    focal_loss_gamma (`float`, *optional*, defaults to 2.0):
        Parameter gamma used to compute the focal loss.
    weight_loss_vfl (`float`, *optional*, defaults to 1.0):
        Relative weight of the varifocal loss in the object detection loss.
    weight_loss_bbox (`float`, *optional*, defaults to 5.0):
        Relative weight of the L1 bounding box loss in the object detection loss.
    weight_loss_giou (`float`, *optional*, defaults to 2.0):
        Relative weight of the generalized IoU loss in the object detection loss.

    Examples:

    ```python
    >>> from transformers import RTDetrConfig, RTDetrModel

    >>> # Initializing a RT-DETR configuration
    >>> configuration = RTDetrConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RTDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rt_detr"
    sub_configs = {"backbone_config": AutoConfig}
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    backbone_config: dict | PreTrainedConfig | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: int | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    d_model: int = 256
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    learn_initial_query: bool = False
    anchor_image_size: int | list[int] | None = None
    disable_custom_kernels: bool = True
    with_box_refine: bool = True
    is_encoder_decoder: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0
    matcher_class_cost: float = 2.0
    matcher_bbox_cost: float = 5.0
    matcher_giou_cost: float = 2.0
    use_focal_loss: bool = True
    auxiliary_loss: bool = True
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    weight_loss_vfl: float = 1.0
    weight_loss_bbox: float = 5.0
    weight_loss_giou: float = 2.0
    eos_coefficient: float = 1e-4

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="rt_detr_resnet",
            default_config_kwargs={"out_indices": [2, 3, 4]},
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["RTDetrConfig"]
