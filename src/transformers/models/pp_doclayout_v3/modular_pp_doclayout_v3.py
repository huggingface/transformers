# Copyright 2025 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2.functional import InterpolationMode

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    SizeDict,
)
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    filter_out_non_signature_kwargs,
    logging,
)
from ...utils.backbone_utils import verify_backbone_config_arguments
from ...utils.generic import TensorType
from ..auto import CONFIG_MAPPING, AutoConfig
from ..rt_detr.modeling_rt_detr import (
    RTDetrDecoder,
    RTDetrDecoderOutput,
    RTDetrForObjectDetection,
    RTDetrHybridEncoder,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrModelOutput,
    RTDetrMultiscaleDeformableAttention,
    get_contrastive_denoising_training_group,
    inverse_sigmoid,
)


logger = logging.get_logger(__name__)


def _default_id2label() -> dict[int, str]:
    return {
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header",
        14: "image",
        15: "formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "text",
        24: "vision_footnote",
    }


class PPDocLayoutV3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PP-DocLayoutV3`]. It is used to instantiate a
    RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PP-DocLayoutV3
    [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `RTDetrResNetConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
            Whether to freeze the batch normalization layers in the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Dimension of the layers in hybrid encoder.
        encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
            Multi level features input for encoder.
        feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        encode_proj_layers (`list[int]`, *optional*, defaults to `[2]`):
            Indexes of the projected layers to be used in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The temperature parameter used to create the positional encodings.
        encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        eval_size (`tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
            feed-forward modules.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
        mask_feat_channels (`list[int]`, *optional*, defaults to `[64, 64]):
            The channels of the multi-level features for mask enhancement.
        x4_feat_dim (`int`, *optional*, defaults to 128):
            The dimension of the x4 feature map.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers exclude hybrid encoder.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries.
        decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
            Multi level features dimension for decoder
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of input feature levels.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training
        mask_enhanced (`bool`, *optional*, defaults to `True`):
            Whether to use enhanced masked attention.
        anchor_image_size (`tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Whether to disable custom kernels.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
        gp_head_size (`int`, *optional*, defaults to 64):
            The size of the global pointer head.
        tril_mask (`bool`, *optional*, defaults to `True`):
            Whether to mask out the upper triangular part of the attention matrix.
        id2label (`dict[int, str]`, *optional*):
            Mapping from class id to class name.

    Examples:

    ```python
    >>> from transformers import PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection

    >>> # Initializing a PP-DocLayoutV3 configuration
    >>> configuration = PPDocLayoutV3Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPDocLayoutV3ForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pp_doclayout_v2"
    sub_configs = {"backbone_config": AutoConfig}

    layer_types = ("basic", "bottleneck")
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        initializer_range=0.01,
        initializer_bias_prior_prob=None,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        freeze_backbone_batch_norms=True,
        backbone_kwargs=None,
        # encoder PPDocLayoutV3HybridEncoder
        encoder_hidden_dim=256,
        encoder_in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        hidden_expansion=1.0,
        mask_feat_channels=[64, 64],
        x4_feat_dim=128,
        # decoder PPDocLayoutV3Transformer
        d_model=256,
        num_prototypes=32,
        label_noise_ratio=0.4,
        box_noise_scale=0.4,
        mask_enhanced=True,
        num_queries=300,
        decoder_in_channels=[256, 256, 256],
        decoder_ffn_dim=1024,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=6,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=100,
        learn_initial_query=False,
        anchor_image_size=None,
        disable_custom_kernels=True,
        is_encoder_decoder=True,
        gp_head_size=64,
        tril_mask=True,
        # label
        id2label=None,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps

        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `HGNetV3` backbone."
            )
            backbone_config = {
                "model_type": "hgnet_v2",
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            }
            config_class = CONFIG_MAPPING["hgnet_v2"]
            backbone_config = config_class.from_dict(backbone_config)
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            if backbone_model_type is None:
                raise ValueError("`backbone_config` dict must contain key `model_type`.")
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
        self.freeze_backbone_batch_norms = freeze_backbone_batch_norms
        self.backbone_kwargs = dict(backbone_kwargs) if backbone_kwargs is not None else None

        # ---- encoder ----
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = list(encoder_in_channels)
        self.feat_strides = list(feat_strides)
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = list(encode_proj_layers)
        self.positional_encoding_temperature = positional_encoding_temperature
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.eval_size = list(eval_size) if eval_size is not None else None
        self.normalize_before = normalize_before
        self.hidden_expansion = hidden_expansion
        self.mask_feat_channels = mask_feat_channels
        self.x4_feat_dim = x4_feat_dim

        # ---- decoder ----
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_prototypes = num_prototypes
        self.decoder_in_channels = list(decoder_in_channels)
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.mask_enhanced = mask_enhanced
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = list(anchor_image_size) if anchor_image_size is not None else None
        self.disable_custom_kernels = disable_custom_kernels
        self.gp_head_size = gp_head_size
        self.tril_mask = tril_mask

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

        if kwargs.get("num_labels") and not id2label:
            self.id2label = None
            self.num_labels = kwargs["num_labels"]
        else:
            self.id2label = {int(k): v for k, v in id2label.items()} if id2label else _default_id2label()
            self.num_labels = len(self.id2label)


def postprocess(outputs, threshold, target_sizes):
    boxes = outputs.pred_boxes
    logits = outputs.logits
    order_logits = outputs.order_logits

    order_seqs = get_order(order_logits)

    cxcy, wh = torch.split(boxes, 2, dim=-1)
    boxes = torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if isinstance(target_sizes, list):
            img_h, img_w = torch.as_tensor(target_sizes).unbind(1)
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    num_top_queries = logits.shape[1]
    num_classes = logits.shape[2]

    scores = torch.nn.functional.sigmoid(logits)
    scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    labels = index % num_classes
    index = index // num_classes
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    order_seqs = order_seqs.gather(dim=1, index=index)

    results = []
    for score, label, box, order_seq in zip(scores, labels, boxes, order_seqs):
        order_seq = order_seq[score >= threshold]
        order_seq, indices = torch.sort(order_seq)
        results.append(
            {
                "scores": score[score >= threshold][indices],
                "labels": label[score >= threshold][indices],
                "boxes": box[score >= threshold][indices],
                "order_seq": order_seq,
            }
        )

    return results


def get_order(order_logits):
    # order_logits: (B, N, N) upper-triangular meaningful
    order_scores = torch.sigmoid(order_logits)
    B, N, _ = order_scores.shape

    one = torch.ones((N, N), dtype=order_scores.dtype, device=order_scores.device)
    upper = torch.triu(one, 1)
    lower = torch.tril(one, -1)

    Q = order_scores * upper + (1.0 - order_scores.transpose(1, 2)) * lower
    order_votes = Q.sum(dim=1)  # (B, N)

    order_pointers = torch.argsort(order_votes, dim=1)  # (B, N)
    order_seq = torch.full_like(order_pointers, -1)
    batch = torch.arange(B, device=order_pointers.device)[:, None]
    order_seq[batch, order_pointers] = torch.arange(N, device=order_pointers.device)[None, :]

    return order_seq


class PPDocLayoutV3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a PPDocLayoutV3 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict[str, int]` *optional*, defaults to `{"height": 640, "width": 640}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method. Available options are:
                - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                    Do NOT keep the aspect ratio.
                - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                    less or equal to `longest_edge`.
                - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                    `max_width`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0, 0, 0]`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `[1, 1, 1]`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0, 0, 0],
        image_std: Optional[Union[float, list[float]]] = [1, 1, 1],
        **kwargs,
    ) -> None:
        size = size if size is not None else {"height": 800, "width": 800}
        size = get_size_dict(size, default_to_square=False)

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to self.size):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `list[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, default_to_square=True)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # transformations
        if do_resize:
            images = [
                resize(
                    image, size=(size["height"], size["width"]), resample=resample, input_data_format=input_data_format
                )
                for image in images
            ]

        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]
        encoded_inputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: Optional[Union[TensorType, list[tuple]]] = None,
    ):
        """
        Converts the raw output of [`PPDocLayoutV3ForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`PPDocLayoutV3ObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.

        Returns:
            `list[Dict]`: An ordered list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        return postprocess(outputs=outputs, threshold=threshold, target_sizes=target_sizes)


class PPDocLayoutV3ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    size = {"height": 800, "width": 800}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional[InterpolationMode],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.
        """
        data = {}
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size=size, interpolation=interpolation)
            # Fused rescale and normalize
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            processed_images.append(image)

        images = processed_images

        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: Optional[Union[TensorType, list[tuple]]] = None,
    ):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        return postprocess(outputs=outputs, threshold=threshold, target_sizes=target_sizes)


class GlobalPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.gp_head_size
        self.tril_mask = config.tril_mask
        self.dense = nn.Linear(config.d_model, self.head_size * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        B, N, _ = inputs.shape
        proj = self.dense(inputs).reshape([B, N, 2, self.head_size])
        proj = self.dropout(proj)
        qw, kw = proj[..., 0, :], proj[..., 1, :]

        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / (self.head_size**0.5)  # [B, N, N]

        if self.tril_mask:
            lower = torch.tril(torch.ones([N, N], dtype=torch.float32, device=logits.device))
            logits = logits - lower.unsqueeze(0).to(logits.dtype) * 1e4

        return logits


class PPDocLayoutV3MultiscaleDeformableAttention(RTDetrMultiscaleDeformableAttention):
    pass


@auto_docstring
class PPDocLayoutV3PreTrainedModel(PreTrainedModel):
    config: PPDocLayoutV3Config
    base_model_prefix = "pp_doclayout_v3"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = [r"PPDocLayoutV3HybridEncoder", r"PPDocLayoutV3DecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, PPDocLayoutV3MultiscaleDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
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

            init.copy_(module.sampling_offsets.bias, grid_init.view(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            init.xavier_uniform_(module.output_proj.weight)
            init.constant_(module.output_proj.bias, 0.0)

        elif isinstance(module, PPDocLayoutV3Model):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            init.xavier_uniform_(module.enc_score_head.weight)
            init.constant_(module.enc_score_head.bias, bias)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)

        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                init.zeros_(module.weight.data[module.padding_idx])


def bbox_xyxy_to_cxcywh(x):
    x1 = x[..., 0]
    y1 = x[..., 1]
    x2 = x[..., 2]
    y2 = x[..., 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)


def mask_to_box_coordinate(mask, normalize=False, format="xyxy", dtype=torch.float32):
    assert mask.ndim == 4, f"Error: The 'mask' variable must be a 4-dimensional array, but got {mask.ndim} dimensions."
    assert format in ["xyxy", "xywh"], (
        f"Error: The 'format' variable must be either 'xyxy' or 'xywh', but got '{format}'."
    )

    mask = mask.bool()

    h, w = mask.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=mask.device), torch.arange(w, device=mask.device), indexing="ij")
    x = x.to(dtype)
    y = y.to(dtype)

    x_mask = x * mask
    x_max = x_mask.flatten(start_dim=-2).max(dim=-1).values + 1
    x_min = (
        torch.where(mask, x_mask, torch.tensor(1e8, device=mask.device, dtype=dtype))
        .flatten(start_dim=-2)
        .min(dim=-1)
        .values
    )

    y_mask = y * mask
    y_max = y_mask.flatten(start_dim=-2).max(dim=-1).values + 1
    y_min = (
        torch.where(mask, y_mask, torch.tensor(1e8, device=mask.device, dtype=dtype))
        .flatten(start_dim=-2)
        .min(dim=-1)
        .values
    )

    out_bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    mask_any = torch.any(mask, dim=(-2, -1)).unsqueeze(-1)
    out_bbox = out_bbox * mask_any

    if normalize:
        norm_tensor = torch.tensor([w, h, w, h], device=mask.device, dtype=dtype)
        out_bbox /= norm_tensor

    return out_bbox if format == "xyxy" else bbox_xyxy_to_cxcywh(out_bbox)


@dataclass
class PPDocLayoutV3DecoderOutput(RTDetrDecoderOutput):
    r"""
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked initial reference points (initial reference points of each layer of the decoder).
    cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
        used to compute the weighted average in the cross-attention heads.
    dec_out_order_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.num_queries, config.num_queries)`):
        Stacked order logits (order logits of each layer of the decoder).
    dec_out_masks (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.num_queries, 200, 200)`):
        Stacked masks (masks of each layer of the decoder).
    """

    dec_out_order_logits: Optional[torch.FloatTensor] = None
    dec_out_masks: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the PP-DocLayoutV3 model.
    """
)
class PPDocLayoutV3ModelOutput(RTDetrModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Initial reference points used for the first decoder layer.
    init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`):
        Logits of predicted bounding boxes coordinates in the encoder stage.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values.
    out_order_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.num_queries, config.num_queries)`):
        Stacked order logits (order logits of each layer of the decoder).
    out_masks (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.num_queries, 200, 200)`):
        Stacked masks (masks of each layer of the decoder).
    """

    out_order_logits: Optional[torch.FloatTensor] = None
    out_masks: Optional[torch.FloatTensor] = None


class PPDocLayoutV3MLPPredictionHead(RTDetrMLPPredictionHead):
    pass


class BaseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y


class MaskFeatFPN(nn.Module):
    def __init__(
        self,
        in_channels=[256, 256, 256],
        fpn_strides=[32, 16, 8],
        feat_channels=256,
        dropout_ratio=0.0,
        out_channels=256,
        align_corners=False,
    ):
        super().__init__()

        assert len(in_channels) == len(fpn_strides), (
            f"Error: The lengths of 'in_channels' and 'fpn_strides' must be equal. "
            f"Got len(in_channels)={len(in_channels)} and len(fpn_strides)={len(fpn_strides)}."
        )

        reorder_index = np.argsort(fpn_strides, axis=0)
        in_channels = [in_channels[i] for i in reorder_index]
        fpn_strides = [fpn_strides[i] for i in reorder_index]

        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)

        self.scale_heads = nn.ModuleList()
        for i in range(len(fpn_strides)):
            head_length = max(1, int(np.log2(fpn_strides[i]) - np.log2(fpn_strides[0])))
            scale_head = []
            for k in range(head_length):
                in_c = in_channels[i] if k == 0 else feat_channels
                scale_head.append(nn.Sequential(BaseConv(in_c, feat_channels, 3, 1)))
                if fpn_strides[i] != fpn_strides[0]:
                    scale_head.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners))

            self.scale_heads.append(nn.Sequential(*scale_head))

        self.output_conv = BaseConv(feat_channels, out_channels, 3, 1)

    def forward(self, inputs):
        x = [inputs[i] for i in self.reorder_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.fpn_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]), size=output.shape[2:], mode="bilinear", align_corners=self.align_corners
            )

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.output_conv(output)
        return output


class PPDocLayoutV3HybridEncoder(RTDetrHybridEncoder):
    def __init__(self, config: PPDocLayoutV3Config):
        super().__init__()

        feat_strides = config.feat_strides
        mask_feat_channels = config.mask_feat_channels
        self.mask_feat_head = MaskFeatFPN(
            [self.encoder_hidden_dim] * len(feat_strides),
            feat_strides,
            feat_channels=mask_feat_channels[0],
            out_channels=mask_feat_channels[1],
        )
        self.enc_mask_lateral = BaseConv(config.x4_feat_dim, mask_feat_channels[1], 3, 1)
        self.enc_mask_output = nn.Sequential(
            BaseConv(mask_feat_channels[1], mask_feat_channels[1], 3, 1),
            nn.Conv2d(in_channels=mask_feat_channels[1], out_channels=config.num_prototypes, kernel_size=1),
        )

    def forward(
        self,
        inputs_embeds=None,
        x4_feat=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # encoder
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states[enc_ind],)
                height, width = hidden_states[enc_ind].shape[2:]
                # flatten [batch, channel, height, width] to [batch, height*width, channel]
                src_flatten = hidden_states[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.encoder_hidden_dim,
                        self.positional_encoding_temperature,
                        device=src_flatten.device,
                        dtype=src_flatten.dtype,
                    )
                else:
                    pos_embed = None

                layer_outputs = self.encoder[i](
                    src_flatten,
                    pos_embed=pos_embed,
                    output_attentions=output_attentions,
                )
                hidden_states[enc_ind] = (
                    layer_outputs[0].permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, height, width).contiguous()
                )

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states[enc_ind],)

        # top-down FPN
        fpn_feature_maps = [hidden_states[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = hidden_states[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            # apply lateral block
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            # apply fpn block
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        mask_feat = self.mask_feat_head(pan_feature_maps)
        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode="bilinear", align_corners=False)
        mask_feat += self.enc_mask_lateral(x4_feat[0])
        mask_feat = self.enc_mask_output(mask_feat)

        if not return_dict:
            return tuple(v for v in [pan_feature_maps, encoder_states, all_attentions, mask_feat] if v is not None)

        return PPDocLayoutV3HybridEncoderOutput(
            last_hidden_state=pan_feature_maps,
            hidden_states=encoder_states,
            attentions=all_attentions,
            mask_feat=mask_feat,
        )


class PPDocLayoutV3Decoder(RTDetrDecoder):
    def __init__(self, config: PPDocLayoutV3Config):
        super().__init__()

        self.num_queries = config.num_queries

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        order_head=None,
        global_pointer=None,
        mask_query_head=None,
        dec_norm=None,
        mask_feat=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()
        dec_out_order_logits = ()
        dec_out_masks = ()

        reference_points = F.sigmoid(reference_points)

        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L252
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed(hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            # get_pred_class_order_and_mask
            out_query = dec_norm(hidden_states)
            mask_query_embed = mask_query_head(out_query)
            batch_size, mask_dim, _ = mask_query_embed.shape
            _, _, mask_h, mask_w = mask_feat.shape
            out_mask = torch.bmm(mask_query_embed, mask_feat.flatten(start_dim=2)).reshape(
                batch_size, mask_dim, mask_h, mask_w
            )
            dec_out_masks += (out_mask,)

            if self.class_embed is not None:
                logits = self.class_embed(out_query)
                intermediate_logits += (logits,)

            if order_head is not None and global_pointer is not None:
                valid_query = out_query[:, -self.num_queries :] if self.num_queries is not None else out_query
                order_logits = global_pointer(order_head(valid_query))
                dec_out_order_logits += (order_logits,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = torch.stack(intermediate_logits, dim=1)
        if order_head is not None and global_pointer is not None:
            dec_out_order_logits = torch.stack(dec_out_order_logits, dim=1)
        dec_out_masks = torch.stack(dec_out_masks, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    dec_out_order_logits,
                    dec_out_masks,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return PPDocLayoutV3DecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
            dec_out_order_logits=dec_out_order_logits,
            dec_out_masks=dec_out_masks,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class PPDocLayoutV3Model(RTDetrModel):
    def __init__(self, config: PPDocLayoutV3Config):
        super().__init__(config)

        self.encoder_input_proj = nn.ModuleList(encoder_input_proj_list[1:])  # noqa

        self.dec_order_head = nn.Linear(config.d_model, config.d_model)
        self.dec_global_pointer = GlobalPointer(config)
        self.dec_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.decoder = PPDocLayoutV3Decoder(config)
        self.decoder.class_embed = self.enc_score_head
        self.decoder.bbox_embed = self.enc_bbox_head

        self.mask_enhanced = config.mask_enhanced
        self.mask_query_head = PPDocLayoutV3MLPPredictionHead(
            config, config.d_model, config.d_model, config.num_prototypes, num_layers=3
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPDocLayoutV3ModelOutput]:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, PPDocLayoutV2Model
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("PekingU/PPDocLayoutV2_r50vd")
        >>> model = PPDocLayoutV2Model.from_pretrained("PekingU/PPDocLayoutV2_r50vd")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        features = self.backbone(pixel_values, pixel_mask)
        x4_feat = features.pop(0)
        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                proj_feats,
                x4_feat,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a PPDocLayoutV3HybridEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, PPDocLayoutV3HybridEncoderOutput):
            encoder_outputs = PPDocLayoutV3HybridEncoderOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if output_hidden_states else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2
                else encoder_outputs[1]
                if output_attentions
                else None,
                mask_feat=encoder_outputs[-1],
            )

        mask_feat = (
            encoder_outputs.mask_feat
            if isinstance(encoder_outputs, PPDocLayoutV3HybridEncoderOutput)
            else encoder_outputs[-1]
        )

        # Equivalent to def _get_encoder_input
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        sources = []
        for level, source in enumerate(encoder_outputs[0]):
            sources.append(self.decoder_input_proj[level](source))

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs[0])[-1])
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs[0][-1]))

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # prepare denoising training
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(source_flatten)
        device = source_flatten.device
        dtype = source_flatten.dtype

        # prepare input for decoder
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure
            # lru_cache is working for generate_anchors()
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        # _get_pred_class_and_mask
        batch_ind = torch.arange(memory.shape[0], device=output_memory.device).unsqueeze(1)
        target = output_memory[batch_ind, topk_ind]
        out_query = self.dec_norm(target)
        mask_query_embed = self.mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        _, _, mask_h, mask_w = mask_feat.shape
        enc_out_masks = torch.bmm(mask_query_embed, mask_feat.flatten(start_dim=2)).reshape(
            batch_size, mask_dim, mask_h, mask_w
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        if self.mask_enhanced:
            reference_points = mask_to_box_coordinate(enc_out_masks > 0, normalize=True, format="xywh")
            reference_points_unact = inverse_sigmoid(reference_points)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        init_reference_points = reference_points_unact.detach()

        # decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            order_head=self.dec_order_head,
            global_pointer=self.dec_global_pointer,
            mask_query_head=self.mask_query_head,
            dec_norm=self.dec_norm,
            mask_feat=mask_feat,
        )

        if not return_dict:
            enc_outputs = tuple(
                value
                for value in [enc_topk_logits, enc_topk_bboxes, enc_outputs_class, enc_outputs_coord_logits]
                if value is not None
            )
            dn_outputs = tuple(value if value is not None else None for value in [denoising_meta_values])
            tuple_outputs = (
                decoder_outputs + encoder_outputs[:-1] + (init_reference_points,) + enc_outputs + dn_outputs
            )

            return tuple_outputs

        return PPDocLayoutV3ModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_predicted_corners=decoder_outputs.intermediate_predicted_corners,
            initial_reference_points=decoder_outputs.initial_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            out_order_logits=decoder_outputs.dec_out_order_logits,
            out_masks=decoder_outputs.dec_out_masks,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


@dataclass
class PPDocLayoutV3HybridEncoderOutput(BaseModelOutput):
    mask_feat: torch.FloatTensor = None


@dataclass
@auto_docstring
class PPDocLayoutV3ForObjectDetectionOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    order_logits (`tuple` of `torch.FloatTensor` of shape `(batch_size, num_queries, num_queries)`):
        Order logits of the final layer of the decoder.
    out_masks (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, height, width)`):
        Masks of the final layer of the decoder.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~RTDetrImageProcessor.post_process_object_detection`] to retrieve the
        unnormalized (absolute) bounding boxes.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked initial reference points (initial reference points of each layer of the decoder).
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the encoder.
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the encoder.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values
    """

    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    order_logits: Optional[torch.FloatTensor] = None
    out_masks: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_logits: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    intermediate_predicted_corners: Optional[torch.FloatTensor] = None
    initial_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    init_reference_points: Optional[tuple[torch.FloatTensor]] = None
    enc_topk_logits: Optional[torch.FloatTensor] = None
    enc_topk_bboxes: Optional[torch.FloatTensor] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    denoising_meta_values: Optional[dict] = None


class PPDocLayoutV3ForObjectDetection(RTDetrForObjectDetection, PPDocLayoutV3PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked", "rel_pos_y_bias", "rel_pos_x_bias"]
    _tied_weights_keys = {
        "model.decoder.class_embed": "model.enc_score_head",
        "model.decoder.bbox_embed": "model.enc_bbox_head",
    }

    def __init__(self, config: PPDocLayoutV3Config):
        super().__init__(config)

        del self.model.decoder.class_embed
        del self.model.decoder.bbox_embed
        del num_pred  # noqa

        self.model.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)
        self.num_queries = config.num_queries

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPDocLayoutV3ForObjectDetectionOutput]:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoModelForObjectDetection, AutoImageProcessor
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> model_path = "PaddlePaddle/PP-DocLayoutV3_safetensors"
        >>> image_processor = AutoImageProcessor.from_pretrained(model_path)
        >>> model = AutoModelForObjectDetection.from_pretrained(model_path)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=[image], return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))

        >>> # print outputs
        >>> for result in results:
        ...     for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        ...         score, label = score.item(), label_id.item()
        ...         box = [round(i, 2) for i in box.tolist()]
        ...         print(f"Order {idx + 1}: {model.config.id2label[label]}: {score:.2f} {box}")
        Order 1: text: 0.99 [334.95, 184.78, 897.25, 654.83]
        Order 2: paragraph_title: 0.97 [337.28, 683.92, 869.16, 798.35]
        Order 3: text: 0.99 [335.75, 842.82, 892.13, 1454.32]
        Order 4: text: 0.99 [920.18, 185.28, 1476.38, 464.49]
        Order 5: text: 0.98 [920.47, 483.68, 1480.63, 765.72]
        Order 6: text: 0.98 [920.62, 846.8, 1482.09, 1220.67]
        Order 7: text: 0.97 [920.92, 1239.41, 1469.55, 1378.02]
        Order 8: footnote: 0.86 [335.03, 1614.68, 1483.33, 1731.73]
        Order 9: footnote: 0.83 [334.64, 1756.74, 1471.78, 1845.69]
        Order 10: text: 0.81 [336.8, 1910.52, 661.64, 1939.92]
        Order 11: footnote: 0.96 [336.24, 2114.42, 1450.14, 2172.12]
        Order 12: number: 0.88 [106.0, 2257.5, 135.84, 2282.18]
        Order 13: footer: 0.93 [338.4, 2255.52, 986.15, 2284.37]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        intermediate_logits = outputs.intermediate_logits if return_dict else outputs[2]
        intermediate_reference_points = outputs.intermediate_reference_points if return_dict else outputs[3]
        order_logits = outputs.out_order_logits if return_dict else outputs[4]
        out_masks = outputs.out_masks if return_dict else outputs[5]

        pred_boxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]
        order_logits = order_logits[:, -1]
        out_masks = out_masks[:, -1]

        if labels is not None:
            raise ValueError("PPDocLayoutV3ForObjectDetection does not support training")

        if not return_dict:
            return (logits, pred_boxes, order_logits, out_masks) + outputs[:4] + outputs[6:]

        return PPDocLayoutV3ForObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            order_logits=order_logits,
            out_masks=out_masks,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            intermediate_predicted_corners=outputs.intermediate_predicted_corners,
            initial_reference_points=outputs.initial_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )


__all__ = [
    "PPDocLayoutV3ForObjectDetection",
    "PPDocLayoutV3ImageProcessor",
    "PPDocLayoutV3ImageProcessorFast",
    "PPDocLayoutV3Config",
    "PPDocLayoutV3Model",
    "PPDocLayoutV3PreTrainedModel",
]
