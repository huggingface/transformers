# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
)
from ...image_utils import (
    PILImageResampling,
)
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.backbone_utils import verify_backbone_config_arguments
from ...utils.generic import TensorType, check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3Attention,
    LayoutLMv3Encoder,
    LayoutLMv3Intermediate,
    LayoutLMv3Layer,
    LayoutLMv3Output,
    LayoutLMv3SelfAttention,
    LayoutLMv3SelfOutput,
    LayoutLMv3TextEmbeddings,
)
from ..rt_detr.modeling_rt_detr import (
    RTDetrForObjectDetection,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
)


logger = logging.get_logger(__name__)


class ReadingOrderConfig(PreTrainedConfig):
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=8,
        attention_probs_dropout_prob=0.1,
        has_relative_attention_bias=False,
        has_spatial_attention_bias=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        intermediate_size=2048,
        hidden_act="gelu",
        num_hidden_layers=6,
        rel_pos_bins=32,
        max_rel_pos=128,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        num_labels=510,
        max_position_embeddings=514,
        max_2d_position_embeddings=1024,
        type_vocab_size=1,
        vocab_size=4,
        start_token_id=0,
        pad_token_id=1,
        end_token_id=2,
        pred_token_id=3,
        coordinate_size=171,
        shape_size=170,
        num_classes=20,
        rel_bias_embed_dim=16,
        rel_bias_temperature=10000,
        rel_bias_scale=100,
        relative_head_num=1,
        relative_head_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.num_labels = num_labels
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.pred_token_id = pred_token_id
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.rel_bias_embed_dim = rel_bias_embed_dim
        self.rel_bias_temperature = rel_bias_temperature
        self.rel_bias_scale = rel_bias_scale
        self.relative_head_num = relative_head_num
        self.relative_head_size = relative_head_size


class PPDocLayoutV2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PP-DocLayoutV2`]. It is used to instantiate a
    PP-DocLayoutV2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PP-DocLayoutV2
    [PaddlePaddle/PP-DocLayoutV2_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2_safetensors) architecture.

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
        anchor_image_size (`tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Whether to disable custom kernels.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
        threshold_mapping (`dict[str, float]`, *optional*):
            Mapping from class name to class priority.
        order_map (`dict[str, float]`, *optional*):
            Mapping from class name to class threshold.
        reading_order_config (`dict`, *optional*):
            The configuration of a `ReadingOrder`.

    Examples:

    ```python
    >>> from transformers import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection

    >>> # Initializing a PP-DocLayoutV2 configuration
    >>> configuration = PPDocLayoutV2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPDocLayoutV2ForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pp_doclayout_v2"
    sub_configs = {"backbone_config": AutoConfig, "reading_order_config": ReadingOrderConfig}

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
        # encoder HybridEncoder
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
        # decoder PPDocLayoutV2Transformer
        d_model=256,
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
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        disable_custom_kernels=True,
        is_encoder_decoder=True,
        # label
        threshold_mapping=None,
        order_map=None,
        reading_order_config=None,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps

        if isinstance(reading_order_config, dict):
            self.reading_order_config = self.sub_configs["reading_order_config"](**reading_order_config)
        elif reading_order_config is None:
            self.reading_order_config = self.sub_configs["reading_order_config"]()

        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `HGNetV2` backbone."
            )
            backbone_config = {
                "model_type": "hgnet_v2",
                "arch": "L",
                "return_idx": [1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage2", "stage3", "stage4"],
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

        # ---- decoder ----
        self.d_model = d_model
        self.num_queries = num_queries
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
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = list(anchor_image_size) if anchor_image_size is not None else None
        self.disable_custom_kernels = disable_custom_kernels

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

        self.threshold_mapping = threshold_mapping
        self.order_map = order_map


def get_order_seqs(order_logits):
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


class PPDocLayoutV2ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    size = {"height": 800, "width": 800}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_order_seqs(self, order_logits):
        """
        Computes the order sequences for a batch of inputs based on logits.
        This function takes in the order logits, calculates order scores using a sigmoid activation,
        and determines the order sequences by ranking the votes derived from the scores.
        Args:
            order_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_queries)`):
                Stacked order logits.
        Returns:
            torch.Tensor: A tensor of shape `(batch_size, num_queries)`:
                Containing the computed order sequences for each input in the batch. Each row represents the ranked order of elements for the corresponding input in the batch.
        """
        order_scores = torch.sigmoid(order_logits)
        batch_size, sequence_length, _ = order_scores.shape

        order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (1.0 - order_scores.transpose(1, 2)).tril(
            diagonal=-1
        ).sum(dim=1)

        order_pointers = torch.argsort(order_votes, dim=1)
        order_seq = torch.empty_like(order_pointers)
        ranks = torch.arange(sequence_length, device=order_pointers.device, dtype=order_pointers.dtype).expand(
            batch_size, -1
        )
        order_seq.scatter_(1, order_pointers, ranks)

        return order_seq

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: TensorType | list[tuple] | None = None,
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
        boxes = outputs.pred_boxes
        logits = outputs.logits
        order_logits = outputs.order_logits

        order_seqs = self._get_order_seqs(order_logits)

        cxcy, wh = torch.split(boxes, 2, dim=-1)
        boxes = torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
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


class GlobalPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.relative_head_num
        self.head_size = config.relative_head_size
        self.dense = nn.Linear(config.hidden_size, self.heads * 2 * self.head_size)

    def forward(self, inputs, attn_mask_1d):
        batch_size, sequence, _ = inputs.shape
        proj = self.dense(inputs).reshape([batch_size, sequence, self.heads, 2, self.head_size])
        qw, kw = proj[..., 0, :], proj[..., 1, :]

        qw_t = qw.transpose(1, 2)
        kw_t = kw.transpose(1, 2)
        logits = torch.einsum("bhmd,bhnd->bhmn", qw_t, kw_t) / (self.head_size**0.5)

        a = attn_mask_1d.float()
        pair_mask = 1.0 - (a.unsqueeze(1).unsqueeze(2) * a.unsqueeze(1).unsqueeze(3))
        logits = logits - pair_mask * 1e4

        lower = torch.tril(torch.ones([sequence, sequence], dtype=torch.float32, device=logits.device))
        lower = lower.bool().unsqueeze(0).unsqueeze(0)
        logits = logits - lower.to(logits.dtype) * 1e4
        pair_mask = torch.logical_or(pair_mask.bool(), lower)

        return logits, pair_mask.bool()


def box_rel_encoding(src_boxes: torch.Tensor, tgt_boxes: torch.Tensor = None, eps: float = 1e-5):
    if tgt_boxes is None:
        tgt_boxes = src_boxes
    xy1, wh1 = src_boxes[..., :2], src_boxes[..., 2:]
    xy2, wh2 = tgt_boxes[..., :2], tgt_boxes[..., 2:]
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos = torch.cat([delta_xy, delta_wh], dim=-1)
    return pos


def get_sine_pos_embed(inputs: torch.Tensor, num_pos_feats: int, temperature: float = 10000.0, scale: float = 100.0):
    half = num_pos_feats // 2
    dim_t = temperature ** (2 * torch.arange(half, dtype=inputs.dtype, device=inputs.device) / half)

    def _encode(t: torch.Tensor):
        t = t * scale
        t = t.unsqueeze(-1) / dim_t
        sin = torch.sin(t)
        cos = torch.cos(t)
        return torch.cat([sin, cos], dim=-1)

    embs = [_encode(inputs[..., i]) for i in range(inputs.shape[-1])]
    out = torch.cat(embs, dim=-1)
    return out


class PositionRelationEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.rel_bias_embed_dim
        self.temperature = config.rel_bias_temperature
        self.scale = config.rel_bias_scale
        self.pos_proj = nn.Conv2d(
            in_channels=self.embed_dim * 4, out_channels=config.num_attention_heads, kernel_size=1
        )

    def forward(self, src_boxes: torch.Tensor, tgt_boxes: torch.Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        with torch.no_grad():
            rel = box_rel_encoding(src_boxes, tgt_boxes)
            pos = get_sine_pos_embed(rel, num_pos_feats=self.embed_dim, temperature=self.temperature, scale=self.scale)
            pos = pos.permute(0, 3, 1, 2)
        out = self.pos_proj(pos)
        return out


class LayoutLMv3SelfAttentionCustom(LayoutLMv3SelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://huggingface.co/papers/2105.13290)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        # custom
        if rel_2d_pos is not None:
            attention_scores += rel_2d_pos

        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # Use the trick of the CogView paper to stabilize training
        attention_probs = self.cogview_attention(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LayoutLMv3SelfOutputCustom(LayoutLMv3SelfOutput):
    pass


class LayoutLMv3IntermediateCustom(LayoutLMv3Intermediate):
    pass


class LayoutLMv3OutputCustom(LayoutLMv3Output):
    pass


class LayoutLMv3AttentionCustom(LayoutLMv3Attention):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttentionCustom(config)
        self.output = LayoutLMv3SelfOutputCustom(config)


class LayoutLMv3LayerCustom(LayoutLMv3Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = LayoutLMv3AttentionCustom(config)
        self.intermediate = LayoutLMv3IntermediateCustom(config)
        self.output = LayoutLMv3OutputCustom(config)


class LayoutLMv3EncoderCustom(LayoutLMv3Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([LayoutLMv3LayerCustom(config) for _ in range(config.num_hidden_layers)])
        self.rel_bias_module = PositionRelationEmbedding(config)

    def _cal_2d_pos_emb(self, bbox):
        x_min, y_min, x_max, y_max = (
            bbox[..., 0],
            bbox[..., 1],
            bbox[..., 2],
            bbox[..., 3],
        )

        width = (x_max - x_min).clamp(min=1e-3)
        height = (y_max - y_min).clamp(min=1e-3)

        center_x = (x_min + x_max) * 0.5
        center_y = (y_min + y_max) * 0.5

        center_wh_bbox = torch.stack([center_x, center_y, width, height], dim=-1)

        result = self.rel_bias_module(center_wh_bbox)

        return result


class LayoutLMv3TextEmbeddingsCustom(LayoutLMv3TextEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        spatial_embed_dim = 4 * config.coordinate_size + 2 * config.shape_size
        self.spatial_proj = nn.Linear(spatial_embed_dim, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        # custom
        spatial_position_embeddings = self.spatial_proj(spatial_position_embeddings)

        embeddings += spatial_position_embeddings
        return embeddings


@auto_docstring
class PPDocLayoutV2PreTrainedModel(RTDetrPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, LayoutLMv3TextEmbeddingsCustom):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                init.zeros_(module.weight.data[module.padding_idx])


class ReadingOrder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = LayoutLMv3TextEmbeddingsCustom(config)
        self.label_embeddings = nn.Embedding(config.num_classes, config.hidden_size)
        self.label_features_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder = LayoutLMv3EncoderCustom(config)
        self.relative_head = GlobalPointer(config)
        self.config = config

    def forward(self, boxes, labels=None, mask=None):
        device = mask.device
        batch_size, seq_len = mask.shape
        num_pred = mask.sum(dim=1)

        input_ids = torch.full(
            (batch_size, seq_len + 2), self.config.pad_token_id, dtype=torch.long, device=boxes.device
        )
        input_ids[:, 0] = self.config.start_token_id

        pred_col_idx = torch.arange(seq_len + 2, device=device).unsqueeze(0)
        pred_mask = (pred_col_idx >= 1) & (pred_col_idx <= num_pred.unsqueeze(1))
        input_ids[pred_mask] = self.config.pred_token_id
        end_col_indices = num_pred + 1
        input_ids[torch.arange(batch_size, device=device), end_col_indices] = self.config.end_token_id

        pad_box = torch.zeros(size=[boxes.shape[0], 1, boxes.shape[-1]], dtype=boxes.dtype, device=boxes.device)
        pad_boxes = torch.cat([pad_box, boxes, pad_box], dim=1)
        bbox_embedding = self.embeddings(input_ids=input_ids, bbox=pad_boxes.long())

        if labels is not None:
            label_embs = self.label_embeddings(labels)
            label_proj = self.label_features_projection(label_embs)
            pad = torch.zeros(
                size=[label_proj.shape[0], 1, label_proj.shape[-1]], dtype=label_proj.dtype, device=labels.device
            )
            label_proj = torch.cat([pad, label_proj, pad], dim=1)
        else:
            label_proj = torch.zeros_like(bbox_embedding)

        final_embeddings = bbox_embedding + label_proj
        final_embeddings = self.embeddings.LayerNorm(final_embeddings)
        final_embeddings = self.embeddings.dropout(final_embeddings)

        attn_1d = pred_col_idx < (num_pred + 2).unsqueeze(1)
        attention_mask = (1.0 - attn_1d.to(dtype=bbox_embedding.dtype)).unsqueeze(1).unsqueeze(2) * -1e9
        encoder_output = self.encoder(hidden_states=final_embeddings, bbox=pad_boxes, attention_mask=attention_mask)
        encoder_output = encoder_output.last_hidden_state
        tok = encoder_output[:, 1 : 1 + seq_len, :]
        attn_1d = torch.arange(seq_len, device=device)[None, :] < num_pred[:, None]
        logits_bh, _ = self.relative_head(tok, attn_1d)
        read_order_logits = logits_bh[:, 0]
        return read_order_logits


@dataclass
@auto_docstring
class PPDocLayoutV2ForObjectDetectionOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    order_logits (`tuple` of `torch.FloatTensor` of shape `(batch_size, num_queries, num_queries)`):
        Order logits for all queries. The first dimension of each tensor is the batch size. The second dimension is the number of queries.
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

    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    order_logits: tuple[torch.FloatTensor] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_logits: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    intermediate_predicted_corners: torch.FloatTensor | None = None
    initial_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    init_reference_points: tuple[torch.FloatTensor] | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


class PPDocLayoutV2MLPPredictionHead(RTDetrMLPPredictionHead):
    pass


class PPDocLayoutV2Model(RTDetrModel):
    pass


class PPDocLayoutV2ForObjectDetection(RTDetrForObjectDetection, PPDocLayoutV2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked", "rel_pos_y_bias", "rel_pos_x_bias"]

    def __init__(self, config: PPDocLayoutV2Config):
        super().__init__(config)

        self.model.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)
        self.class_thresholds = [config.threshold_mapping[v] for v in config.id2label.values()]
        self.class_map = [config.order_map[category] for category in config.order_map]
        self.reading_order = ReadingOrder(config.reading_order_config)
        self.num_queries = config.num_queries

        self.post_init()

    @auto_docstring
    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | PPDocLayoutV2ForObjectDetectionOutput:
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

        >>> model_path = "PaddlePaddle/PP-DocLayoutV2_safetensors"
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

        intermediate_reference_points = outputs.intermediate_reference_points if return_dict else outputs[3]
        intermediate_logits = outputs.intermediate_logits if return_dict else outputs[2]
        raw_bboxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]

        cxcy, wh = raw_bboxes.split(2, dim=-1)
        bboxes = torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1) * 1000
        bboxes = bboxes.clamp_(0.0, 1000.0)

        max_logits, class_ids = logits.max(dim=-1)
        max_probs = max_logits.sigmoid()

        class_thresholds = torch.tensor(self.class_thresholds, dtype=torch.float32, device=logits.device)
        thresholds = class_thresholds[class_ids]
        mask = max_probs >= thresholds

        indices = torch.argsort(mask.to(torch.int8), dim=1, descending=True)

        sorted_class_ids = torch.take_along_dim(class_ids, indices, dim=1)
        sorted_boxes = torch.take_along_dim(bboxes, indices[..., None].expand(-1, -1, 4), dim=1)
        pred_boxes = torch.take_along_dim(raw_bboxes, indices[..., None].expand(-1, -1, 4), dim=1)
        logits = torch.take_along_dim(logits, indices[..., None].expand(-1, -1, logits.size(-1)), dim=1)

        sorted_mask = torch.take_along_dim(mask, indices, dim=1)

        pad_boxes = torch.where(sorted_mask[..., None], sorted_boxes, torch.zeros_like(sorted_boxes))
        pad_class_ids = torch.where(sorted_mask, sorted_class_ids, torch.zeros_like(sorted_class_ids))

        class_map = torch.tensor(self.class_map, dtype=torch.int32, device=logits.device)
        pad_class_ids = class_map[pad_class_ids]

        order_logits = self.reading_order(
            boxes=pad_boxes,
            labels=pad_class_ids,
            mask=mask,
        )
        order_logits = order_logits[:, :, : self.num_queries]

        if labels is not None:
            raise ValueError("PPDocLayoutV2ForObjectDetection does not support training")

        if not return_dict:
            return (logits, pred_boxes, order_logits) + outputs

        return PPDocLayoutV2ForObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            order_logits=order_logits,
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
    "PPDocLayoutV2ForObjectDetection",
    "PPDocLayoutV2ImageProcessorFast",
    "PPDocLayoutV2Config",
    "PPDocLayoutV2Model",
    "PPDocLayoutV2PreTrainedModel",
]
