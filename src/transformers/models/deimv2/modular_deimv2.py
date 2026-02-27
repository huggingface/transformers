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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from ..d_fine.configuration_d_fine import DFineConfig
from ..d_fine.modeling_d_fine import (
    DFineConvNormLayer,
    DFineDecoder,
    DFineDecoderLayer,
    DFineDecoderOutput,
    DFineEncoderLayer,
    DFineForObjectDetection,
    DFineGate,
    DFineHybridEncoder,
    DFineIntegral,
    DFineLQE,
    DFineMLP,
    DFineModel,
    DFineMultiscaleDeformableAttention,
    DFinePreTrainedModel,
    DFineRepVggBlock,
    DFineSCDown,
)
from ..rt_detr.modeling_rt_detr import RTDetrAIFILayer


logger = logging.get_logger(__name__)


class Deimv2Config(DFineConfig):
    """
    This is the configuration class to store the configuration of a [`Deimv2Model`]. It is used to instantiate a
    DEIMv2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of DEIMv2-L-COCO.

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
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `HGNetV2Config()`):
            The configuration of the backbone model.
        freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
            Whether to freeze the batch normalization layers in the backbone.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Dimension of the layers in hybrid encoder.
        encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
            Multi level features input for encoder.
        feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
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
            The non-linear activation function (function or string) in the encoder and pooler.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the general layer.
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
            Multi level features dimension for decoder.
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
            The non-linear activation function (function or string) in the decoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training.
        anchor_image_size (`tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors.
        with_box_refine (`bool`, *optional*, defaults to `True`):
            Whether to apply iterative bounding box refinement.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
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
            Parameter informing if focal loss should be used.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        focal_loss_alpha (`float`, *optional*, defaults to 0.75):
            Parameter alpha used to compute the focal loss.
        focal_loss_gamma (`float`, *optional*, defaults to 2.0):
            Parameter gamma used to compute the focal loss.
        weight_loss_vfl (`float`, *optional*, defaults to 1.0):
            Relative weight of the varifocal loss in the object detection loss.
        weight_loss_mal (`float`, *optional*, defaults to 1.0):
            Relative weight of the matching auxiliary loss in the object detection loss.
        weight_loss_bbox (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        weight_loss_giou (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        weight_loss_fgl (`float`, *optional*, defaults to 0.15):
            Relative weight of the fine-grained localization loss in the object detection loss.
        weight_loss_ddf (`float`, *optional*, defaults to 1.5):
            Relative weight of the decoupled distillation focal loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.0001):
            Relative classification weight of the 'no-object' class in the object detection loss.
        eval_idx (`int`, *optional*, defaults to -1):
            Index of the decoder layer to use for evaluation.
        layer_scale (`float`, *optional*, defaults to `1.0`):
            Scaling factor for the hidden dimension in later decoder layers.
        max_num_bins (`int`, *optional*, defaults to 32):
            Maximum number of bins for the distribution-guided bounding box refinement.
        reg_scale (`float`, *optional*, defaults to 4.0):
            Scale factor for the regression distribution.
        depth_mult (`float`, *optional*, defaults to 1.0):
            Multiplier for the number of blocks in RepNCSPELAN5 layers.
        top_prob_values (`int`, *optional*, defaults to 4):
            Number of top probability values to consider from each corner's distribution.
        lqe_hidden_dim (`int`, *optional*, defaults to 64):
            Hidden dimension size for the Location Quality Estimator (LQE) network.
        lqe_layers (`int`, *optional*, defaults to 2):
            Number of layers in the Location Quality Estimator MLP.
        decoder_offset_scale (`float`, *optional*, defaults to 0.5):
            Offset scale used in deformable attention.
        decoder_method (`str`, *optional*, defaults to `"default"`):
            The method to use for the decoder: `"default"` or `"discrete"`.
        up (`float`, *optional*, defaults to 0.5):
            Controls the upper bounds of the Weighting Function.
        use_dense_o2o (`bool`, *optional*, defaults to `True`):
            Whether to use dense one-to-one matching across decoder layers.
        mal_alpha (`float`, *optional*):
            Alpha parameter for the Matching Auxiliary Loss (MAL). If `None`, uses `focal_loss_alpha`.
        encoder_fuse_op (`str`, *optional*, defaults to `"sum"`):
            Fusion operation used in the encoder FPN. DEIMv2 uses `"sum"` instead of D-Fine's `"cat"`.
        use_spatial_tuning_adapter (`bool`, *optional*, defaults to `False`):
            Whether to use the Spatial Tuning Adapter (STA) for DINOv2 backbone variants.
        sta_inplanes (`int`, *optional*, defaults to 16):
            Number of input planes for the STA convolutional stem.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings.
    """

    model_type = "deimv2"

    def __init__(
        self,
        initializer_range=0.01,
        initializer_bias_prior_prob=None,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        backbone_config=None,
        freeze_backbone_batch_norms=True,
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
        with_box_refine=True,
        is_encoder_decoder=True,
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        auxiliary_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        weight_loss_fgl=0.15,
        weight_loss_ddf=1.5,
        eos_coefficient=1e-4,
        eval_idx=-1,
        layer_scale=1,
        max_num_bins=32,
        reg_scale=4.0,
        depth_mult=1.0,
        top_prob_values=4,
        lqe_hidden_dim=64,
        lqe_layers=2,
        decoder_offset_scale=0.5,
        decoder_method="default",
        up=0.5,
        weight_loss_mal=1.0,
        use_dense_o2o=True,
        mal_alpha=None,
        encoder_fuse_op="sum",
        use_spatial_tuning_adapter=False,
        sta_inplanes=16,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.weight_loss_mal = weight_loss_mal
        self.use_dense_o2o = use_dense_o2o
        self.mal_alpha = mal_alpha
        self.encoder_fuse_op = encoder_fuse_op
        self.use_spatial_tuning_adapter = use_spatial_tuning_adapter
        self.sta_inplanes = sta_inplanes
        super().__init__(
            initializer_range=initializer_range,
            initializer_bias_prior_prob=initializer_bias_prior_prob,
            layer_norm_eps=layer_norm_eps,
            batch_norm_eps=batch_norm_eps,
            backbone_config=backbone_config,
            freeze_backbone_batch_norms=freeze_backbone_batch_norms,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_in_channels=encoder_in_channels,
            feat_strides=feat_strides,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            dropout=dropout,
            activation_dropout=activation_dropout,
            encode_proj_layers=encode_proj_layers,
            positional_encoding_temperature=positional_encoding_temperature,
            encoder_activation_function=encoder_activation_function,
            activation_function=activation_function,
            eval_size=eval_size,
            normalize_before=normalize_before,
            hidden_expansion=hidden_expansion,
            d_model=d_model,
            num_queries=num_queries,
            decoder_in_channels=decoder_in_channels,
            decoder_ffn_dim=decoder_ffn_dim,
            num_feature_levels=num_feature_levels,
            decoder_n_points=decoder_n_points,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_activation_function=decoder_activation_function,
            attention_dropout=attention_dropout,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_initial_query=learn_initial_query,
            anchor_image_size=anchor_image_size,
            with_box_refine=with_box_refine,
            is_encoder_decoder=is_encoder_decoder,
            matcher_alpha=matcher_alpha,
            matcher_gamma=matcher_gamma,
            matcher_class_cost=matcher_class_cost,
            matcher_bbox_cost=matcher_bbox_cost,
            matcher_giou_cost=matcher_giou_cost,
            use_focal_loss=use_focal_loss,
            auxiliary_loss=auxiliary_loss,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            weight_loss_vfl=weight_loss_vfl,
            weight_loss_bbox=weight_loss_bbox,
            weight_loss_giou=weight_loss_giou,
            weight_loss_fgl=weight_loss_fgl,
            weight_loss_ddf=weight_loss_ddf,
            eos_coefficient=eos_coefficient,
            eval_idx=eval_idx,
            layer_scale=layer_scale,
            max_num_bins=max_num_bins,
            reg_scale=reg_scale,
            depth_mult=depth_mult,
            top_prob_values=top_prob_values,
            lqe_hidden_dim=lqe_hidden_dim,
            lqe_layers=lqe_layers,
            decoder_offset_scale=decoder_offset_scale,
            decoder_method=decoder_method,
            up=up,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Deimv2DecoderOutput(DFineDecoderOutput):
    pass


class Deimv2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return (hidden_states * self.scale).to(input_dtype)


class Deimv2SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(hidden_states)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Deimv2Gate(DFineGate):
    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.norm = Deimv2RMSNorm(d_model)


class Deimv2MLP(DFineMLP):
    pass


class Deimv2MultiscaleDeformableAttention(DFineMultiscaleDeformableAttention):
    pass


class Deimv2ConvNormLayer(DFineConvNormLayer):
    pass


class Deimv2RepVggBlock(DFineRepVggBlock):
    pass


class Deimv2CSPRepLayer2(nn.Module):
    def __init__(
        self, config: Deimv2Config, in_channels: int, out_channels: int, num_blocks: int, expansion: float = 1.0
    ):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Deimv2ConvNormLayer(config, in_channels, hidden_channels * 2, 1, 1, activation=activation)
        self.bottlenecks = nn.ModuleList(
            [Deimv2RepVggBlock(config, hidden_channels, hidden_channels) for _ in range(num_blocks)]
        )
        self.conv3 = Deimv2ConvNormLayer(config, hidden_channels, out_channels, 3, 1, activation=activation)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        chunks = list(self.conv1(hidden_state).chunk(2, 1))
        bottleneck_out = chunks[1]
        for bottleneck in self.bottlenecks:
            bottleneck_out = bottleneck(bottleneck_out)
        return self.conv3(chunks[0] + bottleneck_out)


class Deimv2RepNCSPELAN5(nn.Module):
    def __init__(self, config: Deimv2Config, numb_blocks: int = 3):
        super().__init__()
        act = config.activation_function
        c1 = config.encoder_hidden_dim
        c2 = config.encoder_hidden_dim
        c3 = config.encoder_hidden_dim * 2
        c4 = round(config.hidden_expansion * config.encoder_hidden_dim // 2)
        self.conv_dim = c3 // 2
        self.conv1 = Deimv2ConvNormLayer(config, c1, c3, 1, 1, activation=act)
        self.csp_rep1 = Deimv2CSPRepLayer2(config, c3 // 2, c4, num_blocks=numb_blocks)
        self.csp_rep2 = Deimv2CSPRepLayer2(config, c4, c4, num_blocks=numb_blocks)
        self.conv4 = Deimv2ConvNormLayer(config, c3 + (2 * c4), c2, 1, 1, activation=act)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        split_features = list(self.conv1(input_features).split((self.conv_dim, self.conv_dim), 1))
        branch1 = self.csp_rep1(split_features[-1])
        branch2 = self.csp_rep2(branch1)
        split_features.extend([branch1, branch2])
        merged_features = torch.cat(split_features, 1)
        return self.conv4(merged_features)


class Deimv2SCDown(DFineSCDown):
    pass


class Deimv2EncoderLayer(DFineEncoderLayer):
    pass


class Deimv2AIFILayer(RTDetrAIFILayer):
    pass


class Deimv2SpatialTuningAdapter(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()
        inplanes = config.sta_inplanes
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(pixel_values)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4


class Deimv2Integral(DFineIntegral):
    pass


class Deimv2LQE(DFineLQE):
    pass


class Deimv2DecoderLayer(DFineDecoderLayer):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.encoder_attn = Deimv2MultiscaleDeformableAttention(config=config)
        self.gateway = Deimv2Gate(config.d_model)
        self.self_attn_layer_norm = Deimv2RMSNorm(config.d_model)
        self.final_layer_norm = Deimv2RMSNorm(config.d_model)
        self.mlp = Deimv2SwiGLUFFN(config.d_model, config.decoder_ffn_dim // 2, config.d_model)


class Deimv2MLPPredictionHead(DFineMLP):
    pass


class Deimv2PreTrainedModel(DFinePreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (Deimv2ForObjectDetection, Deimv2Decoder)):
            if module.class_embed is not None:
                for layer in module.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = float(-math.log((1 - prior_prob) / prior_prob))
                    init.xavier_uniform_(layer.weight)
                    init.constant_(layer.bias, bias)

            if module.bbox_embed is not None:
                for layer in module.bbox_embed:
                    init.constant_(layer.layers[-1].weight, 0)
                    init.constant_(layer.layers[-1].bias, 0)

            if hasattr(module, "reg_scale"):
                init.constant_(module.reg_scale, self.config.reg_scale)

            if hasattr(module, "up"):
                init.constant_(module.up, self.config.up)

        if isinstance(module, Deimv2MultiscaleDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
            grid_init = grid_init.reshape(module.n_heads, 1, 2).tile([1, sum(module.num_points_list), 1])
            scaling = torch.concat([torch.arange(1, n + 1) for n in module.num_points_list]).reshape(1, -1, 1)
            grid_init *= scaling
            init.copy_(module.sampling_offsets.bias, grid_init.flatten())

            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)

            num_points_scale = [1 / n for n in module.num_points_list for _ in range(n)]
            init.copy_(module.num_points_scale, torch.tensor(num_points_scale, dtype=torch.float32))

        if isinstance(module, Deimv2Model):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            init.xavier_uniform_(module.enc_score_head.weight)
            init.constant_(module.enc_score_head.bias, bias)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)

        if isinstance(module, Deimv2Gate):
            bias = float(-math.log((1 - 0.5) / 0.5))
            init.constant_(module.gate.bias, bias)
            init.constant_(module.gate.weight, 0)

        if isinstance(module, Deimv2LQE):
            init.constant_(module.reg_conf.layers[-1].bias, 0)
            init.constant_(module.reg_conf.layers[-1].weight, 0)

        if isinstance(module, Deimv2SwiGLUFFN):
            init.xavier_uniform_(module.w12.weight)
            init.constant_(module.w12.bias, 0)
            init.xavier_uniform_(module.w3.weight)
            init.constant_(module.w3.bias, 0)

        if isinstance(module, Deimv2RMSNorm):
            init.ones_(module.scale)

        if isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

        if hasattr(module, "weight_embedding") and self.config.learn_initial_query:
            init.xavier_uniform_(module.weight_embedding.weight)
        if hasattr(module, "denoising_class_embed") and self.config.num_denoising > 0:
            init.xavier_uniform_(module.denoising_class_embed.weight)


class Deimv2HybridEncoder(DFineHybridEncoder):
    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(config)
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.num_fpn_stages = len(self.in_channels) - 1
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.encoder_fuse_op = config.encoder_fuse_op

        self.aifi = nn.ModuleList([Deimv2AIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            lateral_layer = Deimv2ConvNormLayer(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1)
            self.lateral_convs.append(lateral_layer)
            num_blocks = round(3 * config.depth_mult)
            fpn_layer = Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks)
            self.fpn_blocks.append(fpn_layer)

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(Deimv2SCDown(config, 3, 2))
            num_blocks = round(3 * config.depth_mult)
            self.pan_blocks.append(Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks))

        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        **kwargs,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
        """
        feature_maps = inputs_embeds

        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind], **kwargs)

        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            if self.encoder_fuse_op == "sum":
                fused_feature_map = top_fpn_feature_map + backbone_feature_map
            else:
                fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            if self.encoder_fuse_op == "sum":
                fused_feature_map = downsampled_feature_map + fpn_feature_map
            else:
                fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return BaseModelOutput(last_hidden_state=pan_feature_maps)


class Deimv2Decoder(DFineDecoder):
    def __init__(self, config: Deimv2Config):
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        super().__init__(config=config)
        self.query_pos_head = Deimv2MLP(4, config.d_model, config.d_model, 3, config.decoder_activation_function)
        self.reg_scale = nn.Parameter(torch.tensor([config.reg_scale]), requires_grad=False)
        self.max_num_bins = config.max_num_bins
        self.d_model = config.d_model
        self.layer_scale = config.layer_scale
        self.pre_bbox_head = Deimv2MLP(config.hidden_size, config.hidden_size, 4, 3)
        self.integral = Deimv2Integral(config)
        self.num_head = config.decoder_attention_heads
        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)
        self.lqe_layers = nn.ModuleList([Deimv2LQE(config) for _ in range(config.decoder_layers)])
        self.layers = nn.ModuleList(
            [Deimv2DecoderLayer(config) for _ in range(config.decoder_layers)]
            + [Deimv2DecoderLayer(config) for _ in range(config.decoder_layers - self.eval_idx - 1)]
        )


class Deimv2Model(DFineModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        del self.decoder_input_proj
        self.encoder = Deimv2HybridEncoder(config=config)
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj = []
        in_channels = config.decoder_in_channels[-1]
        for _ in range(num_backbone_outs):
            if config.hidden_size == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                conv = nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False)
                batchnorm = nn.BatchNorm2d(config.d_model, config.batch_norm_eps)
                decoder_input_proj.append(nn.Sequential(conv, batchnorm))
        for _ in range(config.num_feature_levels - num_backbone_outs):
            if config.hidden_size == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                conv = nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False)
                batchnorm = nn.BatchNorm2d(config.d_model, config.batch_norm_eps)
                decoder_input_proj.append(nn.Sequential(conv, batchnorm))
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj)
        self.decoder = Deimv2Decoder(config)

        if config.use_spatial_tuning_adapter:
            self.spatial_tuning_adapter = Deimv2SpatialTuningAdapter(config)


class Deimv2ForObjectDetection(DFineForObjectDetection):
    _no_split_modules = None
    _tied_weights_keys = {
        r"bbox_embed.(?![0])\d+": r"bbox_embed.0",
        r"class_embed.(?![0])\d+": r"^class_embed.0",
        "class_embed": "model.decoder.class_embed",
        "bbox_embed": "model.decoder.bbox_embed",
    }

    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(self, config)

        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        self.model = Deimv2Model(config)
        scaled_dim = round(config.layer_scale * config.hidden_size)
        num_pred = config.decoder_layers
        self.class_embed = nn.ModuleList([nn.Linear(config.d_model, config.num_labels) for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList(
            [
                Deimv2MLP(config.hidden_size, config.hidden_size, 4 * (config.max_num_bins + 1), 3)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                Deimv2MLP(scaled_dim, scaled_dim, 4 * (config.max_num_bins + 1), 3)
                for _ in range(config.decoder_layers - self.eval_idx - 1)
            ]
        )

        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed
        self.post_init()

    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers.image_utils import load_image
        >>> from transformers import AutoImageProcessor, Deimv2ForObjectDetection

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intellindust/DEIMv2_HGNetv2_N_COCO")
        >>> model = Deimv2ForObjectDetection.from_pretrained("Intellindust/DEIMv2_HGNetv2_N_COCO")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
        >>> result = results[0]  # first image in batch

        >>> for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        ```
        """
        super().forward(**super_kwargs)


__all__ = [
    "Deimv2Config",
    "Deimv2Model",
    "Deimv2PreTrainedModel",
    "Deimv2ForObjectDetection",
]
