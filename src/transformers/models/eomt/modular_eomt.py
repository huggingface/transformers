# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch EoMT model."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    logging,
)
from ...utils.generic import check_model_inputs
from ..dinov2.modeling_dinov2 import (
    Dinov2Embeddings,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2PatchEmbeddings,
)
from ..mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentation, Mask2FormerLoss
from ..siglip.modeling_siglip import SiglipAttention
from ..vit.configuration_vit import ViTConfig


logger = logging.get_logger(__name__)


class EomtConfig(ViTConfig):
    r"""
    This is the configuration class to store the configuration of a [`EomtForUniversalSegmentation`]. It is used to instantiate an EoMT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the EoMT
    [tue-mps/coco_panoptic_eomt_large_640](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each attention layer.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the MLP hidden dimensionality to the hidden size.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 640):
            The size (resolution) of each input image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        layerscale_value (`float`, *optional*, defaults to 1.0):
            Initial value for the LayerScale parameter.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate (drop path) used during training.
        num_upscale_blocks (`int`, *optional*, defaults to 2):
            Number of upsampling blocks used in the decoder or segmentation head.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied after attention projection.
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        num_blocks (`int`, *optional*, defaults to 4):
            Number of feature blocks or stages in the architecture.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Loss weight for the 'no object' class in panoptic/instance segmentation.
        class_weight (`float`, *optional*, defaults to 2.0):
            Loss weight for classification targets.
        mask_weight (`float`, *optional*, defaults to 5.0):
            Loss weight for mask prediction.
        dice_weight (`float`, *optional*, defaults to 5.0):
            Loss weight for the dice loss component.
        train_num_points (`int`, *optional*, defaults to 12544):
            Number of points to sample for mask loss computation during training.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling ratio used in point sampling for mask training.
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points to sample based on importance during training.
        num_queries (`int`, *optional*, defaults to 200):
            Number of object queries in the Transformer.
        num_register_tokens (`int`, *optional*, defaults to 4):
            Number of learnable register tokens added to the transformer input.

    Example:

    ```python
    >>> from transformers import EomtConfig, EomtForUniversalSegmentation

    >>> # Initialize configuration
    >>> config = EomtConfig()

    >>> # Initialize model
    >>> model = EomtForUniversalSegmentation(config)

    >>> # Access config
    >>> config = model.config
    ```"""

    model_type = "eomt"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=640,
        patch_size=16,
        num_channels=3,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        num_upscale_blocks=2,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        num_blocks=4,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        num_queries=200,
        num_register_tokens=4,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            **kwargs,
        )

        del self.intermediate_size
        del self.qkv_bias
        del self.pooler_act
        del self.pooler_output_size
        del self.encoder_stride
        del self.attention_probs_dropout_prob

        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.num_upscale_blocks = num_upscale_blocks
        self.use_swiglu_ffn = use_swiglu_ffn
        self.num_blocks = num_blocks
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.num_queries = num_queries
        self.num_register_tokens = num_register_tokens


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`EomtForUniversalSegmentationOutput`].

    This output can be directly passed to [`~EomtImageProcessor.post_process_semantic_segmentation`] or
    [`~EomtImageProcessor.post_process_instance_segmentation`] or
    [`~EomtImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~EomtImageProcessor] for details regarding usage.
    """
)
class EomtForUniversalSegmentationOutput(ModelOutput):
    r"""
    loss (`torch.Tensor`, *optional*):
        The computed loss, returned when labels are present.
    class_queries_logits (`torch.FloatTensor`):
        A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
        query. Note the `+ 1` is needed because we incorporate the null class.
    masks_queries_logits (`torch.FloatTensor`):
        A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
        query.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Last hidden states (final feature map) of the last layer.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, sequence_length, hidden_size)`. Hidden-states all layers of the model.
    attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    patch_offsets (`list[torch.Tensor]`, *optional*):
        list of tuples indicating the image index and start and end positions of patches for semantic segementation.
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: Optional[torch.FloatTensor] = None
    masks_queries_logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    patch_offsets: Optional[list[torch.Tensor]] = None


class EomtLoss(Mask2FormerLoss):
    pass


class EomtPatchEmbeddings(Dinov2PatchEmbeddings):
    pass


class EomtEmbeddings(Dinov2Embeddings):
    def __init__(self, config: EomtConfig) -> None:
        nn.Module.__init__(self)

        self.config = config
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        self.patch_embeddings = EomtPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_prefix_tokens = 1 + config.num_register_tokens  # 1 for [CLS]
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self):
        raise AttributeError("Not needed for Eomt Model")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


class EomtAttention(SiglipAttention):
    pass


class EomtLayerScale(Dinov2LayerScale):
    pass


class EomtLayer(Dinov2Layer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output, _ = self.attention(hidden_states_norm, head_mask)
        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + hidden_states

        # in Eomt, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class EomtLayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = F.layer_norm(hidden_state, self.normalized_shape, self.weight, self.bias, self.eps)
        hidden_state = hidden_state.permute(0, 3, 1, 2)
        return hidden_state


class EomtScaleLayer(nn.Module):
    def __init__(self, config: EomtConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.conv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.activation = ACT2FN[config.hidden_act]
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )

        self.layernorm2d = EomtLayerNorm2d(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
        return hidden_states


class EomtScaleBlock(nn.Module):
    def __init__(self, config: EomtConfig):
        super().__init__()
        self.num_blocks = config.num_upscale_blocks
        self.block = nn.ModuleList([EomtScaleLayer(config) for _ in range(self.num_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class EomtMaskHead(nn.Module):
    def __init__(self, config: EomtConfig):
        super().__init__()

        hidden_size = config.hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.fc3(hidden_states)
        return hidden_states


@auto_docstring
class EomtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config: EomtConfig
    base_model_prefix = "eomt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = ["EomtLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": EomtLayer,
        "attentions": EomtAttention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, EomtLayerScale):
            if hasattr(module, "lambda1"):
                module.lambda1.data.fill_(self.config.layerscale_value)
        elif isinstance(module, EomtEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=std
            ).to(module.cls_token.dtype)
            module.register_tokens.data.zero_()


@auto_docstring(
    custom_intro="""
    The EoMT Model with head on top for instance/semantic/panoptic segmentation.
    """
)
class EomtForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    def __init__(self, config: EomtConfig):
        PreTrainedModel.__init__(self, config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = EomtEmbeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([EomtLayer(config) for _ in range(config.num_hidden_layers)])

        self.upscale_block = EomtScaleBlock(config)
        self.mask_head = EomtMaskHead(config)

        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        self.grid_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = EomtLoss(config=config, weight_dict=self.weight_dict)

        self.register_buffer("attn_mask_probs", torch.ones(config.num_blocks))

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def get_auxiliary_logits(self):
        raise AttributeError("Note needed for Eomt Model.")

    def predict(self, logits: torch.Tensor):
        query_tokens = logits[:, : self.config.num_queries, :]
        class_logits = self.class_predictor(query_tokens)

        prefix_tokens = logits[:, self.config.num_queries + self.embeddings.num_prefix_tokens :, :]
        prefix_tokens = prefix_tokens.transpose(1, 2)

        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *self.grid_size)

        query_tokens = self.mask_head(query_tokens)
        prefix_tokens = self.upscale_block(prefix_tokens)

        mask_logits = torch.einsum("bqc, bchw -> bqhw", query_tokens, prefix_tokens)

        return mask_logits, class_logits

    @staticmethod
    def _disable_attention_mask(attn_mask, prob, num_query_tokens, encoder_start_tokens, device):
        if prob < 1:
            # Generate random queries to disable based on the probs
            random_queries = torch.rand(attn_mask.shape[0], num_query_tokens, device=device) > prob

            # Disable attention to the query tokens, considering the prefix tokens
            attn_mask[:, :num_query_tokens, encoder_start_tokens:][random_queries] = 1

        return attn_mask

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[list[Tensor]] = None,
        class_labels: Optional[list[Tensor]] = None,
        patch_offsets: Optional[list[Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EomtForUniversalSegmentationOutput:
        r"""
        mask_labels (`list[torch.Tensor]`, *optional*):
            list of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`list[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
        patch_offsets (`list[torch.Tensor]`, *optional*):
            list of tuples indicating the image index and start and end positions of patches for semantic segementation.
        """

        masks_queries_logits_per_layer, class_queries_logits_per_layer = (), ()
        attention_mask = None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        for idx, layer_module in enumerate(self.layers):
            if idx == self.num_hidden_layers - self.config.num_blocks:
                query = self.query.weight[None, :, :].expand(hidden_states.shape[0], -1, -1).to(hidden_states.device)
                hidden_states = torch.cat((query, hidden_states), dim=1)

            if idx >= self.num_hidden_layers - self.config.num_blocks and (
                self.training or self.attn_mask_probs[idx - self.num_hidden_layers + self.config.num_blocks] > 0
            ):
                norm_hidden_states = self.layernorm(hidden_states)
                masks_queries_logits, class_queries_logits = self.predict(norm_hidden_states)

                masks_queries_logits_per_layer += (masks_queries_logits,)
                class_queries_logits_per_layer += (class_queries_logits,)

                attention_mask = torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=torch.bool,
                )

                interpolated_logits = F.interpolate(masks_queries_logits, size=self.grid_size, mode="bilinear")
                interpolated_logits = interpolated_logits.view(
                    interpolated_logits.size(0), interpolated_logits.size(1), -1
                )

                num_query_tokens = self.config.num_queries
                encoder_start_tokens = num_query_tokens + self.embeddings.num_prefix_tokens

                # Set attention mask for queries to focus on encoder tokens based on interpolated logits
                attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0

                # Disable attention mask for random query tokens.
                attention_mask = self._disable_attention_mask(
                    attention_mask,
                    prob=self.attn_mask_probs[idx - self.num_hidden_layers + self.config.num_blocks],
                    num_query_tokens=num_query_tokens,
                    encoder_start_tokens=encoder_start_tokens,
                    device=attention_mask.device,
                )

                # Expand attention mask to 4d mask.
                attention_mask = attention_mask[:, None, ...].expand(-1, self.config.num_attention_heads, -1, -1)
                attention_mask = attention_mask.float().masked_fill(~attention_mask, -1e9)

            hidden_states = layer_module(hidden_states, attention_mask)

        sequence_output = self.layernorm(hidden_states)

        masks_queries_logits, class_queries_logits = self.predict(sequence_output)
        masks_queries_logits_per_layer += (masks_queries_logits,)
        class_queries_logits_per_layer += (class_queries_logits,)

        loss = None
        if mask_labels is not None and class_labels is not None:
            loss = 0.0
            for masks_queries_logits, class_queries_logits in zip(
                masks_queries_logits_per_layer, class_queries_logits_per_layer
            ):
                loss_dict = self.get_loss_dict(
                    masks_queries_logits=masks_queries_logits,
                    class_queries_logits=class_queries_logits,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=None,
                )
                loss += self.get_loss(loss_dict)

        return EomtForUniversalSegmentationOutput(
            loss=loss,
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            last_hidden_state=sequence_output,
            patch_offsets=patch_offsets,
        )


__all__ = ["EomtConfig", "EomtPreTrainedModel", "EomtForUniversalSegmentation"]
