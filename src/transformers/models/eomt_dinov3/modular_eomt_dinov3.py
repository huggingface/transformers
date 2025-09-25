# coding=utf-8
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
"""PyTorch EoMT model backed by DINOv3."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTAttention,
    DINOv3ViTDropPath,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayer,
    DINOv3ViTLayerScale,
    DINOv3ViTMLP,
    DINOv3ViTRopePositionEmbedding,
)
from ..eomt.configuration_eomt import EomtConfig
from ..eomt.modeling_eomt import (
    EomtForUniversalSegmentationOutput,
    EomtHungarianMatcher,
    EomtLayerNorm2d,
    EomtLoss,
    EomtMaskHead,
    EomtScaleBlock,
    EomtScaleLayer,
)
from ..mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentation


class EomtDinov3Config(EomtConfig):
    r"""Configuration for EoMT models powered by a DINOv3 ViT backbone.

    This configuration is used by
    [tue-mps/coco_panoptic_eomt_dinov3_large_640](https://huggingface.co/tue-mps/coco_panoptic_eomt_dinov3_large_640).
    """

    model_type = "eomt_dinov3"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size: Optional[int] = None,
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
        rope_theta: float = 100.0,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        use_gated_mlp: bool = False,
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = 2.0,
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
        del self.mlp_ratio
        del self.use_swiglu_ffn

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.num_upscale_blocks = num_upscale_blocks
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

        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.proj_bias = proj_bias
        self.mlp_bias = mlp_bias
        self.use_gated_mlp = use_gated_mlp
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale


class EomtDinov3ForUniversalSegmentationOutput(EomtForUniversalSegmentationOutput):
    pass


class EomtDinov3HungarianMatcher(EomtHungarianMatcher):
    pass


class EomtDinov3Loss(EomtLoss):
    pass


class EomtDinov3Embeddings(nn.Module):
    def __init__(self, config: EomtDinov3Config) -> None:
        super().__init__()

        self.config = config
        self.patch_size = config.patch_size
        self.num_prefix_tokens = 1 + config.num_register_tokens

        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        embeddings = embeddings.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)
        embeddings = self.dropout(embeddings)

        return embeddings


class EomtDinov3Attention(DINOv3ViTAttention):
    pass


class EomtDinov3LayerScale(DINOv3ViTLayerScale):
    pass


class EomtDinov3DropPath(DINOv3ViTDropPath):
    pass


class EomtDinov3MLP(DINOv3ViTMLP):
    pass


class EomtDinov3SwiGLUFFN(DINOv3ViTGatedMLP):
    pass


class EomtDinov3Layer(DINOv3ViTLayer):
    pass


class EomtDinov3RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    pass


class EomtDinov3LayerNorm2d(EomtLayerNorm2d):
    pass


class EomtDinov3ScaleLayer(EomtScaleLayer):
    pass


class EomtDinov3ScaleBlock(EomtScaleBlock):
    pass


class EomtDinov3MaskHead(EomtMaskHead):
    pass


class EomtDinov3PreTrainedModel(PreTrainedModel):
    config_class = EomtDinov3Config
    base_model_prefix = "eomt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = ["EomtDinov3Layer"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": EomtDinov3Layer,
        "attentions": EomtDinov3Attention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=std).to(
                module.weight.dtype
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, EomtDinov3LayerScale):
            if hasattr(module, "lambda1"):
                module.lambda1.data.fill_(self.config.layerscale_value)
        elif isinstance(module, EomtDinov3Embeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=std
            ).to(module.cls_token.dtype)
            module.register_tokens.data.zero_()


@auto_docstring(
    custom_intro="""
    The EoMT-DINOv3 model with head on top for instance/semantic/panoptic segmentation.
    """,
)
class EomtDinov3ForUniversalSegmentation(EomtDinov3PreTrainedModel, Mask2FormerForUniversalSegmentation):
    def __init__(self, config: EomtDinov3Config):
        PreTrainedModel.__init__(self, config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers

        self.embeddings = EomtDinov3Embeddings(config)
        self.rope_embeddings = EomtDinov3RopePositionEmbedding(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([EomtDinov3Layer(config) for _ in range(config.num_hidden_layers)])

        self.upscale_block = EomtDinov3ScaleBlock(config)
        self.mask_head = EomtDinov3MaskHead(config)

        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        self.grid_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = EomtDinov3Loss(config=config, weight_dict=self.weight_dict)

        self.register_buffer("attn_mask_probs", torch.ones(config.num_blocks))

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def get_auxiliary_logits(self):
        raise AttributeError("Note needed for EomtDinov3 model.")

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
            random_queries = torch.rand(attn_mask.shape[0], num_query_tokens, device=device) > prob
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
    ) -> EomtDinov3ForUniversalSegmentationOutput:
        r"""
        mask_labels (`list[torch.Tensor]`, *optional*):
            list of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`list[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
        patch_offsets (`list[torch.Tensor]`, *optional*):
            list of tuples indicating the image index and start and end positions of patches for semantic segmentation.
        """

        masks_queries_logits_per_layer, class_queries_logits_per_layer = (), ()
        attention_mask = None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        position_embeddings = self.rope_embeddings(pixel_values)
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

                attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0
                attention_mask = self._disable_attention_mask(
                    attention_mask,
                    prob=self.attn_mask_probs[idx - self.num_hidden_layers + self.config.num_blocks],
                    num_query_tokens=num_query_tokens,
                    encoder_start_tokens=encoder_start_tokens,
                    device=attention_mask.device,
                )
                attention_mask = attention_mask[:, None, ...].expand(-1, self.config.num_attention_heads, -1, -1)
                attention_mask = attention_mask.float().masked_fill(~attention_mask, -1e9)

            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

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

        return EomtDinov3ForUniversalSegmentationOutput(
            loss=loss,
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            last_hidden_state=sequence_output,
            patch_offsets=patch_offsets,
        )


__all__ = [
    "EomtDinov3Config",
    "EomtDinov3PreTrainedModel",
    "EomtDinov3ForUniversalSegmentation",
]
