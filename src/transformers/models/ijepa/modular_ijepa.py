from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.models.ijepa.configuration_ijepa import IJepaConfig

from ...modeling_utils import PreTrainedModel
from ...utils import (
    torch_int,
)
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEmbeddings,
    ViTEncoder,
    ViTForImageClassification,
    ViTIntermediate,
    ViTLayer,
    ViTModel,
    ViTPatchEmbeddings,
    ViTPooler,
    ViTSdpaAttention,
    ViTSdpaSelfAttention,
    ViTSelfAttention,
    ViTSelfOutput,
)


_CHECKPOINT_FOR_DOC = "facebook/ijepa_vith14_1k"


class IJepaEmbeddings(ViTEmbeddings):
    def __init__(self, config: IJepaConfig, use_mask_token: bool = False) -> None:
        super().__init__(config, use_mask_token)
        # Remove cls_token from IJepaEmbeddings, as it is not used in the model
        del self.cls_token
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, config.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        patch_pos_embed = self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class IJepaPatchEmbeddings(ViTPatchEmbeddings):
    pass


class IJepaSelfAttention(ViTSelfAttention):
    pass


class IJepaSdpaSelfAttention(ViTSdpaSelfAttention):
    pass


class IJepaSelfOutput(ViTSelfOutput):
    pass


class IJepaAttention(ViTAttention):
    pass


class IJepaSdpaAttention(ViTSdpaAttention):
    pass


class IJepaIntermediate(ViTIntermediate):
    pass


IJepa_ATTENTION_CLASSES = {
    "eager": IJepaAttention,
    "sdpa": IJepaSdpaAttention,
}


class IJepaLayer(ViTLayer):
    pass


class IJepaEncoder(ViTEncoder):
    pass


class IJepaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = IJepaConfig
    base_model_prefix = "ijepa"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IJepaEmbeddings", "IJepaLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, IJepaEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


class IJepaModel(IJepaPreTrainedModel, ViTModel):
    def __init__(self, config: IJepaConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = IJepaEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = IJepaEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = IJepaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()


class IJepaPooler(ViTPooler):
    pass


class IJepaForImageClassification(IJepaPreTrainedModel, ViTForImageClassification):
    def __init__(self, config: IJepaConfig):
        super().__init__(config)
        self.ijepa = IJepaModel(config, add_pooling_layer=False)
        self.post_init()


__all__ = [
    "IJepaPreTrainedModel",
    "IJepaModel",
    "IJepaForImageClassification",
]
