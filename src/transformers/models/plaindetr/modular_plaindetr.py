import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..deformable_detr.modeling_deformable_detr import DeformableDetrSinePositionEmbedding, DeformableDetrLearnedPositionEmbedding, DeformableDetrFrozenBatchNorm2d, DeformableDetrMLPPredictionHead
from typing import Optional, Dict, List, Tuple, Union
from ...utils import requires_backends
from ...utils.backbone_utils import load_backbone
from ...modeling_utils import PreTrainedModel
from ..auto import AutoBackbone

try:
    from timm import create_model
except ImportError:
    create_model = None


class PlainDetrSinePositionEmbedding(DeformableDetrSinePositionEmbedding):
    """
    Plain-DETR's sine position embedding with the missing else condition.
    
    This fixes the missing normalization logic when self.normalize=False.
    """
    
    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
            
        y_embed = pixel_mask.cumsum(1, dtype=pixel_values.dtype)
        x_embed = pixel_mask.cumsum(2, dtype=pixel_values.dtype)
        
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        else:
            # This is the missing logic from HuggingFace DeformableDetr!
            y_embed = (y_embed - 0.5) * self.scale
            x_embed = (x_embed - 0.5) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=pixel_values.dtype, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PlainDetrLearnedPositionEmbedding(DeformableDetrLearnedPositionEmbedding):
    pass

def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = PlainDetrLearnedPositionEmbedding(n_steps)
    elif config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=False)
    else:
        raise ValueError(f"Unknown position embedding type: {config.position_embedding_type}")
    
    return position_embedding



class PlainDetrFrozenBatchNorm2d(DeformableDetrFrozenBatchNorm2d):
    pass


class PlainDetrMLPPredictionHead(DeformableDetrMLPPredictionHead):
    pass


