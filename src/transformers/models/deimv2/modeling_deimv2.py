from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from ...modeling_utils import PreTrainedModel
from ..auto import AutoBackbone
from .configuration_deimv2 import Deimv2Config
from ...utils import logging

logger = logging.get_logger(__name__)

class Deimv2PreTrainedModel(PreTrainedModel):
    config_class = Deimv2Config
    base_model_prefix = "deimv2"
    _no_split_modules = []

class SpatialTuningAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_scales: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim, 1) for _ in range(num_scales)])

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # feat: (B, C, H, W); create a toy pyramid by striding
        feats = []
        x = feat
        for i, p in enumerate(self.proj):
            feats.append(p(x))
            if i < len(self.proj) - 1:
                x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return tuple(feats)

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_queries: int):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True) for _ in range(num_layers)])
        self.decoder = nn.TransformerDecoder(self.layers[0], num_layers=num_layers)

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Use the highest-resolution feature for a stub attention target
        bs = feats[0].size(0)
        tgt = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        # Flatten spatial dims
        f = feats[0].flatten(2).transpose(1, 2)  # (B, HW, C)
        memory = f
        hs = self.decoder(tgt, memory)  # (B, Q, C)
        return hs

class Deimv2Model(Deimv2PreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        out_channels = self.backbone.channels
        hidden = config.hidden_dim
        if isinstance(out_channels, (tuple, list)):
            backbone_dim = out_channels[0]
        else:
            backbone_dim = out_channels
        self.input_proj = nn.Conv2d(backbone_dim, hidden, kernel_size=1)
        self.sta = SpatialTuningAdapter(hidden_dim=hidden, num_scales=config.sta_num_scales)
        self.decoder = SimpleDecoder(hidden_dim=hidden, num_layers=config.num_decoder_layers, num_queries=config.num_queries)

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        features = self.backbone(pixel_values).feature_maps  # tuple of (B, C, H, W)
        x = features[0]
        x = self.input_proj(x)
        feats = self.sta(x)
        hs = self.decoder(feats)  # (B, Q, C)
        return {"decoder_hidden_states": hs}

class Deimv2ForObjectDetection(Deimv2PreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.model = Deimv2Model(config)
        hidden = config.hidden_dim
        self.class_head = nn.Linear(hidden, config.num_labels)
        self.box_head = nn.Linear(hidden, 4)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.model(pixel_values, return_dict=True)
        hs = outputs["decoder_hidden_states"]
        logits = self.class_head(hs)
        boxes = self.box_head(hs).sigmoid()
        out = {"logits": logits, "pred_boxes": boxes}
        # TODO: compute loss if labels provided
        return out
    
    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")
        self.model.backbone.eval()
    
    
