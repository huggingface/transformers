from typing import Optional, Tuple, Dict, Any, List
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
        # Use the first layer instance to create the TransformerDecoder wrapper but keep module list for clarity
        self.decoder = nn.TransformerDecoder(self.layers[0], num_layers=num_layers)

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Use the highest-resolution feature for a stub attention target (feats[0] is highest-res)
        bs = feats[0].size(0)
        tgt = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # (B, Q, C)
        # Flatten spatial dims
        f = feats[0].flatten(2).transpose(1, 2)  # (B, HW, C) -> memory
        memory = f
        hs = self.decoder(tgt, memory)  # (B, Q, C)
        return hs


class Deimv2Model(Deimv2PreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        out_channels = getattr(self.backbone, "channels", None)
        hidden = config.hidden_dim
        if isinstance(out_channels, (tuple, list)):
            backbone_dim = out_channels[0]
        elif isinstance(out_channels, int):
            backbone_dim = out_channels
        else:
            # If AutoBackbone returns a model that exposes feature maps only at call time,
            # use a conservative default (user should pass backbone_config with channel info)
            backbone_dim = hidden

        self.input_proj = nn.Conv2d(backbone_dim, hidden, kernel_size=1)
        self.sta = SpatialTuningAdapter(hidden_dim=hidden, num_scales=config.sta_num_scales)
        self.decoder = SimpleDecoder(hidden_dim=hidden, num_layers=config.num_decoder_layers, num_queries=config.num_queries)

        # standard HF initialization hook
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        # Run backbone. AutoBackbone implementations can return a dataclass or tuple.
        backbone_outputs = self.backbone(pixel_values)
        # Try common attribute names
        if hasattr(backbone_outputs, "feature_maps"):
            features = backbone_outputs.feature_maps
        elif isinstance(backbone_outputs, (tuple, list)) and len(backbone_outputs) > 0:
            # assume first element is tuple/list of feature maps, or it's the feature maps themselves
            candidate = backbone_outputs[0]
            if isinstance(candidate, (tuple, list)):
                features = candidate
            else:
                # If backbone returns feature maps directly as the first element
                features = backbone_outputs
        else:
            # fallback: assume the backbone itself returned the feature maps
            features = backbone_outputs

        # Ensure features is a tuple/list and has at least one feature map
        if isinstance(features, torch.Tensor):
            features = (features,)

        # Take highest resolution feature (first)
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

        # initialize head weights (HF-like)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.model(pixel_values, return_dict=True)
        hs = outputs["decoder_hidden_states"]  # (B, Q, C)
        logits = self.class_head(hs)           # (B, Q, num_labels)
        boxes = self.box_head(hs).sigmoid()    # (B, Q, 4) normalized cxcywh

        out = {"logits": logits, "pred_boxes": boxes}

        # Minimal loss placeholder — replace with full DEIMCriterion integration
        if labels is not None:
            # Example expected format in labels: {"class_labels": LongTensor[B,Q], "boxes": FloatTensor[B,Q,4]}
            # If your label format is different adapt accordingly.
            loss = torch.tensor(0.0, device=logits.device)
            try:
                target_logits = labels.get("class_labels", None)
                target_boxes = labels.get("boxes", None)
                if target_logits is not None:
                    # flatten for CE
                    loss_cls = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_logits.view(-1))
                else:
                    loss_cls = torch.tensor(0.0, device=logits.device)
                if target_boxes is not None:
                    loss_box = nn.functional.l1_loss(boxes, target_boxes)
                else:
                    loss_box = torch.tensor(0.0, device=logits.device)
                loss = loss_cls + loss_box
            except Exception:
                # on mismatch or other issue, return zero loss but log a hint
                logger.warning("Labels provided but loss computation failed — ensure labels contain 'class_labels' and 'boxes' formatted as [B, Q, ...].")
            out["loss"] = loss

        return out

    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")
        self.model.backbone.eval()
