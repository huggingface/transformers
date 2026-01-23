from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..auto.configuration_auto import AutoBackboneConfig
from ...configuration_utils import PretrainedConfig

@dataclass
class Deimv2Preset:
    hidden_dim: int
    num_queries: int
    num_decoder_layers: int
    backbone: str

DEIMV2_PRESETS: Dict[str, Deimv2Preset] = {
    "base-dinov3-s": Deimv2Preset(hidden_dim=256, num_queries=300, num_decoder_layers=6, backbone="facebook/dinov2-small"),
    "base-dinov3-b": Deimv2Preset(hidden_dim=256, num_queries=300, num_decoder_layers=6, backbone="facebook/dinov2-base"),
}

class Deimv2Config(PretrainedConfig):
    model_type = "deimv2"

    def __init__(
        self,
        backbone_config: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_labels: int = 91,
        # STA and decoder knobs (placeholders)
        sta_num_scales: int = 4,
        use_dense_o2o: bool = True,
        layer_norm_type: str = "rms",
        activation: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config or AutoBackboneConfig.from_pretrained(DEIMV2_PRESETS["base-dinov3-b"].backbone).to_dict()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_labels = num_labels
        self.sta_num_scales = sta_num_scales
        self.use_dense_o2o = use_dense_o2o
        self.layer_norm_type = layer_norm_type
        self.activation = activation
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> "Deimv2Config":
        if preset_name not in DEIMV2_PRESETS:
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(DEIMV2_PRESETS.keys())}")
        preset = DEIMV2_PRESETS[preset_name]
        backbone_config = AutoBackboneConfig.from_pretrained(preset.backbone).to_dict()
        return cls(
            backbone_config=backbone_config,
            hidden_dim=preset.hidden_dim,
            num_queries=preset.num_queries,
            num_decoder_layers=preset.num_decoder_layers,
            **kwargs,
        )
