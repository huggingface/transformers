import enum
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Optional

from ...configuration_utils import PretrainedConfig
from ..dac.configuration_dac import DacConfig


class NormalizeType(str, enum.Enum):
    NONE = "none"
    L2 = "l2"
    LAYER_NORM = "layernorm"


@dataclass(frozen=True, kw_only=True)
class Config:
    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)


@dataclass(frozen=True, kw_only=True)
class DACVAEConfig(Config):
    encoder_dim: int = 64
    encoder_rates: list[int] = field(default_factory=lambda: [2, 8, 10, 12])
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [12, 10, 8, 2])
    n_codebooks: int = 16
    codebook_size: int = 1024
    codebook_dim: int = 128
    quantizer_dropout: bool = False
    sample_rate: int = 48000


@dataclass(frozen=True, kw_only=True)
class TransformerConfig(Config):
    in_channels: int = 128
    dim: int = 1024
    n_heads: int = 8
    n_layers: int = 16
    out_channels: int = 1024
    dropout: float = 0.1
    pre_norm: bool = True
    norm_eps: float = 1e-5
    qk_norm: bool = True
    fc_bias: bool = False
    ffn_exp: int = 4
    ffn_dim_multiplier: int = 1
    multiple_of: int = 64
    non_linearity: str = "swiglu"
    use_rope: bool = True
    max_positions: int = 10000
    patch_size: int = 1


@dataclass(frozen=True, kw_only=True)
class TextEncoderConfig(Config):
    dim: int


@dataclass(frozen=True, kw_only=True)
class PETextEncoder(TextEncoderConfig):
    dim: int = 1024


@dataclass(frozen=True, kw_only=True)
class ModernBERTConfig(TextEncoderConfig):
    model_id: str = "answerdotai/ModernBERT-base"
    pad_mode: str = "longest"
    max_length: int = 1024
    dim: int = 768
    nth_layer: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class VideoEncoderConfig(Config):
    backbone: str = "PE-Core-L14-336"
    backbone_checkpoint: Optional[str] = None  # optional path to local checkpoint
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            in_channels=1792,
            dim=1792,
            n_heads=14,
            n_layers=4,
            out_channels=1792,
        )
    )

    @classmethod
    def from_dict(cls, kwargs):
        if "transformer" in kwargs:
            kwargs["transformer"] = TransformerConfig.from_dict(kwargs["transformer"])
        return cls(**kwargs)


EMPTY_DICT = {}


class PerceptionEncoderAVConfig(PretrainedConfig):
    def __init__(
        self,
        video_encoder: dict = EMPTY_DICT,
        dacvae_config: dict = EMPTY_DICT,
        audio_encoder: dict = EMPTY_DICT,
        audio_video_encoder: dict = EMPTY_DICT,
        text_encoder: dict = EMPTY_DICT,
        separate_text_heads: bool = False,
        output_dim: int = 1024,
        contrastive_head_norm_type: str = "L2",
        fixed_len_video: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_encoder = VideoEncoderConfig.from_dict(video_encoder)
        self.dacvae_config = DacConfig(**dacvae_config)
        self.audio_encoder = TransformerConfig.from_dict(audio_encoder)
        self.audio_video_encoder = TransformerConfig.from_dict(audio_video_encoder)
        self.text_encoder = ModernBERTConfig.from_dict(text_encoder)
        self.separate_text_heads = separate_text_heads
        self.output_dim = output_dim
        self.contrastive_head_norm_type = NormalizeType[contrastive_head_norm_type.upper()]
        self.fixed_len_video = fixed_len_video

    def to_dict(self):
        output = super().to_dict()
        # convert any sub-configs that weren't converted by `super().to_dict()`
        return {k: asdict(v) if is_dataclass(v) else v for k, v in output.items()}
