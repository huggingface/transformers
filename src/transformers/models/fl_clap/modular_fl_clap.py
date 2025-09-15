from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

import torch

from ...configuration_utils import PretrainedConfig
from ...modeling_utils import PreTrainedModel
from ..perception_encoder_av.configuration_perception_encoder_av import (
    DACVAEConfig,
    PerceptionEncoderAVModernBertConfig,
    TransformerConfig,
)
from ..perception_encoder_av.modeling_perception_encoder_av import (
    DACVAE,
    PerceptionEncoderAVTextEncoderModernBertModel,
    Transformer,
)


class FlClapTransformerConfig(TransformerConfig): ...


# NOTE: rename `DACVAEConfig` to `FlClapDACVAEConfig` causes import errors
# The generated DAC encoder layers have a `DACVAEConfig` type annotation which doesn't exist
# If we rename this to something else...
class DACVAEConfig(DACVAEConfig): ...


class FlClapModernBertConfig(PerceptionEncoderAVModernBertConfig): ...


class FLCLAPConfig(PretrainedConfig):
    audio_codec: DACVAEConfig
    audio_encoder: FlClapTransformerConfig
    text_encoder: FlClapModernBertConfig

    def __init__(
        self,
        audio_codec: Optional[dict] = None,
        audio_encoder: Optional[dict] = None,
        text_encoder: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        audio_codec = audio_codec or {}
        self.audio_codec = DACVAEConfig(**audio_codec)
        self.audio_encoder = FlClapTransformerConfig.from_dict(audio_encoder or {})
        self.text_encoder = FlClapModernBertConfig.from_dict(text_encoder or {})

    def to_dict(self):
        output = super().to_dict()
        # convert any sub-configs that weren't converted by `super().to_dict()`
        return {k: asdict(v) if is_dataclass(v) else v for k, v in output.items()}


class FlClapDACVAE(DACVAE):
    def __init__(self, config: DACVAEConfig):
        super().__init__(config)


class FlClapTransformer(Transformer): ...


class FlClapModernBertModel(PerceptionEncoderAVTextEncoderModernBertModel): ...


@dataclass
class JudgeOutput:
    overall: torch.Tensor
    recall: torch.Tensor
    precision: torch.Tensor
    faithfulness: torch.Tensor


class ContrastiveHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.layer_norm(x))


@dataclass
class FLCLAPOutput:
    text_embeddings: Optional[torch.Tensor] = None
    audio_embeddings: Optional[torch.Tensor] = None


class FLCLAPModel(PreTrainedModel):
    config: FLCLAPConfig

    def __init__(self, cfg: FLCLAPConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.text_encoder = FlClapModernBertModel(cfg.text_encoder)
        self.audio_encoder = FlClapTransformer(**asdict(cfg.audio_encoder))
        self.audio_codec = FlClapDACVAE(cfg.audio_codec)

        self.audio_head = ContrastiveHead(cfg.audio_encoder.dim, cfg.out_dim)
        self.text_head = ContrastiveHead(cfg.text_encoder.hidden_size, cfg.out_dim)

    @property
    def sample_rate(self):
        return self.cfg.audio_codec.sample_rate

    def encode_text(self, text: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.text_head(self.text_encoder(text, attention_mask=attention_mask))

    def encode_audio(self, audio: torch.Tensor):
        features = self.audio_codec(audio)
        emb, _ = self.audio_encoder(features.transpose(1, 2))
        return self.audio_head(emb)

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        audio_emb = text_emb = None
        if audio is not None:
            audio_emb = self.encode_audio(audio)
        if input_ids is not None:
            text_emb = self.encode_text(input_ids, attention_mask=attention_mask)
        return FLCLAPOutput(text_embeddings=text_emb, audio_embeddings=audio_emb)


__all__ = ["FLCLAPModel", "FLCLAPConfig", "JudgeOutput"]
