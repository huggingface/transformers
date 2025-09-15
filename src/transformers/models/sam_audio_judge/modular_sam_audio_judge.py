from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

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


def batch_audio(audios: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    wavs = [wav.mean(0) for wav in audios]
    sizes = torch.tensor([wav.size(-1) for wav in wavs])
    return pad_sequence(wavs, batch_first=True).unsqueeze(1), sizes


def mask_from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    return torch.arange(sizes.max()).expand(len(sizes), -1) < sizes.unsqueeze(1)


class SamAudioJudgeTransformerConfig(TransformerConfig): ...


# NOTE: rename `DACVAEConfig` to `SamAudioJudgeDACVAEConfig` causes import errors
# The generated DAC encoder layers have a `DACVAEConfig` type annotation which doesn't exist
# If we rename this to something else...
class DACVAEConfig(DACVAEConfig): ...


class SamAudioJudgeModernBertConfig(PerceptionEncoderAVModernBertConfig): ...


class SamAudioJudgeConfig(PretrainedConfig):
    audio_codec: DACVAEConfig
    audio_encoder: SamAudioJudgeTransformerConfig
    finetune_encoder: SamAudioJudgeTransformerConfig
    text_encoder: SamAudioJudgeModernBertConfig

    def __init__(
        self,
        audio_codec: Optional[dict] = None,
        audio_encoder: Optional[dict] = None,
        text_encoder: Optional[dict] = None,
        finetune_encoder: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        audio_codec = audio_codec or {}
        self.audio_codec = DACVAEConfig(**audio_codec)
        self.audio_encoder = SamAudioJudgeTransformerConfig.from_dict(audio_encoder or {})
        self.finetune_encoder = SamAudioJudgeTransformerConfig.from_dict(finetune_encoder or {})
        self.text_encoder = SamAudioJudgeModernBertConfig.from_dict(text_encoder or {})

    def to_dict(self):
        output = super().to_dict()
        # convert any sub-configs that weren't converted by `super().to_dict()`
        return {k: asdict(v) if is_dataclass(v) else v for k, v in output.items()}


class SamAudioJudgeDACVAE(DACVAE): ...


class SamAudioJudgeTransformer(Transformer): ...


class SamAudioJudgeModernBertModel(PerceptionEncoderAVTextEncoderModernBertModel): ...


class AlignModalities(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        with_gate: bool = True,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.normalize = normalize
        if self.normalize:
            self.layer_norm = torch.nn.LayerNorm(out_channels)

        self.gate = None
        if with_gate:
            self.gate = torch.nn.Parameter(torch.tensor([0.0]))

        self.out_channels = out_channels

    def forward(self, anchor: torch.Tensor, tgt: Optional[torch.Tensor] = None):
        """
        Align video features to the input audio features

        Args:
            anchor (torch.Tensor): Input anchor tensor of shape (B, T, C), where B is batch size, C is channel size, and T is sequence length.
            tgt (Optional[torch.Tensor]): Optional features tensor to be aligned to anchor, expected shape (B, in_channels, T).
        """
        if tgt is None:
            return anchor

        post_conv = self.conv(tgt)
        post_conv = post_conv.permute(0, 2, 1)  # BCT -> BTC

        if self.normalize:
            post_conv = self.layer_norm(post_conv)

        if self.gate is None:
            return post_conv
        else:
            return anchor + self.gate.tanh() * post_conv


class MeanPool(torch.nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x has shape B x T x C
        if mask is None:
            return x.mean(dim=1)
        else:
            sizes = mask.sum(-1)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / sizes.unsqueeze(-1)


class Head(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        self.proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.cls_emb = MeanPool()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.cls_emb(self.proj(x), mask)


@dataclass
class JudgeOutput:
    overall: torch.Tensor
    recall: torch.Tensor
    precision: torch.Tensor
    faithfulness: torch.Tensor


class SamAudioJudgeModel(PreTrainedModel):
    config: SamAudioJudgeConfig

    def __init__(self, cfg: SamAudioJudgeConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.audio_codec = SamAudioJudgeDACVAE(cfg.audio_codec)
        self.text_encoder = SamAudioJudgeModernBertModel(cfg.text_encoder)
        self.audio_encoder = SamAudioJudgeTransformer(**asdict(cfg.audio_encoder))
        self.finetune_encoder = SamAudioJudgeTransformer(**asdict(cfg.finetune_encoder))
        self.text_proj = torch.nn.Linear(cfg.text_encoder.hidden_size, cfg.audio_encoder.dim, bias=False)
        self.cat_audio_proj = torch.nn.Linear(2 * cfg.audio_encoder.dim, cfg.finetune_encoder.in_channels)
        self.aligner = AlignModalities(cfg.audio_encoder.dim, cfg.finetune_encoder.in_channels, with_gate=False)
        self.cat_aligned = torch.nn.Linear(2 * cfg.finetune_encoder.in_channels, cfg.finetune_encoder.in_channels)
        self.head = torch.nn.Linear(cfg.finetune_encoder.out_channels, 5, bias=False)
        self.pool = MeanPool()
        self.mean = torch.nn.Parameter(torch.zeros(4, requires_grad=False))
        self.std = torch.nn.Parameter(torch.ones(4, requires_grad=False))

    @property
    def sample_rate(self):
        return self.cfg.audio_codec.sample_rate

    def forward(
        self, input_wavs, hyp_wavs, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> JudgeOutput:
        text_features = self.text_proj(self.text_encoder(input_ids, attention_mask))
        device = text_features.device
        hyp_wavs, hyp_sizes = batch_audio(hyp_wavs)
        input_wavs, input_sizes = batch_audio(input_wavs)

        assert (hyp_sizes == input_sizes).all().item(), "Input and separated audio must be the same size"

        mask = mask_from_sizes(self.audio_codec.wav_idx_to_feature_idx(hyp_sizes))
        if mask is not None:
            mask = mask.to(device)

        input_codec_feats, hyp_codec_feats = (
            self.audio_codec(torch.cat([input_wavs, hyp_wavs], dim=0).to(device)).transpose(1, 2).chunk(2, 0)
        )
        input_features, _ = self.audio_encoder(input_codec_feats, padding_mask=mask)
        hyp_features, _ = self.audio_encoder(hyp_codec_feats, padding_mask=mask)
        audio_features = self.cat_audio_proj(torch.cat([input_features, hyp_features], dim=2))
        aligned = self.aligner(audio_features, text_features.unsqueeze(-1))
        finetune_inp = self.cat_aligned(torch.cat([audio_features, aligned.expand_as(audio_features)], dim=2))
        final_features, _ = self.finetune_encoder(finetune_inp, padding_mask=mask)
        result = self.pool(self.head(final_features), mask)[:, :4]
        de_normalized = result * self.std + self.mean
        return JudgeOutput(*de_normalized.chunk(4, dim=1))


__all__ = ["SamAudioJudgeModel", "SamAudioJudgeConfig", "JudgeOutput"]
