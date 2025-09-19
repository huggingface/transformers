import enum
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PreTrainedModel
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..dac.configuration_dac import DacConfig
from ..dac.modeling_dac import DacEncoder
from ..qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)


class NormalizeTypeConfig(str, enum.Enum):
    NONE = "none"
    L2 = "l2"
    LAYER_NORM = "layernorm"


class TransformerConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_window_layers = max_window_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(**kwargs)


class TransformerWithInputProjectionConfig(TransformerConfig):
    def __init__(self, in_channels: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels


class PerceptionEncoderAVTextEncoderConfig(PretrainedConfig):
    sub_configs = {"sub_config": AutoConfig}
    model_type = "modernbert"

    def __init__(self, nth_layer: Optional[int] = 22, **kwargs):
        self.nth_layer = nth_layer
        self.sub_config = CONFIG_MAPPING[kwargs.get("model_type", "modernbert")](**kwargs)


class VideoEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        backbone: str = "PE-Core-L14-336",
        backbone_checkpoint: Optional[str] = None,  # optional path to local checkpoint
        transformer: Optional[TransformerWithInputProjectionConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.backbone_checkpoint = backbone_checkpoint
        self.transformer = transformer or TransformerWithInputProjectionConfig()


class DACVAEConfig(DacConfig): ...


class PerceptionEncoderAVConfig(PretrainedConfig):
    def __init__(
        self,
        video_encoder: Optional[dict] = None,
        audio_codec: Optional[dict] = None,
        audio_encoder: Optional[dict] = None,
        audio_video_encoder: Optional[dict] = None,
        text_encoder: Optional[dict] = None,
        separate_text_heads: bool = False,
        output_dim: int = 1024,
        contrastive_head_norm_type: NormalizeTypeConfig = NormalizeTypeConfig.L2,
        fixed_len_video: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        audio_codec = audio_codec or {}
        audio_encoder = audio_encoder or {}
        audio_video_encoder = audio_video_encoder or {}
        text_encoder = text_encoder or {}
        video_encoder = video_encoder or {}

        if "transformer" in video_encoder:
            video_encoder["transformer"] = TransformerWithInputProjectionConfig(**video_encoder["transformer"])

        self.video_encoder = VideoEncoderConfig(**video_encoder)
        self.audio_codec = DACVAEConfig(**audio_codec)
        self.audio_encoder = TransformerWithInputProjectionConfig(**audio_encoder)
        self.audio_video_encoder = TransformerWithInputProjectionConfig(**audio_video_encoder)
        self.text_encoder = PerceptionEncoderAVTextEncoderConfig(**text_encoder)
        self.separate_text_heads = separate_text_heads
        self.output_dim = output_dim
        self.contrastive_head_norm_type = contrastive_head_norm_type
        self.fixed_len_video = fixed_len_video


## Patcher
def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    # Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    # Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding))
        return super().forward(x)


class ConvBlock1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.groupnorm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.activation = torch.nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)


class Patcher(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = rearrange(x, "b c (l 1) -> b (c 1) l")
        return x


## Text Encoder
class PerceptionEncoderAVTextEncoder(torch.nn.Module):
    def __init__(self, config: PerceptionEncoderAVTextEncoderConfig):
        super().__init__()
        self.nth_layer = config.nth_layer
        self.model = AutoModel.from_config(config.sub_config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.nth_layer is not None,
        )
        if self.nth_layer is None:
            # Note that `hidden_state[-1]` is not necessarily equivalent to `last_hidden_state`
            # https://huggingface.co/docs/transformers/en/main_classes/output#model-outputs
            return output.last_hidden_state[:, 0]
        return output.hidden_states[self.nth_layer][:, 0]


class AlignModalities(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        btc: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.btc = btc
        self.conv = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        if self.normalize:
            self.layer_norm = torch.nn.LayerNorm(self.out_channels)

    def get_sizes(self, seq, mask):
        if mask is not None:
            sizes = mask.sum(-1)
        else:
            sizes = torch.full((seq.size(0),), seq.size(-1), device=seq.device)
        if sizes.dim() > 1:
            sizes = sizes.squeeze(1)
        return sizes.long()

    def interpolate(self, tgt, tgt_sizes, src_sizes) -> torch.Tensor:
        result = torch.zeros(tgt.size(0), tgt.size(1), src_sizes.max(), device=tgt.device)
        for i, (tgt_row, tgt_size, src_size) in enumerate(zip(tgt, tgt_sizes, src_sizes)):
            tgt_row = tgt_row[:, :tgt_size]
            interpolated = F.interpolate(tgt_row[None], size=src_size, mode="nearest")
            result[i, :, :src_size] = interpolated[0]
        return result

    def forward(self, src, src_mask, tgt, tgt_mask):
        # BxTxC -> BxCxT
        src = src.transpose(1, 2)
        tgt = tgt.transpose(1, 2)

        tgt = self.conv(tgt)

        src_sizes = self.get_sizes(src, src_mask)
        tgt_sizes = self.get_sizes(tgt, tgt_mask)
        if all(src_sizes == tgt_sizes):
            upsampled = tgt
        else:
            upsampled = self.interpolate(tgt, tgt_sizes, src_sizes)

        upsampled = upsampled.permute(0, 2, 1)  # BxCxT -> BxTxC
        if self.normalize:
            upsampled = self.layer_norm(upsampled)
        return upsampled, src_mask


class L2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-6)


class ContrastiveHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm_type: NormalizeTypeConfig = NormalizeTypeConfig.NONE,
    ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)
        if norm_type == NormalizeTypeConfig.L2:
            self.norm = L2()
        elif norm_type == NormalizeTypeConfig.LAYER_NORM:
            self.norm = torch.nn.LayerNorm(out_dim, eps=1e-6)
        else:
            assert norm_type == NormalizeTypeConfig.NONE
            self.norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(self.layer_norm(x))
        return self.norm(projected)


class TextType(enum.Enum):
    audio = "audio"
    visual = "visual"
    audio_visual = "audio_visual"


## Audio Codec
class VAEBottleneck(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = torch.nn.Conv1d(input_dim, bottleneck_dim * 2, kernel_size=1)
        self.out_proj = torch.nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, scale = self.in_proj(z).chunk(2, dim=1)
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return latents, kl


class DACVAEEncoder(DacEncoder): ...


class DacEncoderVAE(torch.nn.Module):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        self.encoder = DACVAEEncoder(config)
        self.bottleneck = VAEBottleneck(config.codebook_size, config.codebook_dim)
        self.hop_length = config.hop_length
        self.mean = 0.0
        self.std = 1.0
        self.sampling_rate = config.sampling_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(self._pad(waveform))
            encoded_frames, _ = self.bottleneck(z)
            encoded_frames = (encoded_frames - self.mean) / self.std
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs

    def feature_idx_to_wav_idx(self, feature_idx, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        orig_freq = sampling_rate
        new_freq = self.sampling_rate
        wav_idx = feature_idx * self.hop_length * (orig_freq / new_freq)
        return wav_idx.int()

    def wav_idx_to_feature_idx(self, wav_idx, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        orig_freq = sampling_rate
        new_freq = self.sampling_rate
        target_length = torch.ceil(new_freq * wav_idx / orig_freq)
        feature_idx = torch.ceil(target_length / self.hop_length)
        return feature_idx.int()


## Transformer


class PerceptionEncoderAVMLP(Qwen3MLP): ...


class PerceptionEncoderAVRMSNorm(Qwen3RMSNorm): ...


class PerceptionEncoderAVRotaryEmbedding(Qwen3RotaryEmbedding): ...


def stack_freqs(cos: torch.Tensor, sin: torch.Tensor):
    dim = cos.size(-1)
    cos = cos.narrow(-1, 0, dim // 2)
    sin = sin.narrow(-1, 0, dim // 2)
    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*cos.size(), 2, 2)
    return freqs_cis


def apply_rotary_pos_emb(q, k, freqs_cis, unsqueeze_dim=1):
    freqs_cis = freqs_cis.unsqueeze(unsqueeze_dim)
    q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q_ * freqs_cis).sum(5).flatten(3), (k_ * freqs_cis).sum(5).flatten(3)


class PerceptionEncoderAVAttention(Qwen3Attention):
    def _reshape_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        B, T, C = x.shape
        # B x T x C -> B x T x C/H x H
        x = x.reshape(B, T, C // heads, heads)
        # B x T x C/H x H -> B x H x T x C/H
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        **kwargs,
    ):
        # The only difference from `Qwen3Attention` is the reshape of Q/K/V
        # We reshape # B x T x C -> B x T x C/H x H, and then permute...
        # Qwen3 reshapes # B x T x C -> B x T x H x C/H

        input_shape = hidden_states.shape[:-1]
        nheads = hidden_states.size(-1) // self.head_dim

        query_states = self.q_norm(self._reshape_heads(self.q_proj(hidden_states), nheads))
        key_states = self.k_norm(self._reshape_heads(self.k_proj(hidden_states), nheads))
        value_states = self._reshape_heads(self.v_proj(hidden_states), nheads)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embeddings)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PerceptionEncoderAVDecoderLayer(Qwen3DecoderLayer): ...


class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.rope_embeddings = PerceptionEncoderAVRotaryEmbedding(config)
        self.layers = torch.nn.ModuleList(
            [PerceptionEncoderAVDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.norm = PerceptionEncoderAVRMSNorm(config.hidden_size, config.rms_norm_eps)

        # output layer
        self.output = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.x_embedder = Patcher(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # Prepend the cls token
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat([padding_mask[:, [0]], padding_mask], dim=1)

        x = rearrange(x, "b l c-> b c l")
        h = self.x_embedder(x)
        h = rearrange(h, "b c l -> b l c")
        original_N = h.shape[1]
        N = h.shape[1]

        cos, sin = self.rope_embeddings(h, torch.arange(h.size(1), device=h.device)[None])
        rope_embeddings = stack_freqs(cos, sin)

        for layer in self.layers:
            h = layer(
                hidden_states=h,
                attention_mask=padding_mask,
                position_embeddings=rope_embeddings,
            )

        # output layer
        if self.norm is not None:
            h = self.norm(h)

        output = self.output(h)
        N = output.shape[1]
        if original_N != N:
            output = output[:, -original_N:]
        return output[:, 1:], output[:, 0]


class TransformerWithInputProjection(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.data_proj = torch.nn.Linear(config.in_channels, config.hidden_size)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return super().forward(self.data_proj(x), *args, **kwargs)


class VideoEncoder(torch.nn.Module):
    def __init__(self, cfg: VideoEncoderConfig):
        super().__init__()
        try:
            from core.vision_encoder import pe
        except ImportError:
            raise RuntimeError(
                "Please install perception_models: `pip install git+https://github.com/facebookresearch/perception_models`"
            )

        self.backbone = pe.CLIP.from_config(cfg.backbone)
        self.proj = torch.nn.Linear(self.backbone.visual.output_dim, cfg.transformer.hidden_size, bias=False)
        self.transformer = TransformerWithInputProjection(cfg.transformer)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = video.shape
        backbone_output = self.backbone.encode_image(video.view(B * N, C, H, W), normalize=True).view(B, N, -1)
        projected = self.proj(backbone_output)
        return self.transformer(projected)


## Audio Video Encoder
class AudioVideoEncoder(TransformerWithInputProjection):
    def __init__(self, config):
        super().__init__(config)
        self.modality_aligner = AlignModalities(
            self.config.hidden_size, self.config.hidden_size, normalize=True, btc=True
        )
        self.concat_modality_proj = torch.nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ):
        video, video_mask = self.modality_aligner(audio, audio_mask, video, video_mask)
        x = torch.cat([audio, video], dim=-1)
        x = self.concat_modality_proj(x)
        return super().forward(x, padding_mask=video_mask)


@dataclass
class PerceptionEncoderAVResult:
    audio_embedding: Optional[torch.Tensor] = None
    text_embedding: Optional[torch.Tensor] = None
    video_embedding: Optional[torch.Tensor] = None
    audio_video_embedding: Optional[torch.Tensor] = None


class PerceptionEncoderAVModel(PreTrainedModel):
    config: PerceptionEncoderAVConfig

    def __init__(
        self,
        cfg: PerceptionEncoderAVConfig,
    ):
        super().__init__(cfg)
        # Synchronize _attn_implementations
        cfg.audio_encoder._attn_implementation = cfg._attn_implementation
        cfg.audio_video_encoder._attn_implementation = cfg._attn_implementation
        cfg.video_encoder.transformer._attn_implementation = cfg._attn_implementation
        self.audio_codec = DacEncoderVAE(cfg.audio_codec)
        self.audio_encoder = TransformerWithInputProjection(cfg.audio_encoder)
        self.audio_video_encoder = AudioVideoEncoder(cfg.audio_video_encoder)
        self.video_encoder = VideoEncoder(cfg.video_encoder)
        self.text_encoder = PerceptionEncoderAVTextEncoder(cfg.text_encoder)

        heads = ["video", "audio", "audio_visual"]
        if cfg.separate_text_heads:
            heads.extend(["audio_text", "visual_text", "audio_visual_text"])

        for head in heads:
            indim = (
                cfg.text_encoder.sub_config.hidden_size
                if head.endswith("text")
                else self.audio_video_encoder.config.hidden_size
            )
            self.add_module(
                f"{head}_head",
                ContrastiveHead(indim, cfg.output_dim, norm_type=cfg.contrastive_head_norm_type),
            )

    def encode_video(self, video):  # b n c h w
        _, raw_feat = self.video_encoder(video)
        return self.video_head(raw_feat)

    def encode_text(self, text: torch.Tensor, kind: TextType = TextType.audio) -> torch.Tensor:
        cls_embedding = self.text_encoder(text)
        head = getattr(self, f"{kind.value}_text_head")
        return head(cls_embedding)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        codec_features = self.audio_codec(audio).transpose(1, 2)
        _, cls_emb = self.audio_encoder(codec_features)
        return self.audio_head(cls_emb)

    def encode_audio_video(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        # Encode audio
        codec_features = self.audio_codec(audio).transpose(1, 2)
        audio_features, _ = self.audio_encoder(codec_features)
        # Encode video
        video_feats, _ = self.video_encoder(video)
        _, cls_emb = self.audio_video_encoder(audio_features, video_feats)
        return self.audio_visual_head(cls_emb)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> PerceptionEncoderAVResult:
        audio_emb = text_emb = video_emb = audio_video_emb = None
        if input_ids is not None:
            text_emb = self.encode_text(input_ids)
        if pixel_values_videos is not None:
            video_emb = self.encode_video(pixel_values_videos)
        if audio is not None:
            audio_emb = self.encode_audio(audio)
        if audio is not None and pixel_values_videos is not None:
            audio_video_emb = self.encode_audio_video(audio, pixel_values_videos)

        return PerceptionEncoderAVResult(
            audio_embedding=audio_emb,
            text_embedding=text_emb,
            video_embedding=video_emb,
            audio_video_embedding=audio_video_emb,
        )

    @classmethod
    def _load_pretrained_model(cls, *args, **kwargs):
        res = super()._load_pretrained_model(*args, **kwargs)
        model = res[0]
        # Otherwise, the rope `freqs` will still be on `meta` device
        model.video_encoder.backbone.visual.rope.init_tensors()
        return res


__all__ = ["PerceptionEncoderAVModel", "PerceptionEncoderAVConfig"]
