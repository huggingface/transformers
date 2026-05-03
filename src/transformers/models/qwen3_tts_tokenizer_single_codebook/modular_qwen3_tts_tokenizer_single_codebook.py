import math
import operator
from dataclasses import dataclass
from functools import cache
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ...utils.hub import cached_file
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    AMPBlock,
    DiTDecoderLayer,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
    Qwen2_5OmniToken2WavModel,
    SnakeBeta,
    TorchActivation1d,
)
from .configuration_qwen3_tts_tokenizer_single_codebook import (
    Qwen3TTSTokenizerSingleCodebookConfig,
    Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig,
    Qwen3TTSTokenizerSingleCodebookDecoderConfig,
    Qwen3TTSTokenizerSingleCodebookDiTConfig,
    Qwen3TTSTokenizerSingleCodebookEncoderConfig,
)


logger = logging.get_logger(__name__)


class Qwen3TTSTokenizerSingleCodebookPreTrainedModel(PreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookConfig
    base_model_prefix = "model"
    _no_split_modules = []
    _supports_sdpa = True


class Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(DiTDecoderLayer):
    pass


class Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel(Qwen2_5OmniToken2WavBigVGANModel):
    config: Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig):
        super().__init__(config)
        # Override conv_pre: kernel 5 + padding 2 instead of parent's kernel 7 + padding 3
        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 5, 1, padding=2)
        # Override resblocks: SingleCodebook uses causal AMPBlock with causal_type parameter
        self.resblocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookAMPBlock(
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    dilation,
                    "1" if layer_idx > 1 else "2",
                )
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )


@cache
def _v1_mel_filters(device, n_mels: int) -> torch.Tensor:
    """Compute mel filterbank using audio_utils (Whisper-compatible: 16kHz, n_fft=400)."""
    from ...audio_utils import mel_filter_bank

    if n_mels not in {80, 128}:
        raise ValueError(f"Unsupported n_mels: {n_mels}")
    mel = mel_filter_bank(
        num_frequency_bins=1 + 400 // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(mel).to(device)


def _v1_log_mel_spectrogram(audio, n_mels=80, padding=0, device=None):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _v1_mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def _v1_get_T_after_cnn(L_in, dilation=1):
    for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def _v1_get_mel_audio(audio, padding=False, audio_vq_ds_rate=1, n_mels=128):
    audio_len = len(audio)
    if padding:
        reduction = 160 * 2 * audio_vq_ds_rate
        audio_pad = math.ceil(audio_len / reduction) * reduction - audio_len
        mel = _v1_log_mel_spectrogram(audio, n_mels=n_mels, padding=audio_pad)
    else:
        mel = _v1_log_mel_spectrogram(audio, n_mels=n_mels)
    return mel


def _v1_sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ── VQ core classes (inference-only port of core_vq.py) ──────────────────────


class Qwen3TTSTokenizerSingleCodebookEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance (inference subset)."""

    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.99,
        epsilon=1e-5,
        threshold_ema_dead_code=2.0,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        # buffers are held by DistributedResidualVectorQuantization and passed at call-time
        self.inited = None
        self.cluster_size = None
        self.embed = None
        self.embed_avg = None

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices

    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)

    def encode(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        embed_ind = self.quantize(x)
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers
        return self.dequantize(embed_ind)


class Qwen3TTSTokenizerSingleCodebookVectorQuantization(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        decay=0.99,
        epsilon=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2.0,
        commitment_weight=1.0,
    ):
        super().__init__()
        _codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim
        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        self._codebook = Qwen3TTSTokenizerSingleCodebookEuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    def encode(self, x, buffers):
        x = self.project_in(x)
        return self._codebook.encode(x, buffers)

    def decode(self, embed_ind, buffers):
        quantize = self._codebook.decode(embed_ind, buffers)
        return self.project_out(quantize)


class Qwen3TTSTokenizerSingleCodebookDistributedRVQ(nn.Module):
    """Distributed residual VQ (inference subset of DistributedResidualVectorQuantization)."""

    def __init__(self, *, num_quantizers, quantize_dropout=False, rand_num_quant=None, **kwargs):
        super().__init__()
        codebook_size = kwargs["codebook_size"]
        codebook_dim = kwargs.get("codebook_dim") or kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]

        if isinstance(kmeans_init, bool):
            if not kmeans_init:
                embed = torch.empty(num_quantizers, codebook_size, codebook_dim)
                nn.init.kaiming_uniform_(embed)
                inited = True
            else:
                embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
                inited = False
        else:
            raise TypeError("kmeans_init should be bool")

        self.register_buffer("inited", torch.Tensor([[inited]] * num_quantizers))
        self.register_buffer("cluster_size", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerSingleCodebookVectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def encode(self, x, n_q=None):
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            buffers = [self.inited[i], self.cluster_size[i], self.embed[i], self.embed_avg[i]]
            indices = layer.encode(residual, buffers)
            quantized = layer.decode(indices, buffers)
            residual = residual - quantized
            all_indices.append(indices)
        return torch.stack(all_indices)

    def decode(self, q_indices):
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            buffers = [self.inited[i], self.cluster_size[i], self.embed[i], self.embed_avg[i]]
            quantized_out = quantized_out + self.layers[i].decode(indices, buffers)
        return quantized_out


class Qwen3TTSTokenizerSingleCodebookDistributedGroupRVQ(nn.Module):
    """Distributed group RVQ (inference subset of DistributedGroupResidualVectorQuantization)."""

    def __init__(self, *, num_groups, num_quantizers, quantize_dropout=False, rand_num_quant=None, **kwargs):
        super().__init__()
        self.rvqs = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookDistributedRVQ(
                    num_quantizers=num_quantizers,
                    quantize_dropout=quantize_dropout,
                    rand_num_quant=rand_num_quant,
                    **kwargs,
                )
                for _ in range(num_groups)
            ]
        )
        self.num_groups = num_groups

    def encode(self, x, n_q=None):
        x_lst = torch.chunk(x, chunks=self.num_groups, dim=1)
        return torch.stack([mod.encode(item, n_q) for mod, item in zip(self.rvqs, x_lst)], dim=1)

    def decode(self, q_indices):
        q_indices_lst = torch.chunk(q_indices, chunks=self.num_groups, dim=1)
        return torch.cat([mod.decode(item.squeeze(1)) for mod, item in zip(self.rvqs, q_indices_lst)], dim=1)


# ── Whisper encoder classes (port of vq/whisper_encoder.py) ──────────────────


class _V1Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class _V1ConvTranspose1d(nn.ConvTranspose1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class Qwen3TTSTokenizerSingleCodebookLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class Qwen3TTSTokenizerSingleCodebookMultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Qwen3TTSTokenizerSingleCodebookLinear(n_state, n_state)
        self.key = Qwen3TTSTokenizerSingleCodebookLinear(n_state, n_state, bias=False)
        self.value = Qwen3TTSTokenizerSingleCodebookLinear(n_state, n_state)
        self.out = Qwen3TTSTokenizerSingleCodebookLinear(n_state, n_state)

    def forward(self, x, cu_seqlens=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self._qkv_attention_manual(q, k, v, cu_seqlens=cu_seqlens)
        return self.out(x)

    def _qkv_attention_manual(self, q, k, v, cu_seqlens):
        n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        scale = head_dim**-0.5

        q = q.view(n_ctx, self.n_head, head_dim)
        k = k.view(n_ctx, self.n_head, head_dim)
        v = v.view(n_ctx, self.n_head, head_dim)

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)

        q_padded = torch.zeros(batch_size, max_seqlen, self.n_head, head_dim, dtype=q.dtype, device=q.device)
        k_padded = torch.zeros_like(q_padded)
        v_padded = torch.zeros_like(q_padded)

        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_len = seqlens[i]
            q_padded[i, :seq_len] = q[start_idx:end_idx]
            k_padded[i, :seq_len] = k[start_idx:end_idx]
            v_padded[i, :seq_len] = v[start_idx:end_idx]

        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        attn_mask = (
            (torch.arange(max_seqlen, device=q.device)[None, :] < torch.tensor(seqlens, device=q.device)[:, None])
            .unsqueeze(1)
            .unsqueeze(2)
        )
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -torch.finfo(q.dtype).max)

        attn_scores = torch.matmul(q_padded, k_padded.transpose(-2, -1)) * scale + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_padded)
        context = context.transpose(1, 2).contiguous().view(batch_size, max_seqlen, n_state)
        return torch.cat([context[i, : seqlens[i]] for i in range(batch_size)], dim=0)


class Qwen3TTSTokenizerSingleCodebookResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, enable_mp=False, sequence_parallel=False):
        super().__init__()
        n_mlp = n_state * 4
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp_ln = nn.LayerNorm(n_state)
        self.attn = Qwen3TTSTokenizerSingleCodebookMultiHeadAttention(n_state, n_head)
        self.mlp = nn.Sequential(
            Qwen3TTSTokenizerSingleCodebookLinear(n_state, n_mlp),
            nn.GELU(),
            Qwen3TTSTokenizerSingleCodebookLinear(n_mlp, n_state),
        )

    def forward(self, x, cu_seqlens=None):
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class Qwen3TTSTokenizerSingleCodebookWhisperEncoder(nn.Module):
    def __init__(
        self,
        n_mels,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        n_window=1500,
        output_dim=512,
        grad_checkpointing=False,
        enable_mp=False,
        audio_sequence_parallel=False,
    ):
        super().__init__()
        self.conv1 = _V1Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = _V1Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", _v1_sinusoids(n_ctx, n_state))
        self.n_layer = n_layer
        self.n_mels = n_mels
        self.blocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookResidualAttentionBlock(
                    n_state, n_head, enable_mp=enable_mp, sequence_parallel=audio_sequence_parallel
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_post = nn.LayerNorm(n_state)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(n_state, output_dim)
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        self.output_dim = output_dim
        self.n_head = n_head
        self.n_state = n_state
        self.n_window = n_window


class Qwen3TTSTokenizerSingleCodebookWhisperEncoderVQ(Qwen3TTSTokenizerSingleCodebookWhisperEncoder):
    """WhisperEncoder extended with a VQ bottleneck (inference-only port of WhisperEncoderVQ)."""

    def __init__(
        self,
        n_mels,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        n_window=1500,
        output_dim=512,
        grad_checkpointing=False,
        enable_mp=False,
        audio_sequence_parallel=False,
        audio_vq_layers=-1,
        audio_vq_type="NULL",
        audio_vq_codebook_size=4096,
        audio_vq_pe=False,
        audio_vq_commit_loss=0.0,
        audio_vq_out_commit_loss=0.0,
        audio_vq_no_quantize=False,
        audio_vq_ff_layer=0,
        audio_vq_threshold_ema_dead_code=0.1,
        audio_vq_codebook_dim=None,
        audio_vq_ds_rate=None,
    ):
        super().__init__(
            n_mels,
            n_ctx,
            n_state,
            n_head,
            n_layer,
            n_window,
            output_dim,
            grad_checkpointing,
            enable_mp,
            audio_sequence_parallel,
        )
        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_type = audio_vq_type
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_commit_loss = audio_vq_commit_loss
        self.audio_vq_out_commit_loss = audio_vq_out_commit_loss
        self.audio_vq_no_quantize = audio_vq_no_quantize
        self.audio_vq_ff_layer = audio_vq_ff_layer

        if audio_vq_layers > 0:
            self.vq_feature_dim = self.n_state
            self.audio_vq_ds_rate = 1
        else:
            raise NotImplementedError(f"Unsupported audio_vq_layers: {audio_vq_layers}")

        if self.audio_vq_ds_rate == audio_vq_ds_rate:
            self.audio_vq_downsample = nn.Identity()
            self.audio_vq_upsample = nn.Identity()
        else:
            assert audio_vq_ds_rate % self.audio_vq_ds_rate == 0
            stride = audio_vq_ds_rate // self.audio_vq_ds_rate
            self.audio_vq_downsample = _V1Conv1d(
                self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride
            )
            self.audio_vq_upsample = _V1ConvTranspose1d(
                self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride
            )
            self.audio_vq_ds_rate = audio_vq_ds_rate

        codebook_dim_for_vq = audio_vq_codebook_dim if audio_vq_codebook_dim is not None else self.vq_feature_dim
        if audio_vq_type == "GRVQ":
            self.audio_quantizer = Qwen3TTSTokenizerSingleCodebookDistributedGroupRVQ(
                codebook_size=audio_vq_codebook_size,
                dim=self.vq_feature_dim,
                codebook_dim=codebook_dim_for_vq,
                num_groups=1,
                num_quantizers=1,
                kmeans_init=False,
                threshold_ema_dead_code=audio_vq_threshold_ema_dead_code,
            )
        else:
            raise NotImplementedError(f"Unsupported audio_vq_type: {audio_vq_type}")

        if self.audio_vq_pe:
            self.project_after_vq_pe = nn.Linear(self.n_state, self.n_state)

    def _do_quantize(self, x, pe=None, y=None):
        x = x.unsqueeze(0)
        x = self.audio_vq_downsample(x.transpose(1, 2))
        x = x.transpose(1, 2)
        indices = self.audio_quantizer.encode(x)
        x = self.audio_quantizer.decode(indices)
        indices = indices.squeeze(2).squeeze(1)
        x, indices = x.squeeze(0), indices.squeeze(0)
        if self.audio_vq_pe:
            x = x + pe
            x = self.project_after_vq_pe(x)
        x = self.audio_vq_upsample(x.unsqueeze(0).transpose(1, 2))
        x = x.transpose(1, 2).squeeze(0)
        return x, indices, {}

    def forward(
        self, x_list, audio_mellens, audio_aftercnnlens, audio_seqlens, return_indices=False, audio_pitchs=None
    ):
        aftercnn_x_list = []
        pe_for_vq_list = []
        for each_x in x_list:
            for each_x_split in each_x.split(self.n_window * 2, dim=1):
                each_x_split = F.gelu(self.conv1(each_x_split))
                each_x_split = F.gelu(self.conv2(each_x_split))
                each_x_split = each_x_split.permute(1, 0)
                each_positional_embedding_split = self.positional_embedding[: each_x_split.shape[0]]
                aftercnn_x_list.append(each_x_split + each_positional_embedding_split.to(each_x_split.dtype))
                pe_for_vq_split = self.positional_embedding[: each_x_split.shape[0] // self.audio_vq_ds_rate]
                pe_for_vq_list.append(pe_for_vq_split.to(each_x_split.dtype))

        pe_for_vq = torch.cat(pe_for_vq_list, dim=0)
        x = torch.cat(aftercnn_x_list, dim=0)

        output_list = []
        for item in audio_aftercnnlens:
            while item > self.n_window:
                output_list.append(self.n_window)
                item -= self.n_window
            output_list.append(item)

        cu_seqlens_list = list(accumulate(output_list, func=operator.add, initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens_list).to(device=x.device, dtype=torch.int32)

        layer_id = 0
        for block in self.blocks:
            layer_id += 1
            x = block(x, cu_seqlens=cu_seqlens)
            if self.audio_vq_layers == layer_id:
                x, indices, vq_stats = self._do_quantize(x, pe_for_vq)
                if return_indices:
                    return x, indices

        if self.avg_pooler:
            x_list_split = x.split(audio_aftercnnlens, dim=0)
            token_x_list = []
            for xi in x_list_split:
                xi = xi.permute(1, 0)
                xi = self.avg_pooler(xi)
                xi = xi.permute(1, 0)
                token_x_list.append(xi)
            x = torch.cat(token_x_list, dim=0)

        x = self.ln_post(x)
        x = self.proj(x)

        output = torch.zeros((x.size(0) + len(audio_seqlens) * 2, x.size(1)), device=x.device, dtype=x.dtype)
        audio_seqlens_acc = list(accumulate(audio_seqlens, func=operator.add, initial=0))
        start_ids = torch.tensor(audio_seqlens_acc[:-1], device=x.device, dtype=torch.int32)
        end_ids = torch.tensor(audio_seqlens_acc[1:], device=x.device, dtype=torch.int32) - 1
        audio_tokens_mask = torch.ones(output.size(0), device=x.device, dtype=torch.bool)
        audio_tokens_mask[start_ids] = False
        audio_tokens_mask[end_ids] = False
        output[start_ids] = self.audio_bos_eos_token.weight[0].to(x.dtype)
        output[end_ids] = self.audio_bos_eos_token.weight[1].to(x.dtype)
        output[audio_tokens_mask] = x

        if self.audio_vq_type != "NULL":
            return output, vq_stats
        return output


# ── Qwen3TTSTokenizerSingleCodebookXVectorExtractor (lazy external imports) ─────────────────────────────────


class Qwen3TTSTokenizerSingleCodebookMelSpectrogramFeatures(nn.Module):
    """Mel spectrogram extractor used by Qwen3TTSTokenizerSingleCodebookXVectorExtractor."""

    def __init__(
        self,
        filter_length=1024,
        hop_length=160,
        win_length=640,
        n_mel_channels=80,
        mel_fmin=0,
        mel_fmax=8000,
        sampling_rate=16000,
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate

    def extract(self, audio):
        from ...audio_utils import mel_filter_bank

        y = audio
        if len(y.shape) == 3:
            y = y.squeeze(1) if y.shape[1] == 1 else y.squeeze(2)
        mel = mel_filter_bank(
            num_frequency_bins=1 + self.filter_length // 2,
            num_mel_filters=self.n_mel_channels,
            min_frequency=self.mel_fmin,
            max_frequency=self.mel_fmax,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        mel_basis = torch.from_numpy(mel).float().to(y.device)
        hann_window = torch.hann_window(self.win_length).to(y.device)
        pad = int((self.filter_length - self.hop_length) / 2)
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(mel_basis, spec)
        return torch.log(torch.clamp(spec, min=1e-5))


class Qwen3TTSTokenizerSingleCodebookXVectorExtractor(nn.Module):
    """Speaker x-vector extractor using an ONNX model (campplus.onnx).

    External dependencies (onnxruntime, sox, torchaudio) are imported lazily
    so that the main transformers package does not require them.
    """

    def __init__(self, audio_codec_with_xvector):
        super().__init__()
        import onnxruntime
        import sox

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.ort_session = onnxruntime.InferenceSession(
            audio_codec_with_xvector, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.tfm = sox.Transformer()
        self.tfm.norm(db_level=-6)
        self.mel_ext = Qwen3TTSTokenizerSingleCodebookMelSpectrogramFeatures()

    def extract_code(self, audio):
        import copy

        import torchaudio.compliance.kaldi as kaldi

        with torch.no_grad():
            norm_audio = self._sox_norm(audio)
            norm_audio_tensor = torch.from_numpy(copy.deepcopy(norm_audio)).unsqueeze(0)
            feat = kaldi.fbank(norm_audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            norm_embedding = self.ort_session.run(
                None, {self.ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
            )[0].flatten()
            norm_embedding = F.normalize(torch.from_numpy(norm_embedding), dim=0)
            ref_mel = self.mel_ext.extract(audio=norm_audio_tensor)
        return norm_embedding.numpy(), ref_mel.permute(0, 2, 1).squeeze(0).numpy()

    def _sox_norm(self, audio):
        return self.tfm.build_array(input_array=audio, sample_rate_in=16000)


# ── SingleCodebook Encoder PreTrainedModel + Encoder ────────────────────────


@dataclass
@auto_docstring
class Qwen3TTSTokenizerSingleCodebookEncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discrete code embeddings computed using `model.encode`, each tensor has shape `(codes_length_i,)`.
    xvectors (`List[torch.FloatTensor]`):
        X-vector speaker embeddings, each tensor has shape `(xvector_dim,)`.
    ref_mels (`List[torch.FloatTensor]`):
        Reference mel spectrogram, each tensor has shape `(mel_length_i, mel_dim)`.
    """

    audio_codes: list[torch.LongTensor] = None
    xvectors: list[torch.FloatTensor] = None
    ref_mels: list[torch.FloatTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerSingleCodebookDecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio waveforms, each tensor has shape `(segment_length_i,)`.
    """

    audio_values: list[torch.FloatTensor] = None


class Qwen3TTSTokenizerSingleCodebookEncoderPreTrainedModel(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookEncoderConfig
    _can_compile_fullgraph = False


class Qwen3TTSTokenizerSingleCodebookEncoder(Qwen3TTSTokenizerSingleCodebookEncoderPreTrainedModel):
    """Whisper-based VQ encoder that converts waveforms to discrete audio codes."""

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)
        self.tokenizer = Qwen3TTSTokenizerSingleCodebookWhisperEncoderVQ(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_window=config.n_window,
            output_dim=config.output_dim,
            grad_checkpointing=config.grad_checkpointing,
            enable_mp=config.enable_mp,
            audio_sequence_parallel=config.audio_sequence_parallel,
            audio_vq_type=config.audio_vq_type,
            audio_vq_layers=config.audio_vq_layers,
            audio_vq_codebook_size=config.audio_vq_codebook_size,
            audio_vq_codebook_dim=config.audio_vq_codebook_dim,
            audio_vq_pe=config.audio_vq_pe,
            audio_vq_ds_rate=config.audio_vq_ds_rate,
        )
        self.padding = True
        self.audio_vq_ds_rate = self.tokenizer.audio_vq_ds_rate
        self.post_init()

    def speech2mel(self, speechs):
        return [
            _v1_get_mel_audio(speech, padding=self.padding, audio_vq_ds_rate=self.audio_vq_ds_rate)
            .to(speech.dtype)
            .to(self.tokenizer.conv1.weight.device)
            for speech in speechs
        ]

    def mel2code(self, mels):
        audio_mellens = [mel.size(-1) for mel in mels]
        audio_aftercnnlens = [_v1_get_T_after_cnn(T) for T in audio_mellens]
        audio_seqlens = [T + 2 for T in audio_aftercnnlens]
        with torch.no_grad():
            _, indices = self.tokenizer(
                x_list=mels,
                audio_mellens=audio_mellens,
                audio_aftercnnlens=audio_aftercnnlens,
                audio_seqlens=audio_seqlens,
                return_indices=True,
            )
        indice_lens = [T // self.tokenizer.audio_vq_ds_rate for T in audio_aftercnnlens]
        indices = pad_sequence(torch.split(indices, indice_lens), batch_first=True, padding_value=0)
        return indices, indice_lens

    def quantize_speech(self, speechs):
        mels = self.speech2mel(speechs)
        return self.mel2code(mels)


# ── SingleCodebook Top-level model ──────────────────────────────────────────


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerSingleCodebook model combining a Whisper-based VQ encoder and a DiT-based decoder.
    """
)
class Qwen3TTSTokenizerSingleCodebookModel(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookConfig):
        super().__init__(config)
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerSingleCodebookEncoder._from_config(self.config.encoder_config)
        self.decoder = Qwen3TTSTokenizerSingleCodebookDecoder._from_config(self.config.decoder_config)
        self.encoder_xvector_extractor = None

        self.post_init()

    def load_encoder_xvector_extractor(self, model_path):
        self.encoder_xvector_extractor = Qwen3TTSTokenizerSingleCodebookXVectorExtractor(model_path)

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        try:
            extractor_path = cached_file(pretrained_model_name_or_path, "campplus.onnx")
            if extractor_path is not None:
                model.load_encoder_xvector_extractor(extractor_path)
        except Exception:
            logger.warning_once(
                "Could not load campplus.onnx for Qwen3TTSTokenizerSingleCodebookXVectorExtractor. "
                "Call model.load_encoder_xvector_extractor(path) manually before calling encode()."
            )
        return model

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | Qwen3TTSTokenizerSingleCodebookEncoderOutput:
        """
        Encodes input audio waveforms into discrete codes, x-vectors, and reference mel spectrograms.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Binary mask where 1 = valid, 0 = padding.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        wavs = [value[: mask.sum()] for value, mask in zip(input_values, padding_mask)]
        codes, codes_lens = self.encoder.quantize_speech(wavs)
        codes = [c[:length] for c, length in zip(codes, codes_lens)]

        xvectors = []
        ref_mels = []
        for wav in wavs:
            xvector, ref_mel = self.encoder_xvector_extractor.extract_code(wav.cpu().numpy())
            xvector = torch.tensor(xvector).to(wav.dtype).to(wav.device)
            ref_mel = torch.tensor(ref_mel).to(wav.dtype).to(wav.device)
            xvectors.append(xvector)
            ref_mels.append(ref_mel)

        if not return_dict:
            return (codes, xvectors, ref_mels)
        return Qwen3TTSTokenizerSingleCodebookEncoderOutput(audio_codes=codes, xvectors=xvectors, ref_mels=ref_mels)

    def decode(
        self,
        audio_codes: torch.Tensor,
        xvectors: torch.Tensor,
        ref_mels: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple | Qwen3TTSTokenizerSingleCodebookDecoderOutput:
        """
        Decodes discrete codes + speaker conditioning into an audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length)`):
                Discrete code embeddings from `model.encode`.
            xvectors (`torch.FloatTensor` of shape `(batch_size, xvector_dim)`):
                X-vector speaker embeddings from `model.encode`.
            ref_mels (`torch.FloatTensor` of shape `(batch_size, mel_length, mel_dim)`):
                Reference mel spectrogram from `model.encode`.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        audio_values = self.decoder(code=audio_codes, reference_mel=ref_mels, conditioning=xvectors)
        audio_lengths = (audio_codes > 0).sum(1) * self.decode_upsample_rate
        audio_values = [a[:length] for a, length in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)
        return Qwen3TTSTokenizerSingleCodebookDecoderOutput(audio_values=audio_values)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class Qwen3TTSTokenizerSingleCodebookAMPBlock(AMPBlock):
    """AMPBlock with CausalConv1d support for Qwen3TTS."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        causal_type="1",
    ):
        nn.Module.__init__(self)

        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[0]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[1]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[2]),
            ]
        )

        if causal_type == "1":
            self.convs2 = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                ]
            )
        else:
            self.convs2 = nn.ModuleList(
                [
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                ]
            )

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

        if causal_type == "2":
            self.pre_conv = nn.Conv1d(
                channels, channels, kernel_size, stride=1, padding=self._get_padding(kernel_size, 1)
            )

            self.pre_act = TorchActivation1d(activation=SnakeBeta(channels))
        else:
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()


class Qwen3TTSTokenizerSingleCodebookDecoderPreTrainedModel(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookDecoderConfig
    _can_compile_fullgraph = False


class Qwen3TTSTokenizerSingleCodebookDecoderDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerSingleCodebookDecoderDiTModel(Qwen2_5OmniToken2WavDiTModel):
    config: Qwen3TTSTokenizerSingleCodebookDiTConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDiTConfig
    _no_split_modules = ["Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDiTConfig):
        super().__init__(config)
        # Uses a simpler rotary embedding that takes only x (no position_ids)
        self.rotary_embed = Qwen3TTSTokenizerSingleCodebookDecoderDiTRotaryEmbedding(config.head_dim)
        # Uses the SingleCodebook DiT decoder layer alias
        self.transformer_blocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
        **kwargs,
    ):
        # batch_size accounts for CFG doubling that happens inside input_embed
        batch_size = hidden_states.shape[0] * 2
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        # rotary_embed takes only hidden_states (no separate position_ids)
        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)
        return output

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        # Single Codebook: pre-allocate large buffer then slice to needed duration
        noise_initialization = torch.randn(
            [quantized_code.shape[0], 30000, self.mel_dim], dtype=reference_mel_spectrogram.dtype
        )
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(quantized_code.device)
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                return self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        time_embedding = torch.linspace(0, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype)
        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        # Single Codebook: Euler ODE solver (parent uses RK4)
        values = initial_state.clone()
        for t0, t1 in zip(time_embedding[:-1], time_embedding[1:]):
            dt = t1 - t0
            vt = ode_function(t0, values)
            values = values + vt * dt

        return values.permute(0, 2, 1)


class Qwen3TTSTokenizerSingleCodebookDecoder(Qwen2_5OmniToken2WavModel):
    config: Qwen3TTSTokenizerSingleCodebookDecoderConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDecoderConfig
    _no_split_modules = [
        "Qwen3TTSTokenizerSingleCodebookDecoderDiTModel",
        "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel",
    ]

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDecoderConfig):
        # Skip parent's __init__ to use SingleCodebook attribute names (self.dit / self.bigvgan)
        # that match the original checkpoint state dict keys
        PreTrainedModel.__init__(self, config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen3TTSTokenizerSingleCodebookDecoder must inference with fp32, but flash_attention_2 only supports "
                "fp16 and bf16, attention implementation of Qwen3TTSTokenizerSingleCodebookDecoder will fallback to "
                "sdpa."  # noqa: E501
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen3TTSTokenizerSingleCodebookDecoder does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.dit = Qwen3TTSTokenizerSingleCodebookDecoderDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.bigvgan = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )
        self.post_init()

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generates a waveform from input code and conditioning parameters."""
        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        waveform = self.bigvgan(mel_spectrogram)
        return waveform


__all__ = [
    "Qwen3TTSTokenizerSingleCodebookPreTrainedModel",
    "Qwen3TTSTokenizerSingleCodebookDecoderPreTrainedModel",
    "Qwen3TTSTokenizerSingleCodebookDecoderDiTModel",
    "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel",
    "Qwen3TTSTokenizerSingleCodebookDecoder",
    "Qwen3TTSTokenizerSingleCodebookEncoderPreTrainedModel",
    "Qwen3TTSTokenizerSingleCodebookEncoder",
    "Qwen3TTSTokenizerSingleCodebookModel",
]
