# coding=utf-8
# Copyright 2024 Resemble AI, xingchensong and The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from:
#   - Chatterbox S3Tokenizer implementation
#   - xingchensong/S3Tokenizer repository: https://github.com/xingchensong/S3Tokenizer
#   - Original Whisper model: https://github.com/openai/whisper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch S3Tokenizer model - self-contained implementation."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_s3tokenizer import S3TokenizerConfig


logger = logging.get_logger(__name__)


# Sampling rate and frame configuration for S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561

# Special tokens
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.

    Parameters
    ----------
        mask (torch.Tensor): Boolean mask tensor (B, ?).

    Returns:
    -------
        torch.Tensor: Mask tensor with large negative values for masked positions (B, ?).
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)

    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e10
    return mask


def padding(data: list[torch.Tensor]):
    """Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats [B, 128, T_max], feats lengths [B]
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample], dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


def merge_tokenized_segments(tokenized_segments, overlap, token_rate):
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
    - tokenized_segments (List[List[int]]): List of tokenized sequences.
    - overlap (int): Overlapping duration in seconds (default: 4s).
    - token_rate (int): Number of tokens per second.

    Returns:
    - List[int]: A single merged token sequence.
    """
    merged_tokens = []
    overlap_tokens = (overlap // 2) * token_rate  # Tokens corresponding to half of the overlap duration

    for i, tokens in enumerate(tokenized_segments):
        l = 0 if i == 0 else overlap_tokens
        r = -overlap_tokens if i != len(tokenized_segments) - 1 else len(tokens)
        # Keep only the middle part (drop overlap / 2 from both sides)
        merged_tokens.extend(tokens[l:r])

    return merged_tokens


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization that preserves dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(torch.nn.Linear):
    """Linear layer that preserves dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(torch.nn.Conv1d):
    """Conv1d layer that preserves dtype."""

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention module."""

    def __init__(self, n_state: int, n_head: int, use_sdpa: bool = False):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.use_sdpa = use_sdpa

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
            output = output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            return output, None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling=None):
    """Precompute frequencies for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, : D // 2], xq[:, :, :, D // 2 :]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]
    half_l, half_r = xk[:, :, :, : D // 2], xk[:, :, :, D // 2 :]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


class FSQCodebook(torch.nn.Module):
    """Finite Scalar Quantization codebook."""

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions except last: equivalent to rearrange(x, "... d -> (...) d")
        x = x.view(-1, x.shape[-1])
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = self.preprocess(x)
        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        powers = torch.pow(self.level, torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1]).int()
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("There is no official up project component provided")


class FSQVectorQuantization(torch.nn.Module):
    """FSQ Vector quantization implementation (inference-only)."""

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        # Transpose dimensions: equivalent to rearrange(quantize, "b n d -> b d n")
        quantize = quantize.transpose(1, 2)
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):
    """FSMN (Feed-forward Sequential Memory Network) Multi-Head Attention."""

    def __init__(self, n_state: int, n_head: int, kernel_size: int = 31, use_sdpa: bool = False):
        super().__init__(n_state, n_head)

        self.fsmn_block = torch.nn.Conv1d(
            n_state,
            n_state,
            kernel_size,
            stride=1,
            padding=0,
            groups=n_state,
            bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)
        self.use_sdpa = use_sdpa

    def forward_fsmn(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (
                (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2),
                qk.detach(),
                fsm_memory,
            )
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
            output = output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            return output, None, fsm_memory

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(torch.nn.Module):
    """Residual attention block with FSMN."""

    def __init__(self, n_state: int, n_head: int, kernel_size: int = 31, use_sdpa: bool = False):
        super().__init__()
        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size, use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state, eps=1e-6)
        n_mlp = n_state * 4
        self.mlp = torch.nn.Sequential(Linear(n_state, n_mlp), torch.nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(torch.nn.Module):
    """Audio encoder for S3TokenizerV2."""

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("freqs_cis", precompute_freqs_cis(64, 1024 * 2), persistent=False)
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)
        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[: x.size(1)])

        return x, x_len


class S3TokenizerV2Core(torch.nn.Module):
    """Core S3 tokenizer v2 implementation."""

    def __init__(
        self,
        name: str,
        n_mels: int,
        n_audio_state: int,
        n_audio_head: int,
        n_audio_layer: int,
        n_codebook_size: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.name = name
        self.encoder = AudioEncoderV2(n_mels, n_audio_state, n_audio_head, n_audio_layer, 2, use_sdpa)
        self.quantizer = FSQVectorQuantization(n_audio_state, n_codebook_size)

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize mel spectrogram to tokens, with automatic long audio handling."""
        max_frames = 3000
        long_audio_mask = mel_len > max_frames

        if long_audio_mask.any():
            return self._quantize_mixed_batch(mel, mel_len, long_audio_mask, max_frames)
        else:
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return code, code_len

    @torch.inference_mode()
    def _quantize_mixed_batch(
        self,
        mel: torch.Tensor,
        mel_len: torch.Tensor,
        long_audio_mask: torch.Tensor,
        max_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle mixed batch with both short and long audio using unified batch processing."""
        batch_size = mel.size(0)
        sample_rate, hop_length = 16000, 160
        window_size, overlap = 30, 4
        frames_per_window = window_size * sample_rate // hop_length
        frames_per_overlap = overlap * sample_rate // hop_length
        frames_per_stride = frames_per_window - frames_per_overlap

        all_segments, all_segments_len, segment_info = [], [], []

        for batch_idx in range(batch_size):
            audio_mel = mel[batch_idx]
            audio_mel_len = mel_len[batch_idx]
            is_long_audio = long_audio_mask[batch_idx].item()

            if not is_long_audio:
                segment = audio_mel[:, :audio_mel_len]
                seg_len = audio_mel_len.item()
                if seg_len < frames_per_window:
                    segment = torch.nn.functional.pad(segment, (0, frames_per_window - seg_len))
                all_segments.append(segment)
                all_segments_len.append(torch.tensor(seg_len, device=mel.device))
                segment_info.append(
                    {
                        "batch_idx": batch_idx,
                        "is_long_audio": False,
                        "segment_idx": 0,
                        "total_segments": 1,
                    }
                )
            else:
                start, segment_idx = 0, 0
                while start < audio_mel_len:
                    end = min(start + frames_per_window, audio_mel_len)
                    segment = audio_mel[:, start:end]
                    seg_len = segment.size(1)
                    if seg_len < frames_per_window:
                        segment = torch.nn.functional.pad(segment, (0, frames_per_window - seg_len))
                    all_segments.append(segment)
                    all_segments_len.append(torch.tensor(seg_len, device=mel.device))
                    segment_info.append(
                        {
                            "batch_idx": batch_idx,
                            "is_long_audio": True,
                            "segment_idx": segment_idx,
                            "total_segments": None,
                        }
                    )
                    segment_idx += 1
                    start += frames_per_stride

                for info in segment_info:
                    if info["batch_idx"] == batch_idx and info["is_long_audio"]:
                        info["total_segments"] = segment_idx

        if not all_segments:
            return torch.zeros(batch_size, 0, dtype=torch.long, device=mel.device), torch.zeros(
                batch_size, dtype=torch.long, device=mel.device
            )

        unified_batch_mel = torch.stack(all_segments)
        unified_batch_lens = torch.stack(all_segments_len)
        hidden, code_len = self.encoder(unified_batch_mel, unified_batch_lens)
        codes = self.quantizer.encode(hidden)

        results = {}
        for seg_idx, info in enumerate(segment_info):
            batch_idx = info["batch_idx"]
            segment_code = codes[seg_idx, : code_len[seg_idx].item()].cpu().numpy().tolist()
            if not info["is_long_audio"]:
                code_tensor = torch.tensor(segment_code, dtype=torch.long, device=mel.device)
                results[batch_idx] = (code_tensor, len(segment_code))
            else:
                if batch_idx not in results:
                    results[batch_idx] = []
                results[batch_idx].append(segment_code)

        for batch_idx in range(batch_size):
            if long_audio_mask[batch_idx].item():
                audio_codes = results[batch_idx]
                merged_codes = merge_tokenized_segments(audio_codes, overlap=overlap, token_rate=25)
                merged_codes_tensor = torch.tensor(merged_codes, dtype=torch.long, device=mel.device)
                results[batch_idx] = (merged_codes_tensor, len(merged_codes))

        max_code_len = max(code_info[1] for code_info in results.values())
        output_codes = torch.zeros(batch_size, max_code_len, dtype=torch.long, device=mel.device)
        output_codes_len = torch.zeros(batch_size, dtype=torch.long, device=mel.device)

        for batch_idx, (code_tensor, code_len) in results.items():
            output_codes[batch_idx, :code_len] = code_tensor
            output_codes_len[batch_idx] = code_len

        return output_codes, output_codes_len

    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
@auto_docstring
class S3TokenizerOutput(ModelOutput):
    r"""
    speech_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Discrete speech tokens computed using `model.quantize`.
    speech_token_lens (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Length of each speech token sequence.
    """

    speech_tokens: Optional[torch.LongTensor] = None
    speech_token_lens: Optional[torch.LongTensor] = None


class S3TokenizerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = S3TokenizerConfig
    base_model_prefix = "s3tokenizer"
    main_input_name = "input_values"


class S3TokenizerModel(S3TokenizerPreTrainedModel):
    """
    S3Tokenizer model for speech tokenization.

    This model integrates the S3Tokenizer implementation from xingchensong/S3Tokenizer
    repository into HuggingFace Transformers.

    Args:
            config (`S3TokenizerConfig`): <fill_docstring>
            name (`str`, *optional*, defaults to `"speech_tokenizer_v2_25hz"`): <fill_docstring>
    """

    ignore_state_dict_missing = ("_mel_filters",)
    all_tied_weights_keys = {}

    def __init__(self, config: S3TokenizerConfig, name: str = "speech_tokenizer_v2_25hz"):
        super().__init__(config)
        self.config = config

        # Init core S3TokenizerV2 model
        # code adapted from xingchensong/S3Tokenizer
        self.s3_model = S3TokenizerV2Core(
            name=name,
            n_mels=config.n_mels,
            n_audio_state=config.n_audio_state,
            n_audio_head=config.n_audio_head,
            n_audio_layer=config.n_audio_layer,
            n_codebook_size=config.vocab_size,
            use_sdpa=config.use_sdpa,
        )

        self.n_fft = config.n_fft
        try:
            import librosa

            _mel_filters = librosa.filters.mel(sr=config.sampling_rate, n_fft=self.n_fft, n_mels=config.n_mels)
            self.register_buffer("_mel_filters", torch.FloatTensor(_mel_filters))
        except ImportError:
            logger.warning(
                "librosa is not installed. Mel filters will not be initialized. "
                "Install librosa with: pip install librosa"
            )
            self.register_buffer("_mel_filters", torch.zeros(config.n_mels, self.n_fft // 2 + 1))
        # self.window = torch.hann_window(self.n_fft)
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def pad(self, wavs: list[Union[torch.Tensor, np.ndarray]], sr: int) -> list[torch.Tensor]:
        """Pad waveforms to be multiple of 40ms (S3 runs at 25 token/sec)."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = int(n_tokens * (sr / S3_TOKEN_RATE))
            wav = torch.nn.functional.pad(wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs: list[Union[torch.Tensor, np.ndarray]]) -> list[torch.Tensor]:
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            processed_wavs.append(wav)
        return processed_wavs

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0) -> torch.Tensor:
        """Compute the log-Mel spectrogram of audio."""
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window.to(self.device),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters.to(self.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if squeeze_output:
            log_spec = log_spec.squeeze(0)

        return log_spec

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, S3TokenizerOutput]:
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform at 16kHz sampling rate.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing operations on padding token indices.
            max_len (`int`, *optional*):
                Maximum length to truncate the output sequence to (25 token/sec).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `S3TokenizerOutput` or `tuple`: Speech tokens and their lengths.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_values, list):
            wavs = input_values
        elif input_values.dim() == 1:
            # Single waveform
            wavs = [input_values]
        else:
            # Batch Mode
            wavs = [input_values[i] for i in range(input_values.shape[0])]

        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []

        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav.squeeze(0))
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if max_len is not None:
                mel = mel[..., : max_len * 4]
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        mels = mels.to(self.device)
        mel_lens = mel_lens.to(self.device)

        speech_tokens, speech_token_lens = self.s3_model.quantize(mels, mel_lens)
        speech_tokens = speech_tokens.long().detach()
        speech_token_lens = speech_token_lens.long().detach()

        if not return_dict:
            return (speech_tokens, speech_token_lens)

        return S3TokenizerOutput(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
        )

    def get_input_embeddings(self):
        """S3Tokenizer does not use input embeddings in the traditional sense."""
        return None

    @property
    def device(self):
        return next(self.parameters()).device


def drop_invalid_tokens(x: torch.Tensor) -> torch.Tensor:
    """Drop SoS and EoS tokens from speech token sequence."""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"

    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0

    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None

    x = x[s:e]
    return x


__all__ = [
    "S3TokenizerModel",
    "S3TokenizerPreTrainedModel",
    "S3TokenizerOutput",
    "drop_invalid_tokens",
]
