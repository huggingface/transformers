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

import torch
from torch.nn.utils.rnn import pad_sequence

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from ..llama.modeling_llama import LlamaAttention
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
    """Make mask tensor containing indices of non-padded part."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention."""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e10
    return mask


def padding(data: list[torch.Tensor]):
    """Padding the data into batch data"""
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample], dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


def merge_tokenized_segments(tokenized_segments, overlap, token_rate):
    """Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens."""
    merged_tokens = []
    overlap_tokens = (overlap // 2) * token_rate

    for i, tokens in enumerate(tokenized_segments):
        l = 0 if i == 0 else overlap_tokens
        r = -overlap_tokens if i != len(tokenized_segments) - 1 else len(tokens)
        merged_tokens.extend(tokens[l:r])

    return merged_tokens


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


class FSMNMultiHeadAttention(LlamaAttention):
    """FSMN (Feed-forward Sequential Memory Network) Multi-Head Attention."""

    def __init__(self, config: S3TokenizerConfig, layer_idx: Optional[int] = None, kernel_size: int = 31):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.n_head = config.num_attention_heads
        self.attention_bias = config.attention_bias
        self.attention_dropout = config.attention_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_scaling = config.rope_scaling
        self.n_audio_head = config.n_audio_head
        self.n_audio_state = config.n_audio_state
        self.num_key_value_heads = config.num_key_value_heads

        self.fsmn_block = torch.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)

        # Re-initialize to match Chatterbox biases and use standard nn.Linear
        self.query = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # Remove the old names created by LlamaAttention from _modules
        del self._modules["q_proj"]
        del self._modules["k_proj"]
        del self._modules["v_proj"]
        del self._modules["o_proj"]

    def forward_fsmn(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _ = inputs.size()
        if mask is not None and mask.size(2) > 0:
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        if mask is not None:
            x = x * mask
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate fsm_memory BEFORE permuting v
        fsm_memory = self.forward_fsmn(v, mask_pad)

        # Exact baseline logic from here
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not getattr(self.config, "use_sdpa", False):
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            return self.out(wv) + fsm_memory, qk.detach()
        else:
            k = k.permute(0, 2, 1, 3) * scale
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
            output = output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            return self.out(output) + fsm_memory, None


class FSQCodebook(torch.nn.Module):
    """Finite Scalar Quantization codebook."""

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
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
        quantize = quantize.transpose(1, 2)
        return quantize


class ResidualAttentionBlock(torch.nn.Module):
    """Residual attention block with FSMN."""

    def __init__(self, config: S3TokenizerConfig, layer_idx: Optional[int] = None, kernel_size: int = 31):
        super().__init__()
        self.attn = FSMNMultiHeadAttention(config, layer_idx, kernel_size)
        self.attn_ln = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        n_mlp = config.hidden_size * 4

        # Using numeric keys to match state_dict
        self.mlp = torch.nn.ModuleList(
            [torch.nn.Linear(config.hidden_size, n_mlp), torch.nn.GELU(), torch.nn.Linear(n_mlp, config.hidden_size)]
        )
        self.mlp_ln = torch.nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        # LN Stability: cast to float32 for computation
        attn_input = self.attn_ln(x.float()).to(x.dtype)
        attn_out, _ = self.attn(
            attn_input,
            mask=mask,
            mask_pad=mask_pad,
            freqs_cis=freqs_cis,
        )
        x = x + attn_out

        # LN Stability: cast to float32 for computation
        mlp_input = self.mlp_ln(x.float()).to(x.dtype)
        mlp_out = mlp_input
        for layer in self.mlp:
            mlp_out = layer(mlp_out)

        x = x + mlp_out
        return x


class AudioEncoderV2(torch.nn.Module):
    """Audio encoder for S3TokenizerV2."""

    def __init__(self, config: S3TokenizerConfig):
        super().__init__()
        self.stride = 2  # Hardcoded for V2
        self.conv1 = torch.nn.Conv1d(config.n_mels, config.hidden_size, kernel_size=3, stride=self.stride, padding=1)
        self.conv2 = torch.nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)

        self.register_buffer("freqs_cis", precompute_freqs_cis(64, 1024 * 2), persistent=False)

        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(config, layer_idx=i) for i in range(config.n_audio_layer)]
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

        mask_pad = mask.transpose(1, 2)
        mask_bias = mask_to_bias(mask, x.dtype)
        freqs_cis = self.freqs_cis.to(x.device)

        for block in self.blocks:
            x = block(x, mask=mask_bias.unsqueeze(1), mask_pad=mask_pad, freqs_cis=freqs_cis[: x.size(1)])

        return x, x_len


class S3TokenizerV2Core(torch.nn.Module):
    """Core S3 tokenizer v2 implementation."""

    def __init__(self, config: S3TokenizerConfig):
        super().__init__()
        self.encoder = AudioEncoderV2(config)
        self.quantizer = FSQVectorQuantization(config.hidden_size, config.vocab_size)

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
    main_input_name = "input_features"

    def _init_weights(self, module):
        """Initialize weights.

        S3Tokenizer models are expected to be loaded from pretrained checkpoints, but we still need to correctly
        (re)initialize buffers when the model is created on meta device then materialized with `to_empty()`.
        """
        # These buffers are registered in `__init__` to match checkpoint keys. When a model is initialized on meta
        # device, then materialized via `to_empty()`, buffers may contain uninitialized values and need to be restored
        # deterministically here.
        if isinstance(module, S3TokenizerModel):
            # During `from_pretrained`, core loading will set `_is_hf_initialized=True` on loaded tensors.
            # Do not overwrite buffers that were loaded from the checkpoint.
            if not getattr(module.window, "_is_hf_initialized", False):
                module.window = torch.zeros(
                    module.config.n_fft, device=module.window.device, dtype=module.window.dtype
                )
            if not getattr(module._mel_filters, "_is_hf_initialized", False):
                module._mel_filters = torch.zeros(
                    module.config.n_mels,
                    module.config.n_fft // 2 + 1,
                    device=module._mel_filters.device,
                    dtype=module._mel_filters.dtype,
                )
        elif isinstance(module, AudioEncoderV2):
            if not getattr(module.freqs_cis, "_is_hf_initialized", False):
                module.freqs_cis = precompute_freqs_cis(64, 1024 * 2).to(device=module.freqs_cis.device)


class S3TokenizerModel(S3TokenizerPreTrainedModel):
    """
    S3Tokenizer model for speech tokenization.

    This model integrates the S3Tokenizer implementation from xingchensong/S3Tokenizer
    repository into HuggingFace Transformers.

    Args:
            config (`S3TokenizerConfig`): <fill_docstring>
            name (`str`, *optional*, defaults to `"speech_tokenizer_v2_25hz"`): <fill_docstring>
    """

    all_tied_weights_keys = {}

    def __init__(self, config: S3TokenizerConfig, name: str = "speech_tokenizer_v2_25hz"):
        super().__init__(config)
        self.config = config

        self.s3_model = S3TokenizerV2Core(config)

        # Register buffers for STFT to match checkpoint keys
        self.register_buffer("window", torch.zeros(config.n_fft))
        self.register_buffer("_mel_filters", torch.zeros(config.n_mels, config.n_fft // 2 + 1))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, S3TokenizerOutput]:
        """
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, n_mels)`):
                Float values of log-mel spectrogram features.
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

        if max_len is not None:
            input_features = input_features[..., : max_len * 4]

        # Get mel lengths from attention_mask or input_features shape
        if attention_mask is not None:
            mel_lens = attention_mask.sum(dim=-1).int()
        else:
            mel_lens = torch.full(
                (input_features.size(0),), input_features.size(1), device=input_features.device, dtype=torch.int32
            )

        # Transpose from [batch, time, n_mels] to [batch, n_mels, time] for conv layers
        input_features = input_features.transpose(1, 2)

        speech_tokens, speech_token_lens = self.s3_model.quantize(input_features, mel_lens)
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
