# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch T3 model."""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided
from torch import Tensor

from ...generation.utils import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ..llama.modeling_llama import LlamaConfig, LlamaModel, LlamaPreTrainedModel
from .configuration_t3 import T3Config

logger = logging.getLogger(__name__)

T3_PRETRAINED_MODEL_ARCHIVE_LIST = []


# ============================================================================
# Voice Encoder Components
# ============================================================================


class VoiceEncConfig:
    """Configuration for Voice Encoder."""

    def __init__(self):
        self.sample_rate = 16000
        self.num_mels = 40
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.fmin = 0
        self.fmax = 8000
        self.ve_partial_frames = 160
        self.ve_hidden_size = 256
        self.speaker_embed_size = 256
        self.normalized_mels = True
        self.ve_final_relu = False
        self.flatten_lstm_params = False


def melspectrogram_voice_encoder(wav, config: VoiceEncConfig):
    """Extract mel spectrogram for voice encoder."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.num_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm


def stride_as_partials(mel: np.ndarray, hp: VoiceEncConfig, overlap=0.5, rate: float = None, min_coverage=0.8):
    """Stride mel spectrogram into overlapping partials."""

    def get_frame_step(overlap, rate, hp):
        assert 0 <= overlap < 1
        if rate is None:
            frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
        else:
            frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
        assert 0 < frame_step <= hp.ve_partial_frames
        return frame_step

    def get_num_wins(n_frames, step, min_coverage, hp):
        assert n_frames > 0
        win_size = hp.ve_partial_frames
        n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
        if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
            n_wins += 1
        target_n = win_size + step * (n_wins - 1)
        return n_wins, target_n

    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Trim or pad
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    mel = mel.astype(np.float32, order="C")
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


class VoiceEncoder(nn.Module):
    """Voice encoder for speaker embedding extraction."""

    def __init__(self, config: VoiceEncConfig = None):
        super().__init__()
        self.config = config if config is not None else VoiceEncConfig()

        self.lstm = nn.LSTM(self.config.num_mels, self.config.ve_hidden_size, num_layers=3, batch_first=True)
        self.proj = nn.Linear(self.config.ve_hidden_size, self.config.speaker_embed_size)

        # Cosine similarity scaling
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]), requires_grad=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor):
        """Compute embeddings from mel spectrograms."""
        _, (hidden, _) = self.lstm(mels)
        raw_embeds = self.proj(hidden[-1])
        if self.config.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def embeds_from_wavs(
        self, wavs: List[np.ndarray], sample_rate: int, overlap=0.5, rate: float = 1.3, batch_size=32
    ):
        """Extract embeddings from waveforms."""
        if sample_rate != self.config.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.config.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]

        wavs = [librosa.effects.trim(wav, top_db=20)[0] for wav in wavs]

        # Extract mel spectrograms
        mels = [melspectrogram_voice_encoder(w, self.config).T for w in wavs]

        # Stride into partials
        all_partials = []
        n_partials_per_wav = []
        for mel in mels:
            partials = stride_as_partials(mel, self.config, overlap=overlap, rate=rate)
            all_partials.append(torch.from_numpy(partials))
            n_partials_per_wav.append(len(partials))

        # Stack and process
        all_partials = torch.cat(all_partials, dim=0).to(self.device)

        # Forward in batches
        n_chunks = int(np.ceil(len(all_partials) / batch_size))
        partial_embeds = []
        for chunk in all_partials.chunk(n_chunks):
            with torch.inference_mode():
                partial_embeds.append(self(chunk))
        partial_embeds = torch.cat(partial_embeds, dim=0).cpu()

        # Aggregate partials per wav
        slices = np.concatenate(([0], np.cumsum(n_partials_per_wav)))
        embeds = []
        for start, end in zip(slices[:-1], slices[1:]):
            raw_embed = torch.mean(partial_embeds[start:end], dim=0)
            embeds.append(raw_embed / torch.linalg.norm(raw_embed))

        return torch.stack(embeds).numpy()


# ============================================================================
# Learned Position Embeddings
# ============================================================================


class LearnedPositionEmbeddings(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        """Returns positional embeddings for index 0 up to the length of x."""
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, idx: Union[int, Tensor]):
        """Get embeddings for specific indices."""
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx = torch.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx)


# ============================================================================
# Perceiver Resampler
# ============================================================================


class AttentionQKV(nn.Module):
    """Attention module with separate Q, K, V projections."""

    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        if self.flash and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout_rate if self.training else 0.0
            )
        else:
            out = self.scaled_dot_product_attention(q, k, v, mask=mask)
        return self.combine_heads(out)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bhts,bhls->bhlt", attn, v)

    def split_heads(self, x):
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, length, -1)


class AttentionBlock(nn.Module):
    """Cross-attention block for perceiver."""

    def __init__(
        self, channels, num_heads=1, num_head_channels=-1, flash_attention=True, dropout_rate=0.2, scale=None
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.attention = AttentionQKV(
            self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x1, x2, mask=None):
        b1, c1, *spatial1 = x1.shape
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)
        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """Perceiver resampler for conditioning."""

    def __init__(
        self, pre_attention_query_token=32, pre_attention_query_size=1024, embedding_dim=1024, num_attn_heads=4
    ):
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, pre_attention_query_token, pre_attention_query_size))
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)
        self.attn = AttentionBlock(embedding_dim, num_attn_heads)

    def forward(self, h):
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)
        pre_att = self.attn(query_, h)
        attn = self.attn(pre_att, pre_att)
        return attn


# ============================================================================
# T3 Conditioning
# ============================================================================


@dataclass
class T3Cond:
    """Dataclass container for T3 conditioning information."""

    speaker_emb: Tensor
    clap_emb: Optional[Tensor] = None
    cond_prompt_speech_tokens: Optional[Tensor] = None
    cond_prompt_speech_emb: Optional[Tensor] = None
    emotion_adv: Optional[Tensor] = None

    def to(self, *, device=None, dtype=None):
        """Cast to a device and dtype."""
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = not v.dtype in [torch.long, torch.int, torch.int32, torch.int64]
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self


class T3CondEnc(nn.Module):
    """Encoder for T3 conditioning (speaker, emotion, prompts)."""

    def __init__(self, config: T3Config):
        super().__init__()
        self.config = config

        if config.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(config.speaker_embed_size, config.hidden_size)
        else:
            raise NotImplementedError(str(config.encoder_type))

        self.emotion_adv_fc = None
        if config.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, config.hidden_size, bias=False)

        self.perceiver = None
        if config.use_perceiver_resampler:
            self.perceiver = Perceiver(
                pre_attention_query_token=config.perceiver_num_latents,
                pre_attention_query_size=config.perceiver_latent_dim,
                embedding_dim=config.hidden_size,
                num_attn_heads=config.perceiver_num_heads,
            )

    def forward(self, cond: T3Cond):
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), (
            "no embeddings for cond_prompt_speech_tokens"
        )

        # Speaker embedding projection
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.config.speaker_embed_size))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])

        # CLAP (not implemented)
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty

        # Conditioning prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty
        elif self.config.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion
        cond_emotion_adv = empty
        if self.config.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))

        # Concatenate
        cond_embeds = torch.cat((cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim=1)
        return cond_embeds


# ============================================================================
# Alignment Stream Analyzer (for multilingual)
# ============================================================================

LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


class AlignmentStreamAnalyzer:
    """Alignment analyzer for detecting hallucinations in multilingual models."""

    def __init__(self, tfmr, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []
        self.last_aligned_attns = []

        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """Add forward hook to collect attention weights."""

        def attention_forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

        target_layer = tfmr.layers[layer_idx].self_attn
        target_layer.register_forward_hook(attention_forward_hook)
        if hasattr(tfmr, "config") and hasattr(tfmr.config, "output_attentions"):
            tfmr.config.output_attentions = True

    def step(self, logits, next_token=None):
        """Analyze alignment and potentially modify logits."""
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            A_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            A_chunk = aligned_attn[:, i:j].clone().cpu()

        A_chunk[:, self.curr_frame_pos + 1 :] = 0
        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        long_tail = self.complete and (A[self.completed_at :, -3:].sum(dim=0).max() >= 5)
        alignment_repetition = self.complete and (A[self.completed_at :, :-5].max(dim=1).values.sum() > 5)

        # Track tokens
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-2:])) == 1

        # Suppress EOS early
        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -(2**15)

        # Force EOS on bad endings
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"Forcing EOS: {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits


class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    """
    Lightweight wrapper so we can reuse HuggingFace's generation utilities while feeding custom embeddings/logits.
    """

    def __init__(
        self,
        config,
        llama: LlamaModel,
        *,
        speech_enc: nn.Embedding,
        speech_head: nn.Linear,
        alignment_stream_analyzer: Optional[AlignmentStreamAnalyzer] = None,
    ):
        super().__init__(config)
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self._added_cond = False
        self.alignment_stream_analyzer = alignment_stream_analyzer

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        decoder_cond: torch.Tensor,
        use_cache: bool,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.Tensor] = None,  # kept for API parity
    ):
        if not use_cache:
            past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs_embeds = self.speech_enc(input_ids)

        if not self._added_cond:
            assert past_key_values is not None
            if decoder_cond.size(0) != inputs_embeds.size(0):
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([decoder_cond, inputs_embeds], dim=1)
            self._added_cond = True

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.inference_mode()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        assert return_dict
        assert output_hidden_states

        output = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = output.hidden_states[-1]
        logits = self.speech_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


# ============================================================================
# T3 Model
# ============================================================================


class T3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T3Config
    base_model_prefix = "t3"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]


class T3Model(T3PreTrainedModel):
    """
    T3 (Token-To-Token) TTS model using LLaMA as backbone.

    This model generates speech tokens from text tokens, which can then be decoded by S3Gen.
    """

    def __init__(self, config: T3Config):
        super().__init__(config)
        self.config = config

        # Create LLaMA backbone
        llama_config = LlamaConfig(**config.llama_config_dict)
        self.tfmr = LlamaModel(llama_config)
        self.dim = llama_config.hidden_size

        # Conditioning encoder
        self.cond_enc = T3CondEnc(config)

        # Text and speech embeddings
        self.text_emb = nn.Embedding(config.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(config.speech_tokens_dict_size, self.dim)

        # Positional embeddings
        if config.input_pos_emb == "learned":
            max_text_seq_len = config.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = config.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Output heads
        self.text_head = nn.Linear(self.dim, config.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, config.speech_tokens_dict_size, bias=False)

        # Voice encoder for speaker conditioning
        self.voice_encoder = VoiceEncoder()

        # Initialize weights
        self.post_init()

        self.patched_model: Optional[T3HuggingfaceBackend] = None
        self.compiled = False

    def prepare_conditioning(self, t3_cond: T3Cond):
        """Prepare conditioning embeddings."""
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + self.speech_pos_emb(
                t3_cond.cond_prompt_speech_tokens
            )
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        """Prepare input embeddings for the transformer."""
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens)
        if cfg_weight > 0.0 and text_emb.size(0) > 1:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)
        if self.config.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.stack([torch.cat((ce, te, se)) for ce, te, se in zip(cond_emb, text_emb, speech_emb)])
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        """Forward pass of T3 model."""
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_tokens
        )

        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]

        # Split hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)

        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, : ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, : stl[i]] = hidden_states[i, speech_start:speech_end]

        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return {
            "text_logits": text_logits,
            "text_latents": text_latents,
            "speech_logits": speech_logits,
            "speech_latents": speech_latents,
            "hidden_states": hidden_states,
        }

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        """Compute training loss."""
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )

        IGNORE_ID = -100
        device = out["text_logits"].device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

        loss_text = F.cross_entropy(out["text_logits"].transpose(1, 2), masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out["speech_logits"].transpose(1, 2), masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """Generate speech tokens from text tokens."""
        from transformers.generation.logits_process import (
            MinPLogitsWarper,
            RepetitionPenaltyLogitsProcessor,
            TopPLogitsWarper,
        )

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        if initial_speech_tokens is None:
            initial_speech_tokens = self.config.start_speech_token * torch.ones_like(text_tokens[:, :1])

        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight
        )

        if not self.compiled:
            alignment_stream_analyzer = None
            if self.config.use_alignment_analyzer:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=self.config.alignment_layer_idx,
                    eos_idx=self.config.stop_speech_token,
                )

            self.patched_model = T3HuggingfaceBackend(
                config=self.tfmr.config,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.compiled = True

        alignment_stream_analyzer = self.patched_model.alignment_stream_analyzer

        device = embeds.device
        bos_token = torch.tensor([[self.config.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        use_cfg = cfg_weight > 0.0 and embeds.size(0) > 1
        if use_cfg:
            bos_embed = torch.cat([bos_embed, bos_embed], dim=0)

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        max_steps = max_new_tokens or self.config.max_speech_tokens
        predicted = []

        for i in range(max_steps):
            logits_step = output.logits[:, -1, :]

            if use_cfg:
                cond = logits_step[0:1, :]
                uncond = logits_step[1:2, :]
                cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
                logits = cond + cfg * (cond - uncond)
            else:
                logits = logits_step

            if alignment_stream_analyzer is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if generated_ids.size(1) > 0 else None
                logits = alignment_stream_analyzer.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)

            if temperature != 1.0:
                logits = logits / temperature

            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.view(-1) == self.config.stop_speech_token:
                logger.info(f"EOS token detected at step {i + 1}")
                break

            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
            if use_cfg:
                next_token_embed = torch.cat([next_token_embed, next_token_embed], dim=0)

            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

        if predicted:
            predicted_tokens = torch.cat(predicted, dim=1)
        else:
            predicted_tokens = torch.empty((1, 0), dtype=torch.long, device=device)

        return predicted_tokens
