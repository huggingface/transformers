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
"""PyTorch Chatterbox model - Complete TTS Pipeline."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from ...generation.utils import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...models.s3gen.modeling_s3gen import S3GenModel
from ...models.s3tokenizer.modeling_s3tokenizer import drop_invalid_tokens
from ...utils import auto_docstring, is_librosa_available
from ..llama.modeling_llama import LlamaConfig, LlamaModel, LlamaPreTrainedModel
from .configuration_chatterbox import ChatterboxConfig
from .feature_extraction_chatterbox import (
    ChatterboxFeatureExtractor,
    VoiceEncConfig,
    melspectrogram_voice_encoder,
    stride_as_partials,
)


if is_librosa_available():
    import librosa
else:
    librosa = None


logger = logging.getLogger(__name__)


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or containing chars not seen often in the dataset.
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (
            """, '"'),
        (""",
            '"',
        ),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


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
        self, wavs: list[np.ndarray], sample_rate: int, overlap=0.5, rate: float = 1.3, batch_size=32
    ):
        """Extract embeddings from waveforms."""
        if librosa is None:
            raise ImportError(
                "librosa is required for Chatterbox voice encoder preprocessing (resampling + trimming). "
                "Please install it with `pip install librosa`."
            )
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
                is_fp = v.dtype not in [torch.long, torch.int, torch.int32, torch.int64]
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self


class T3CondEnc(nn.Module):
    """Encoder for T3 conditioning (speaker, emotion, prompts)."""

    def __init__(self, config):
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


# ============================================================================
# T3 Model
# ============================================================================


class T3PreTrainedModel(LlamaPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = None  # Will be set by ChatterboxConfig
    base_model_prefix = "t3"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]


@auto_docstring
class T3Model(T3PreTrainedModel, LlamaModel, GenerationMixin):
    """
    T3 (Token-To-Token) TTS model using LLaMA as backbone.

    This model generates speech tokens from text tokens, which can then be decoded by S3Gen.
    """

    def __init__(self, config):
        # Create LLaMA backbone config and initialize parent LlamaModel
        llama_config = LlamaConfig(**config.llama_config_dict)
        super().__init__(llama_config)

        # Store the full T3 config for T3-specific settings
        self.t3_config = config
        self.dim = llama_config.hidden_size

        # llama_config_name is stored in config for reference
        _ = config.llama_config_name

        # Conditioning encoder
        self.cond_enc = T3CondEnc(self.t3_config)

        # Text and speech embeddings
        self.text_emb = nn.Embedding(self.t3_config.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3_config.speech_tokens_dict_size, self.dim)

        # Positional embeddings
        if self.t3_config.input_pos_emb == "learned":
            max_text_seq_len = self.t3_config.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = self.t3_config.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Output heads
        self.text_head = nn.Linear(self.dim, self.t3_config.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, self.t3_config.speech_tokens_dict_size, bias=False)

        # Voice encoder for speaker conditioning
        self.voice_encoder = VoiceEncoder()

        # Set main input name for generation
        self.main_input_name = "inputs_embeds"

        # Initialize weights
        self.post_init()

        # Generation state
        self._decoder_cond = None
        self._added_cond = False
        self._current_position = 0
        self.alignment_stream_analyzer = None

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
        if self.t3_config.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.stack([torch.cat((ce, te, se)) for ce, te, se in zip(cond_emb, text_emb, speech_emb)])
        return embeds, len_cond

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Training-specific arguments
        t3_cond: Optional[T3Cond] = None,
        text_tokens: Optional[torch.LongTensor] = None,
        text_token_lens: Optional[torch.LongTensor] = None,
        speech_tokens: Optional[torch.LongTensor] = None,
        speech_token_lens: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass of T3 model.

        Supports both training mode (with t3_cond, text_tokens, speech_tokens) and
        generation mode (with inputs_embeds from prepare_inputs_for_generation).
        """
        return_dict = return_dict if return_dict is not None else self.t3_config.use_return_dict

        # Training/evaluation mode
        if t3_cond is not None and text_tokens is not None and speech_tokens is not None:
            embeds, len_cond = self.prepare_input_embeds(
                t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_tokens
            )

            tfmr_out = super().forward(
                inputs_embeds=embeds,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
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

        # Generation mode
        else:
            output_attentions = output_attentions if output_attentions is not None else False
            output_hidden_states = output_hidden_states if output_hidden_states is not None else True
            use_cache = use_cache if use_cache is not None else True

            tfmr_out = super().forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            hidden_states = tfmr_out.hidden_states[-1]
            logits = self.speech_head(hidden_states)

            if not return_dict:
                return (logits, tfmr_out.past_key_values, hidden_states, tfmr_out.attentions)

            return CausalLMOutputWithCrossAttentions(
                logits=logits,
                past_key_values=tfmr_out.past_key_values,
                hidden_states=tfmr_out.hidden_states,
                attentions=tfmr_out.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        Prepare inputs for generation step.

        This method is called by HuggingFace's generate() at each step.
        """
        # First call: add conditioning embeddings
        if past_key_values is None:
            # Initial call - input_ids is the BOS token(s)
            inputs_embeds = self.speech_emb(input_ids)
            inputs_embeds = inputs_embeds + self.speech_pos_emb.get_fixed_embedding(0)

            # Prepend decoder conditioning if available
            if self._decoder_cond is not None:
                inputs_embeds = torch.cat([self._decoder_cond, inputs_embeds], dim=1)
                self._current_position = 0
        else:
            # Subsequent calls - only process the new token
            input_ids = input_ids[:, -1:]
            inputs_embeds = self.speech_emb(input_ids)
            self._current_position += 1
            inputs_embeds = inputs_embeds + self.speech_pos_emb.get_fixed_embedding(self._current_position)

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache if use_cache is not None else True,
            "output_attentions": self.t3_config.use_alignment_analyzer
            if hasattr(self.t3_config, "use_alignment_analyzer")
            else False,
            "output_hidden_states": True,
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
            TemperatureLogitsWarper,
            TopPLogitsWarper,
        )

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        if initial_speech_tokens is None:
            initial_speech_tokens = self.t3_config.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare conditioning embeddings (text + initial speech)
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight
        )

        # Setup alignment analyzer if needed
        if self.t3_config.use_alignment_analyzer:
            alignment_analyzer = AlignmentStreamAnalyzer(
                self,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=self.t3_config.alignment_layer_idx,
                eos_idx=self.t3_config.stop_speech_token,
            )
        else:
            alignment_analyzer = None

        max_steps = max_new_tokens or self.t3_config.max_speech_tokens
        device = embeds.device
        use_cfg = cfg_weight > 0.0 and embeds.size(0) > 1

        # If using CFG, we need manual generation loop (HF generate doesn't support CFG batching)
        if use_cfg:
            # Manual generation loop for CFG
            bos_token = torch.tensor([[self.t3_config.start_speech_token]], dtype=torch.long, device=device)
            bos_embed = self.speech_emb(bos_token)
            bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
            bos_embed = torch.cat([bos_embed, bos_embed], dim=0)  # Duplicate for CFG

            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
            generated_ids = bos_token.clone()

            top_p_warper = TopPLogitsWarper(top_p=top_p)
            min_p_warper = MinPLogitsWarper(min_p=min_p)
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

            output = self(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
                output_attentions=alignment_analyzer is not None,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

            predicted = []

            for i in range(max_steps):
                logits_step = output.logits[:, -1, :]

                # Apply CFG
                cond = logits_step[0:1, :]
                uncond = logits_step[1:2, :]
                cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
                logits = cond + cfg * (cond - uncond)

                # Apply alignment analyzer
                if alignment_analyzer is not None:
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    last_token = generated_ids[0, -1].item() if generated_ids.size(1) > 0 else None
                    logits = alignment_analyzer.step(logits, next_token=last_token)

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

                if stop_on_eos and next_token.view(-1) == self.t3_config.stop_speech_token:
                    logger.info(f"EOS token detected at step {i + 1}")
                    break

                next_token_embed = self.speech_emb(next_token)
                next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
                next_token_embed = torch.cat([next_token_embed, next_token_embed], dim=0)

                output = self(
                    inputs_embeds=next_token_embed,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=alignment_analyzer is not None,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = output.past_key_values

            if predicted:
                predicted_tokens = torch.cat(predicted, dim=1)
            else:
                predicted_tokens = torch.empty((1, 0), dtype=torch.long, device=device)

            return predicted_tokens[0]

        else:
            # Use HuggingFace's generate() for non-CFG case
            # Store decoder conditioning for prepare_inputs_for_generation
            self._decoder_cond = embeds
            self._current_position = 0
            self.alignment_stream_analyzer = alignment_analyzer

            # Create custom logits processor for alignment analyzer
            class CustomLogitsProcessor:
                def __init__(self, alignment_analyzer_inst):
                    self.alignment_analyzer = alignment_analyzer_inst
                    self.generated_tokens = []

                def __call__(self, input_ids, scores):
                    # Apply alignment analyzer
                    if self.alignment_analyzer is not None:
                        last_token = input_ids[0, -1].item() if input_ids.size(1) > 0 else None
                        scores = self.alignment_analyzer.step(scores, next_token=last_token)

                    return scores

            # Build logits processors
            from transformers.generation.logits_process import LogitsProcessorList

            logits_processors = LogitsProcessorList()

            logits_processors.append(CustomLogitsProcessor(alignment_analyzer))
            logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty)))

            if temperature != 1.0:
                logits_processors.append(TemperatureLogitsWarper(temperature))

            logits_processors.append(MinPLogitsWarper(min_p=min_p))
            logits_processors.append(TopPLogitsWarper(top_p=top_p))

            # Generate using HuggingFace's generate (batch size 1)
            bos_token = torch.tensor([[self.t3_config.start_speech_token]], dtype=torch.long, device=device)

            generated_ids = self.generate(
                input_ids=bos_token,
                max_new_tokens=max_steps,
                do_sample=do_sample,
                logits_processor=logits_processors,
                bos_token_id=self.t3_config.start_speech_token,
                eos_token_id=self.t3_config.stop_speech_token if stop_on_eos else None,
                pad_token_id=self.t3_config.stop_speech_token,
                num_return_sequences=num_return_sequences,
                output_attentions=self.t3_config.use_alignment_analyzer,
                output_hidden_states=True,
                return_dict_in_generate=False,
                use_cache=True,
            )

            # Extract generated tokens (remove BOS)
            predicted_tokens = generated_ids[:, 1:]

            # Clean up generation state
            self._decoder_cond = None
            self._current_position = 0

            return predicted_tokens[0]

    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device


# ============================================================================
# Chatterbox Model
# ============================================================================


class ChatterboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChatterboxConfig
    base_model_prefix = "chatterbox"
    supports_gradient_checkpointing = False


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict


@auto_docstring
class ChatterboxModel(ChatterboxPreTrainedModel):
    """
    Complete Chatterbox TTS Pipeline Model.

    This model combines T3, S3Gen, and HiFTNet to provide a complete text-to-speech pipeline:
    1. Text tokens → T3 → Speech tokens
    2. Speech tokens → S3Gen → Mel spectrogram
    3. Mel spectrogram → HiFTNet → Waveform
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__(config)
        self.config = config

        # Store configuration for multilingual and hiftnet settings
        # Note: hiftnet_config is embedded within s3gen_config, is_multilingual affects t3_config
        self.is_multilingual = config.is_multilingual
        _ = config.hiftnet_config  # Stored in config for serialization

        # Initialize sub-models
        logger.info("Initializing T3 model...")
        self.t3 = T3Model(config.t3_config)

        logger.info("Initializing S3Gen model...")
        self.s3gen = S3GenModel(config.s3gen_config)

        # Sampling rates
        self.s3_sr = 16000  # S3 tokenizer sampling rate
        self.s3gen_sr = 24000  # S3Gen output sampling rate

        # Text tokenizer
        self.text_tokenizer = None
        self.feature_extractor = ChatterboxFeatureExtractor(
            sampling_rate=self.s3_sr,
            s3gen_sampling_rate=self.s3gen_sr,
        )

        # Post init
        self.post_init()

    def load_text_tokenizer(self, tokenizer_path):
        """Load text tokenizer from tokenizer.json file."""
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            logger.warning(f"Tokenizer file not found: {tokenizer_path}")
            return False

        try:
            self.text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            logger.info(f"✓ Loaded text tokenizer from: {tokenizer_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False

    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device

    def prepare_text_tokens(self, text: str, tokenizer=None) -> torch.Tensor:
        """
        Prepare text tokens from raw text.

        Args:
            text: Input text string
            tokenizer: Text tokenizer. If None, uses `self.text_tokenizer`.

        Returns:
            Text tokens with start/stop markers
        """
        text = punc_norm(text)
        # Match chatterbox `EnTokenizer`: replace spaces with a dedicated token before encoding.
        text = text.replace(" ", "[SPACE]")

        # Use provided tokenizer, or self.text_tokenizer
        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                # Tokenizers-style: may return an Encoding with `.ids` or directly a list of ids.
                encoding = tokenizer.encode(text)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                text_tokens = torch.tensor([ids], dtype=torch.long)
            else:
                text_tokens = tokenizer.text_to_tokens(text)
        elif self.text_tokenizer is not None:
            # Use loaded tokenizer
            encoding = self.text_tokenizer.encode(text)
            text_tokens = torch.tensor([encoding.ids], dtype=torch.long)
        else:
            raise ValueError(
                "No text tokenizer provided and `self.text_tokenizer` is not loaded. "
                "Please pass a tokenizer or call `load_text_tokenizer()` first."
            )

        # Add start/stop tokens if not already present
        sot = self.config.t3_config.start_text_token
        eot = self.config.t3_config.stop_text_token

        # Check if start/stop tokens are already in the sequence
        has_start = (text_tokens[0, 0] == sot).item() if text_tokens.numel() > 0 else False
        has_stop = (text_tokens[0, -1] == eot).item() if text_tokens.numel() > 0 else False

        if not has_start:
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        if not has_stop:
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        return text_tokens.to(self.device)

    def prepare_conditionals(
        self, reference_wav: np.ndarray, reference_sr: int, exaggeration: float = 0.5
    ) -> Conditionals:
        """
        Mirror the original Chatterbox prepare_conditionals method for parity.
        """
        extracted = self.feature_extractor.extract_conditioning(
            reference_wav,
            reference_sr,
            s3gen=self.s3gen,
            voice_encoder=self.t3.voice_encoder,
            device=self.device,
            exaggeration=exaggeration,
            speech_cond_prompt_len=self.config.t3_config.speech_cond_prompt_len,
        )
        t3_cond = T3Cond(
            speaker_emb=extracted["speaker_emb"],
            cond_prompt_speech_tokens=extracted["cond_prompt_speech_tokens"],
            emotion_adv=extracted["emotion_adv"],
        )

        return Conditionals(t3=t3_cond, gen=extracted["s3gen_ref_dict"])

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        reference_wav: np.ndarray,
        reference_sr: int,
        tokenizer=None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        max_new_tokens: int = 1000,
        return_intermediates: bool = False,
    ):
        """
        Generate speech from text using the complete pipeline.

        Args:
            text: Input text to synthesize
            reference_wav: Reference audio for voice cloning (numpy array)
            reference_sr: Sampling rate of reference audio
            tokenizer: Optional text tokenizer
            exaggeration: Emotion/expressiveness level (0.0 to 1.0)
            temperature: Sampling temperature for T3
            top_p: Top-p sampling for T3
            min_p: Min-p sampling for T3
            repetition_penalty: Repetition penalty for T3
            cfg_weight: Classifier-free guidance weight for T3
            max_new_tokens: Maximum speech tokens to generate
            return_intermediates: Whether to return intermediate outputs (tokens, mel)

        Returns:
            Waveform tensor, or tuple of (waveform, intermediates) if return_intermediates=True
        """
        logger.info(f"Generating speech for text: '{text}'")

        # Step 1: Prepare text tokens
        text_tokens = self.prepare_text_tokens(text, tokenizer)

        # Step 2: Prepare conditionals (T3 + S3Gen) as in original pipeline
        conds = self.prepare_conditionals(reference_wav, reference_sr, exaggeration)
        t3_cond = conds.t3

        # For CFG: duplicate text tokens before passing to T3
        t3_text_tokens = text_tokens[0]  # Remove batch dimension
        if cfg_weight > 0.0:
            t3_text_tokens = torch.cat([t3_text_tokens.unsqueeze(0), t3_text_tokens.unsqueeze(0)], dim=0)

        speech_tokens = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=t3_text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            cfg_weight=cfg_weight,
        )

        # Extract conditional batch (first sequence)
        if speech_tokens.dim() > 1:
            speech_tokens = speech_tokens[0]

        # Clean up speech tokens
        speech_tokens = drop_invalid_tokens(speech_tokens)

        # Additional safety check - ensure all tokens are valid
        if speech_tokens.max() >= 6561:
            speech_tokens = speech_tokens[speech_tokens < 6561]

        # Step 3: Generate waveform with S3Gen using prepared reference dict
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict={k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in conds.gen.items()},
            finalize=True,
        )

        # Remove batch dimension
        wav = wav.squeeze(0)

        if return_intermediates:
            intermediates = {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
            }
            return wav, intermediates

        return wav

    def forward(
        self,
        text: str,
        reference_wav: np.ndarray,
        reference_sr: int,
        tokenizer=None,
        **kwargs,
    ):
        """Forward pass - calls generate."""
        return self.generate(text, reference_wav, reference_sr, tokenizer, **kwargs)


__all__ = ["ChatterboxPreTrainedModel", "ChatterboxModel"]
