# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""GraniteSpeechNar: Non-autoregressive ASR with conformer encoder, QFormer projector,
and bidirectional Granite LLM backbone."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...masking_utils import (
    create_bidirectional_mask,
    find_packed_sequence_indices,
    packed_sequence_mask_function,
)
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from ..granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteForCausalLM, GraniteModel
from ..granite_speech.modeling_granite_speech import GraniteSpeechConformerBlock
from .configuration_granite_speech_nar import (
    GraniteSpeechNarConfig,
    GraniteSpeechNarEncoderConfig,
    GraniteSpeechNarProjectorConfig,
)


logger = logging.get_logger(__name__)


@dataclass
class GraniteSpeechNarEncoderOutput(ModelOutput):
    """Output of the GraniteSpeechNar encoder."""

    loss: torch.Tensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    all_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class GraniteSpeechNarOutput(ModelOutput):
    """Output of the GraniteSpeechNarForASR model.

    Attributes:
        loss: Combined CTC + auxiliary losses (only when labels provided).
        preds: List of predicted token ID tensors per sample (after CTC collapse, inference only).
        logits: List of per-sample logit tensors from the LLM head.
        encoder_logits: Flat BPE CTC logits from the encoder.
        encoder_preds: List of CTC-collapsed encoder predictions per sample.
    """

    loss: torch.Tensor | None = None
    preds: list[torch.Tensor] | None = None
    logits: list[torch.Tensor] | None = None
    encoder_logits: torch.Tensor | None = None
    encoder_preds: list[torch.Tensor] | None = None


class GraniteSpeechNarConformerBlock(GraniteSpeechConformerBlock):
    pass


def _posterior_weighted_pool(hidden: torch.Tensor, importance: torch.Tensor, window_size: int = 4) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = hidden.shape
    pad_len = (window_size - seq_len % window_size) % window_size
    if pad_len > 0:
        hidden = F.pad(hidden, (0, 0, 0, pad_len))
        importance = F.pad(importance, (0, pad_len))
    num_windows = hidden.shape[1] // window_size
    hidden = hidden.view(batch_size, num_windows, window_size, hidden_dim)
    importance = importance.view(batch_size, num_windows, window_size)
    weights = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
    pooled = (hidden * weights.unsqueeze(-1)).sum(dim=2)
    return pooled


class GraniteSpeechNarQFormerCrossAttention(nn.Module):
    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        encoder_len = encoder_hidden_states.shape[1]

        query_states = (
            self.q_proj(hidden_states).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.k_proj(encoder_hidden_states)
            .view(batch_size, encoder_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(encoder_hidden_states)
            .view(batch_size, encoder_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
        return self.o_proj(attn_output)


class GraniteSpeechNarQFormerMLP(nn.Module):
    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        mlp_hidden_size = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, mlp_hidden_size, bias=config.mlp_bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_hidden_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class GraniteSpeechNarQFormerLayer(nn.Module):
    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.cross_attention = GraniteSpeechNarQFormerCrossAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.mlp = GraniteSpeechNarQFormerMLP(config)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.cross_attention(self.attn_norm(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class GraniteSpeechNarQFormer(nn.Module):
    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        self.layers = nn.ModuleList([GraniteSpeechNarQFormerLayer(config) for _ in range(config.num_layers)])

    def forward(self, query_embeds: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = query_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class GraniteSpeechNarProjector(nn.Module):
    """Windowed QFormer projector that maps multi-layer encoder features to LLM embedding space."""

    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        self.config = config
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.encoder_dim, eps=config.layernorm_eps) for _ in range(config.num_encoder_layers)]
        )
        self.layer_projector = nn.Linear(config.encoder_dim * config.num_encoder_layers, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.projector_act = nn.GELU()
        self.qformer = GraniteSpeechNarQFormer(config)

        query_length = config.block_size // config.downsample_rate
        embed_std = config.hidden_size**-0.5
        self.query = nn.Parameter(torch.randn(1, query_length, config.hidden_size) * embed_std)
        self.window_positions = nn.Parameter(torch.randn(1, config.block_size, config.hidden_size) * embed_std)
        self.out_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.out_linear = nn.Linear(config.hidden_size, config.llm_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.size()

        hidden_states = hidden_states.view(
            batch_size, seq_len, self.config.num_encoder_layers, self.config.encoder_dim
        )
        normalized_layers = []
        for i, layer_norm in enumerate(self.layer_norms):
            normalized_layers.append(layer_norm(hidden_states[:, :, i]))
        hidden_states = torch.cat(normalized_layers, dim=-1)

        hidden_states = self.projector_act(self.layer_projector(hidden_states))

        block_size = self.config.block_size
        nblocks = seq_len // block_size
        rest = seq_len % block_size
        if rest > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, block_size - rest), "constant", 0)
            nblocks += 1

        hidden_states = hidden_states.view(batch_size * nblocks, block_size, self.config.hidden_size)
        query_length = self.query.shape[1]
        mean_pool = hidden_states.view(
            batch_size * nblocks, query_length, self.config.downsample_rate, self.config.hidden_size
        ).mean(dim=-2)

        hidden_states = self.qformer(
            query_embeds=self.dropout(self.query + mean_pool),
            encoder_hidden_states=self.dropout(hidden_states + self.window_positions),
        )

        hidden_states = hidden_states.view(batch_size, nblocks * query_length, -1)
        hidden_states = self.dropout(self.out_norm(hidden_states))
        return self.out_linear(hidden_states)


@auto_docstring
class GraniteSpeechNarPreTrainedModel(PreTrainedModel):
    config_class = GraniteSpeechNarConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["GraniteSpeechNarConformerBlock", "GraniteDecoderLayer"]
    input_modalities = ("audio",)


class GraniteSpeechNarAttention(GraniteAttention):
    """GraniteAttention with is_causal=False for bidirectional attention."""

    is_causal = False

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False


class GraniteSpeechNarDecoderLayer(GraniteDecoderLayer):
    """GraniteDecoderLayer using bidirectional attention."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteSpeechNarAttention(config=config, layer_idx=layer_idx)


class GraniteSpeechNarModel(GraniteModel):
    """GraniteModel with bidirectional (non-causal) attention.

    Uses GraniteSpeechNarDecoderLayer which sets is_causal=False,
    and replaces create_causal_mask() with create_bidirectional_mask() so all
    attention backends (SDPA, FA2, eager, flex) get a proper non-causal mask.
    """

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteSpeechNarDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        packed_seq_mask = find_packed_sequence_indices(position_ids)
        and_mask_fn = packed_sequence_mask_function(packed_seq_mask) if packed_seq_mask is not None else None
        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            and_mask_function=and_mask_fn,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # KV cache is not needed in a non-autoregressive model
        kwargs["use_cache"] = False
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=bidirectional_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class GraniteSpeechNarCTCEncoder(GraniteSpeechNarPreTrainedModel):
    """Conformer encoder with BPE CTC head and multi-layer output."""

    config_class = GraniteSpeechNarEncoderConfig

    def __init__(self, config: GraniteSpeechNarEncoderConfig):
        super().__init__(config)
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = nn.ModuleList([GraniteSpeechNarConformerBlock(config) for _ in range(config.num_layers)])
        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.out_bpe = None
        if config.bpe_output_dim is not None:
            self.out_bpe = nn.Linear(config.hidden_dim, config.bpe_output_dim, bias=True)
        self.dropout = nn.Dropout(config.pred_dropout)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> GraniteSpeechNarEncoderOutput:
        if attention_mask is None:
            attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.bool, device=input_features.device)

        hidden_states = self.input_linear(input_features.to(self.dtype))
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        blank_probs = None

        context_size = self.config.context_size
        seq = torch.arange(context_size, device=hidden_states.device)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + self.config.max_pos_emb

        for layer_idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=attention_dists)

            if layer_idx == self.config.self_conditioning_layer:
                mid_logits = self.out(self.dropout(hidden_states))
                mid_probs = torch.softmax(mid_logits.float(), dim=-1)
                blank_probs = mid_probs[:, :, 0]
                hidden_states = hidden_states + self.out_mid(mid_probs.to(hidden_states.dtype))

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.dropout(hidden_states)

        logits = None
        loss = None
        if self.out_bpe is not None and blank_probs is not None:
            pool_window = self.config.bpe_pooling_window
            importance = 1.0 - blank_probs
            pooled = _posterior_weighted_pool(hidden_states.float(), importance, window_size=pool_window).to(
                hidden_states.dtype
            )
            encoder_lengths = attention_mask.sum(dim=1)
            lengths = -(encoder_lengths // -pool_window)
            lengths_list = lengths.tolist()
            logits = self.out_bpe(torch.cat([pooled[i, :length] for i, length in enumerate(lengths_list)]))

            if labels is not None:
                logits_padded = logits.new_zeros(len(lengths_list), max(lengths_list), logits.shape[-1])
                offset = 0
                for i, length in enumerate(lengths_list):
                    logits_padded[i, :length] = logits[offset : offset + length]
                    offset += length

                log_probs = torch.log_softmax(logits_padded.float(), dim=-1)
                loss = (
                    F.ctc_loss(
                        log_probs.transpose(0, 1),
                        labels + 1,
                        lengths,
                        label_lengths,
                        blank=0,
                        reduction="sum",
                        zero_infinity=True,
                    )
                    / lengths.sum()
                )

        return GraniteSpeechNarEncoderOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The bidirectional language model component of GraniteSpeechNar, used internally
    to refine CTC predictions in a single non-autoregressive pass.
    """
)
class GraniteSpeechNarLM(GraniteForCausalLM):
    """GraniteForCausalLM with a bidirectional (non-causal) backbone."""

    def __init__(self, config):
        super().__init__(config)
        self.model = GraniteSpeechNarModel(config)


@auto_docstring(
    custom_intro="""
    The GraniteSpeechNar model for non-autoregressive automatic speech recognition.
    Consists of a conformer encoder with BPE CTC head, a QFormer-based projector,
    and a bidirectional Granite LLM backbone that refines CTC predictions in a single pass.
    """
)
class GraniteSpeechNarForASR(GraniteSpeechNarPreTrainedModel):
    def __init__(self, config: GraniteSpeechNarConfig):
        super().__init__(config)

        self.encoder = GraniteSpeechNarCTCEncoder(config.encoder_config)
        self.projector = GraniteSpeechNarProjector(config.projector_config)

        text_config = config.text_config
        if hasattr(config, "_attn_implementation"):
            text_config._attn_implementation = config._attn_implementation
        self.language_model = GraniteSpeechNarLM._from_config(text_config)

        self.post_init()

    def _ctc_collapse_decode(
        self,
        bpe_logits_flat: torch.Tensor,
        bpe_lengths: list[int],
    ) -> list[torch.Tensor]:
        """GPU CTC greedy decode: argmax -> unique_consecutive -> remove blank -> shift."""
        preds_flat = bpe_logits_flat.argmax(dim=-1)
        per_sample = preds_flat.split(bpe_lengths)
        return [(collapsed := torch.unique_consecutive(seq))[collapsed != 0] - 1 for seq in per_sample]

    def _add_insertion_slots(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Insert blank tokens between each CTC token as editing slots for the LLM."""
        blank_id = self.config.blank_token_id
        n = token_ids.numel()
        total_len = max(2 * n + 1, self.config.min_edit_sequence_length)
        idx = torch.arange(n, device=token_ids.device)
        out_idx = 2 * idx + 1
        out = torch.full((total_len,), fill_value=blank_id, dtype=token_ids.dtype, device=token_ids.device)
        out[out_idx] = token_ids
        return out

    def _build_flat_inputs(
        self,
        ctc_token_ids: list[torch.Tensor],
        audio_embeds: torch.Tensor,
        audio_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Build flat (pad-free) LLM input: [audio_0, text_0, audio_1, text_1, ...]"""
        embed_tokens = self.language_model.model.embed_tokens

        embeds_list = []
        position_ids_list = []
        text_lengths = []

        for i, audio_len in enumerate(audio_lengths):
            audio_emb = audio_embeds[i, :audio_len]
            text_ids_with_slots = self._add_insertion_slots(ctc_token_ids[i])
            text_emb = embed_tokens(text_ids_with_slots)
            sample_embeds = torch.cat([audio_emb, text_emb], dim=0)
            embeds_list.append(sample_embeds)
            position_ids_list.append(torch.arange(sample_embeds.shape[0], device=audio_embeds.device))
            text_lengths.append(text_ids_with_slots.shape[0])

        flat_embeds = torch.cat(embeds_list, dim=0).unsqueeze(0)
        flat_position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)
        return flat_embeds, flat_position_ids, text_lengths

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        output_encoder_logits: bool = False,
        **kwargs,
    ) -> GraniteSpeechNarOutput:
        r"""
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, seq_len, input_dim)`):
                Mel spectrogram features.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Encoder attention mask (1 for valid frames, 0 for padding).
            labels (`torch.Tensor` of shape `(batch_size, max_label_len)`, *optional*):
                Ground truth LLM token IDs for training.
            label_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Number of valid tokens per sample in `labels`.
            output_encoder_logits (`bool`, *optional*, defaults to `False`):
                Whether to return encoder BPE logits. When False, the large logits
                tensor is freed early to reduce peak memory.

        Returns:
            [`GraniteSpeechNarOutput`]
        """
        encoder_labels = labels if self.config.encoder_ctc_loss_lambda else None
        enc_out = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            labels=encoder_labels,
            label_lengths=label_lengths if encoder_labels is not None else None,
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.bool, device=input_features.device)

        encoder_lengths = attention_mask.sum(dim=1)

        pool_window = self.encoder.config.bpe_pooling_window
        bpe_lengths = (-(encoder_lengths // -pool_window)).tolist()
        ctc_token_ids = self._ctc_collapse_decode(enc_out.logits, bpe_lengths)

        multilayer_features = torch.cat(
            [enc_out.all_hidden_states[idx] for idx in self.config.encoder_layer_indices], dim=-1
        )

        encoder_loss = enc_out.loss
        encoder_logits = enc_out.logits if output_encoder_logits else None
        del enc_out

        audio_embeds = self.projector(multilayer_features)
        del multilayer_features
        if self.config.scale_projected_embeddings:
            embedding_multiplier = getattr(self.config.text_config, "embedding_multiplier", 1.0)
            audio_embeds = audio_embeds / embedding_multiplier
        audio_embeds = audio_embeds.to(self.language_model.model.embed_tokens.weight.dtype)

        audio_lengths = (encoder_lengths // self.projector.config.downsample_rate).cpu().tolist()

        flat_embeds, flat_position_ids, text_lengths = self._build_flat_inputs(
            ctc_token_ids, audio_embeds, audio_lengths
        )

        llm_out = self.language_model(
            inputs_embeds=flat_embeds,
            position_ids=flat_position_ids,
        )
        all_logits = llm_out.logits.squeeze(0)

        segment_lengths = [l for a, t in zip(audio_lengths, text_lengths) for l in (a, t)]
        text_logits = torch.cat(list(all_logits.split(segment_lengths)[1::2]))
        logits_per_sample = list(text_logits.split(text_lengths))

        loss = None
        if labels is not None:
            log_probs = torch.log_softmax(text_logits.float(), dim=-1)

            log_probs_padded = log_probs.new_zeros(len(text_lengths), max(text_lengths), log_probs.shape[-1])
            offset = 0
            for i, tl in enumerate(text_lengths):
                log_probs_padded[i, :tl] = log_probs[offset : offset + tl]
                offset += tl

            input_lengths = torch.tensor(text_lengths, device=text_logits.device)

            loss = (
                F.ctc_loss(
                    log_probs_padded.transpose(0, 1),
                    labels,
                    input_lengths,
                    label_lengths,
                    blank=self.config.blank_token_id,
                    reduction="sum",
                    zero_infinity=True,
                )
                / input_lengths.sum()
            )

            if self.config.ce_loss_lambda > 0.0:
                ce_targets = torch.cat([self._add_insertion_slots(ids) for ids in ctc_token_ids])
                ce_loss = F.cross_entropy(
                    text_logits,
                    ce_targets.long(),
                    reduction="mean",
                    ignore_index=-100,
                )
                loss = loss + self.config.ce_loss_lambda * ce_loss

            if encoder_loss is not None:
                loss = loss + self.config.encoder_ctc_loss_lambda * encoder_loss

        return GraniteSpeechNarOutput(
            loss=loss,
            logits=logits_per_sample,
            encoder_logits=encoder_logits,
            encoder_preds=ctc_token_ids,
        )

    @torch.inference_mode()
    def transcribe(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_encoder_logits: bool = False,
    ) -> GraniteSpeechNarOutput:
        """Single-pass non-autoregressive inference: forward + CTC collapse on LLM output.

        Returns token ID tensors in `preds`. Use `GraniteSpeechNarProcessor.batch_decode()`
        to convert to strings.
        """
        output = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_encoder_logits=output_encoder_logits,
        )

        blank_id = self.config.blank_token_id
        preds = []
        for sample_logits in output.logits:
            pred = torch.unique_consecutive(sample_logits.argmax(-1))
            pred = pred[pred != blank_id]
            preds.append(pred)

        return GraniteSpeechNarOutput(
            preds=preds,
            logits=output.logits,
            encoder_logits=output.encoder_logits,
            encoder_preds=output.encoder_preds,
        )


__all__ = [
    "GraniteSpeechNarModel",
    "GraniteSpeechNarCTCEncoder",
    "GraniteSpeechNarForASR",
    "GraniteSpeechNarLM",
    "GraniteSpeechNarPreTrainedModel",
]
