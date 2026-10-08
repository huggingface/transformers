# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from .boundary import BoundaryHead, BoundarySettings
from .boundary_records import RecordHead
from .boundary_relations import RelationProposalSettings, SparseRelationScorer, TypedRelationPairGenerator
from .configuration_gliner2 import Gliner2Config


_CLASSIFICATION_TASK = 4


def _mlp(input_dim, intermediate_dims, output_dim, dropout=0.0, activation="relu"):
    """Build the sequential MLP whose indices match published checkpoints."""
    activations = {"relu": nn.ReLU, "gelu": nn.GELU}
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        layers.append(activations[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _projection(hidden_size, dropout, out_dim=None):
    """Expand by 4, then ReLU and dropout, then project back."""
    if out_dim is None:
        out_dim = hidden_size
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim),
    )


def _extract_elements(sequence, indices):
    """Gather `[B, K, D]` rows from `[B, L, D]`."""
    hidden = sequence.size(-1)
    expanded = indices.unsqueeze(2).expand(-1, -1, hidden)
    return torch.gather(sequence, 1, expanded)


class CompileSafeGRU(nn.Module):
    """Single-layer GRU with `nn.GRU` parameter names."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size**0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, inputs, hidden):
        """Run the GRU. `inputs` is `(seq, batch, input)` and `hidden` is `(batch, hidden)`."""
        seq_len = inputs.shape[0]
        if seq_len == 0:
            return inputs.new_empty(0, hidden.shape[0], self.hidden_size)
        gi_all = F.linear(inputs, self.weight_ih_l0, self.bias_ih_l0)
        outputs = []
        for step in range(seq_len):
            gi = gi_all[step]
            gh = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            reset = torch.sigmoid(i_r + h_r)
            update = torch.sigmoid(i_z + h_z)
            new = torch.tanh(i_n + reset * h_n)
            hidden = (1 - update) * new + update * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=0)


class DownscaledTransformer(nn.Module):
    """Project into a small encoder, then back to the input width."""

    def __init__(self, input_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_projector = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_projector = _mlp(
            hidden_size + input_size,
            [input_size, input_size],
            input_size,
            dropout=0.0,
        )

    def forward(self, inputs):
        projected = self.in_projector(inputs)
        transformed = self.transformer(projected)
        return self.out_projector(torch.cat([transformed, inputs], dim=-1))


class CountLSTM(nn.Module):
    """Count-step embeddings from a learned position and a GRU."""

    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.projector = _mlp(hidden_size * 2, [hidden_size * 4], hidden_size, dropout=0.0)

    def forward(self, field_emb, count):
        """Return `(count, fields, hidden)` embeddings."""
        fields, hidden = field_emb.shape
        count = min(count, self.max_count)
        indices = torch.arange(count, device=field_emb.device)
        positions = self.pos_embedding(indices).unsqueeze(1).expand(count, fields, hidden)
        output = self.gru(positions, field_emb)
        broadcast = field_emb.unsqueeze(0).expand_as(output)
        return self.projector(torch.cat([output, broadcast], dim=-1))


class CountLSTMv2(nn.Module):
    """Count-step embeddings with a GRU followed by a downscaled transformer."""

    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.transformer = DownscaledTransformer(hidden_size, hidden_size=128, num_heads=4, num_layers=2, dropout=0.1)

    def forward(self, field_emb, count):
        """Return `(count, fields, hidden)` embeddings."""
        fields, _ = field_emb.size()
        count = min(count, self.max_count)
        indices = torch.arange(self.max_count, device=field_emb.device)[:count]
        positions = self.pos_embedding(indices).unsqueeze(1).expand(-1, fields, -1)
        output = self.gru(positions, field_emb)
        broadcast = field_emb.unsqueeze(0).expand_as(output)
        return self.transformer(output + broadcast)


class SpanMarkerV0(nn.Module):
    """Span states from projected start and end markers."""

    def __init__(self, hidden_size, max_width, dropout=0.1):
        super().__init__()
        self.max_width = max_width
        self.project_start = _projection(hidden_size, dropout)
        self.project_end = _projection(hidden_size, dropout)
        self.out_project = _projection(hidden_size * 2, dropout, hidden_size)

    def forward(self, hidden, span_idx):
        """Return `[B, L, max_width, D]` span states."""
        batch, length, _ = hidden.size()
        start = _extract_elements(self.project_start(hidden), span_idx[:, :, 0])
        end = _extract_elements(self.project_end(hidden), span_idx[:, :, 1])
        return self.out_project(torch.cat([start, end], dim=-1).relu()).view(batch, length, self.max_width, -1)


class SpanRepLayer(nn.Module):
    """`markerV0` span representation. The submodule name is `span_rep_layer`."""

    def __init__(self, hidden_size, max_width, span_mode="markerV0", dropout=0.1):
        super().__init__()
        if span_mode != "markerV0":
            raise ValueError(f"Unknown span mode {span_mode}")
        self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, dropout=dropout)

    def forward(self, hidden, span_idx):
        return self.span_rep_layer(hidden, span_idx)


def _gather(token_embeddings, indices, mask):
    """Gather rows, clamping pads, then zero them with the mask."""
    hidden = token_embeddings.shape[-1]
    safe = indices.clamp(0, token_embeddings.shape[1] - 1)
    states = token_embeddings.gather(1, safe.unsqueeze(-1).expand(-1, -1, hidden))
    return states * mask.unsqueeze(-1).to(states.dtype)


def _span_indices(length, max_width, device):
    """Build safe `(1, length * max_width, 2)` start/end indices."""
    starts = torch.arange(length, device=device).unsqueeze(1).expand(-1, max_width)
    offsets = torch.arange(max_width, device=device).unsqueeze(0)
    ends = starts + offsets
    valid = ends < length
    starts_flat = starts.reshape(-1)
    ends_flat = ends.reshape(-1)
    invalid = ~valid.reshape(-1)
    starts_flat = torch.where(invalid, torch.zeros_like(starts_flat), starts_flat)
    ends_flat = torch.where(invalid, torch.zeros_like(ends_flat), ends_flat)
    return torch.stack([starts_flat, ends_flat], dim=-1).unsqueeze(0)


def _rows(states, mask, groups, group):
    """Return the valid rows of one sample whose group id matches."""
    keep = mask & (groups == group)
    return states[keep]


def _boundary_parts(parent):
    """Read every boundary-config field into the head settings and side modules."""
    config = parent.boundary_config
    settings = BoundarySettings(
        boundary_dim=config.boundary_dim,
        pair_dim=config.pair_dim,
        boundary_refinement_layers=config.boundary_refinement_layers,
        boundary_ffn_multiplier=config.boundary_ffn_multiplier,
        start_top_k=config.start_top_k,
        end_top_k=config.end_top_k,
        ends_per_start=config.ends_per_start,
        starts_per_end=config.starts_per_end,
        candidate_budget=config.candidate_budget,
        training_candidate_budget=config.training_candidate_budget,
        max_gold_per_query=config.max_gold_per_query,
        end_block_size=config.end_block_size,
        bidirectional_proposals=config.bidirectional_proposals,
        use_inside_evidence=config.use_inside_evidence,
        dropout=config.dropout,
        export_mode=config.export_mode,
        vectorized_pair_elements=config.vectorized_pair_elements,
        enable_span_content=config.enable_span_content,
        content_dim=config.content_dim,
        content_soft_max_pool=config.content_soft_max_pool,
        enable_rotary_endpoints=config.enable_rotary_endpoints,
        rotary_base=config.rotary_base,
        boundary_attention_layers=config.boundary_attention_layers,
        boundary_attention_heads=config.boundary_attention_heads,
        boundary_attention_window=config.boundary_attention_window,
        query_conditioned_inside_weight=config.query_conditioned_inside_weight,
        endpoint_difference_features=config.endpoint_difference_features,
        reranker_endpoint_compat=config.reranker_endpoint_compat,
        multihead_pair_compat_heads=config.multihead_pair_compat_heads,
        boundary_top_k_alpha=config.boundary_top_k_alpha,
        boundary_top_k_max=config.boundary_top_k_max,
        boundary_top_k_bucket=config.boundary_top_k_bucket,
        candidate_pool=config.candidate_pool,
        pool_boundary_top_k=config.pool_boundary_top_k,
        pool_size=config.pool_size,
        min_pool_per_query=config.min_pool_per_query,
        candidate_attention_layers=config.candidate_attention_layers,
        candidate_attention_heads=config.candidate_attention_heads,
        query_attention_layers=config.query_attention_layers,
        enable_abstention=config.enable_abstention,
        enable_count_head=config.enable_count_head,
    )
    extras = {
        "enable_records": config.enable_records,
        "enable_relations": config.enable_relations,
        "record_dim": config.record_dim,
        "record_instance_queries": config.record_instance_queries,
        "relation_heads_per_type": config.relation_heads_per_type,
        "relation_tails_per_type": config.relation_tails_per_type,
        "relation_pair_cap": config.relation_pair_cap,
        "relation_argument_proposal_threshold": config.relation_argument_proposal_threshold,
        "directional_relation_states": config.directional_relation_states,
        "relation_biaffine_content": config.relation_biaffine_content,
        "pair_temperature": config.pair_temperature,
        "relation_temperature": config.relation_temperature,
        "record_temperature": config.record_temperature,
        "overlap_policy": config.overlap_policy,
        "abstention_threshold": config.abstention_threshold,
        "record_anchor_proposal_threshold": config.record_anchor_proposal_threshold,
        "record_anchor_threshold": config.record_anchor_threshold,
        "record_field_threshold": config.record_field_threshold,
        "dropout": config.dropout,
    }
    return settings, extras


@auto_docstring
class Gliner2PreTrainedModel(PreTrainedModel):
    config: Gliner2Config
    base_model_prefix = "gliner2"
    input_modalities = ("text",)
    _no_split_modules = []
    # The text encoder is DeBERTa, which rejects SDPA and runs eager.
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, CompileSafeGRU):
            module.reset_parameters()
            return
        super()._init_weights(module)


@auto_docstring
@dataclass
class Gliner2ModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Encoder states.
    text_states (`torch.FloatTensor` of shape `(batch_size, num_words, hidden_size)`, *optional*):
        One state per word.
    query_states (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        Field-marker states.
    cls_states (`torch.FloatTensor` of shape `(batch_size, num_labels, hidden_size)`, *optional*):
        Classification-label states.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    text_states: torch.FloatTensor | None = None
    query_states: torch.FloatTensor | None = None
    cls_states: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class Gliner2SchemaExtractionOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Encoder states.
    classification_logits (`list`, *optional*):
        One logit vector per classification group, per batch row. Temperature is not applied.
    span_logits (`list`, *optional*):
        One `(count, fields, words, width)` logit tensor per span group, per batch row.
    counts (`list`, *optional*):
        Predicted instance counts, aligned with `span_logits`.
    boundary (`BoundaryHeadOutput`, *optional*):
        Boundary-head marginals and candidates.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    classification_logits: list | None = None
    span_logits: list | None = None
    counts: list | None = None
    boundary: object | None = None


@auto_docstring(
    custom_intro="""
    Encoder plus the shared gather of word, query, and classification-label states.
    """
)
class Gliner2Model(Gliner2PreTrainedModel):
    def __init__(self, config: Gliner2Config):
        super().__init__(config)
        self.encoder = _load_encoder(config.encoder_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_word_indices: torch.LongTensor | None = None,
        text_word_mask: torch.Tensor | None = None,
        query_marker_indices: torch.LongTensor | None = None,
        query_marker_mask: torch.Tensor | None = None,
        cls_marker_indices: torch.LongTensor | None = None,
        cls_marker_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gliner2ModelOutput:
        r"""
        text_word_indices (`torch.LongTensor` of shape `(batch_size, num_words)`, *optional*):
            First-subword index of each word.
        text_word_mask (`torch.Tensor` of shape `(batch_size, num_words)`, *optional*):
            Mask of valid words.
        query_marker_indices (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Token index of each field marker.
        query_marker_mask (`torch.Tensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask of valid field markers.
        cls_marker_indices (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Token index of each classification label.
        cls_marker_mask (`torch.Tensor` of shape `(batch_size, num_labels)`, *optional*):
            Mask of valid classification labels.
        """
        encoder_kwargs = {
            key: kwargs[key] for key in ("output_hidden_states", "output_attentions", "return_dict") if key in kwargs
        }
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **encoder_kwargs,
        )
        hidden = encoded.last_hidden_state
        if text_word_indices is None:
            return Gliner2ModelOutput(
                last_hidden_state=hidden,
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
            )
        return Gliner2ModelOutput(
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            text_states=_gather(hidden, text_word_indices, text_word_mask),
            query_states=_gather(hidden, query_marker_indices, query_marker_mask),
            cls_states=_gather(hidden, cls_marker_indices, cls_marker_mask),
        )


@auto_docstring(
    custom_intro="""
    Schema extraction over a text encoder. The span head or the boundary head is selected by
    `config.architecture`. Every label in the schema is scored in one forward pass.
    """
)
class Gliner2ForSchemaExtraction(Gliner2PreTrainedModel):
    def __init__(self, config: Gliner2Config):
        super().__init__(config)
        self.encoder = _load_encoder(config.encoder_config)
        hidden = config.encoder_config.hidden_size
        self.max_width = config.max_width
        self.counting_layer = config.counting_layer
        self.token_pooling = config.token_pooling
        self.max_len = config.max_len
        self.classification_temperature = config.classification_temperature
        dropout = 0.0
        if config.architecture == "boundary":
            settings, extras = _boundary_parts(config)
            dropout = extras["dropout"]
            self.boundary_head = BoundaryHead(
                hidden,
                settings,
                build_candidate_states=extras["enable_records"],
            )
            self.record_decoder = (
                RecordHead(hidden, extras["record_dim"], extras["record_instance_queries"])
                if extras["enable_records"]
                else None
            )
            if extras["enable_relations"]:
                query_dim = hidden * 2 if extras["directional_relation_states"] else hidden
                self.relation_scorer = SparseRelationScorer(
                    hidden,
                    dropout=extras["dropout"],
                    relation_query_dim=query_dim,
                    use_biaffine_content=extras["relation_biaffine_content"],
                )
                self.relation_pair_generator = TypedRelationPairGenerator(
                    RelationProposalSettings(
                        heads_per_relation=extras["relation_heads_per_type"],
                        tails_per_relation=extras["relation_tails_per_type"],
                        pair_cap=extras["relation_pair_cap"],
                        argument_threshold=extras["relation_argument_proposal_threshold"],
                    )
                )
            else:
                self.relation_scorer = None
                self.relation_pair_generator = None
            self._boundary_extras = extras
        else:
            self.span_rep = SpanRepLayer(hidden, config.max_width, span_mode="markerV0", dropout=0.1)
            if config.counting_layer == "count_lstm_v2":
                self.count_embed = CountLSTMv2(hidden)
            else:
                self.count_embed = CountLSTM(hidden)
            self.count_pred = _mlp(hidden, [hidden * 2], 20, dropout=0.0)
            self.boundary_head = None
            self.record_decoder = None
            self.relation_scorer = None
        self.classifier = _mlp(hidden, [hidden * 2], 1, dropout=dropout)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_word_indices: torch.LongTensor | None = None,
        text_word_mask: torch.Tensor | None = None,
        query_marker_indices: torch.LongTensor | None = None,
        query_marker_mask: torch.Tensor | None = None,
        query_group_index: torch.LongTensor | None = None,
        cls_marker_indices: torch.LongTensor | None = None,
        cls_marker_mask: torch.Tensor | None = None,
        cls_group_index: torch.LongTensor | None = None,
        prompt_marker_indices: torch.LongTensor | None = None,
        prompt_marker_mask: torch.Tensor | None = None,
        prompt_group_index: torch.LongTensor | None = None,
        task_type_ids: torch.LongTensor | None = None,
        group_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gliner2SchemaExtractionOutput:
        r"""
        text_word_indices (`torch.LongTensor` of shape `(batch_size, num_words)`, *optional*):
            First-subword index of each word.
        text_word_mask (`torch.Tensor` of shape `(batch_size, num_words)`, *optional*):
            Mask of valid words.
        query_marker_indices (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Token index of each field marker.
        query_marker_mask (`torch.Tensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask of valid field markers.
        query_group_index (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Schema group of each field marker.
        cls_marker_indices (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Token index of each classification label.
        cls_marker_mask (`torch.Tensor` of shape `(batch_size, num_labels)`, *optional*):
            Mask of valid classification labels.
        cls_group_index (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Schema group of each classification label.
        prompt_marker_indices (`torch.LongTensor` of shape `(batch_size, num_prompts)`, *optional*):
            Token index of each `[P]` marker.
        prompt_marker_mask (`torch.Tensor` of shape `(batch_size, num_prompts)`, *optional*):
            Mask of valid `[P]` markers.
        prompt_group_index (`torch.LongTensor` of shape `(batch_size, num_prompts)`, *optional*):
            Schema group of each `[P]` marker.
        task_type_ids (`torch.LongTensor` of shape `(batch_size, num_groups)`, *optional*):
            Task id of each schema group.
        group_mask (`torch.Tensor` of shape `(batch_size, num_groups)`, *optional*):
            Mask of valid schema groups.
        """
        encoder_kwargs = {
            key: kwargs[key] for key in ("output_hidden_states", "output_attentions", "return_dict") if key in kwargs
        }
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **encoder_kwargs,
        )
        hidden = encoded.last_hidden_state
        if text_word_indices is None:
            return Gliner2SchemaExtractionOutput(
                last_hidden_state=hidden,
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
            )

        text_states = _gather(hidden, text_word_indices, text_word_mask)
        query_states = _gather(hidden, query_marker_indices, query_marker_mask)
        cls_states = _gather(hidden, cls_marker_indices, cls_marker_mask)
        if self.config.architecture == "boundary":
            boundary = self.boundary_head(
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                return_candidates=True,
            )
            return Gliner2SchemaExtractionOutput(
                last_hidden_state=hidden,
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
                classification_logits=_classification_logits(
                    cls_states, cls_marker_mask, cls_group_index, task_type_ids, group_mask, self.classifier
                ),
                boundary=boundary,
            )

        prompt_states = None
        if prompt_marker_indices is not None:
            prompt_states = _gather(hidden, prompt_marker_indices, prompt_marker_mask)
        span_logits, counts = _span_logits(
            text_states,
            text_word_mask,
            query_states,
            query_marker_mask,
            query_group_index,
            prompt_states,
            prompt_marker_mask,
            prompt_group_index,
            task_type_ids,
            group_mask,
            self.span_rep,
            self.count_pred,
            self.count_embed,
            self.max_width,
        )
        return Gliner2SchemaExtractionOutput(
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            classification_logits=_classification_logits(
                cls_states, cls_marker_mask, cls_group_index, task_type_ids, group_mask, self.classifier
            ),
            span_logits=span_logits,
            counts=counts,
        )


def _load_encoder(encoder_config):
    """Build the encoder from its config. DeBERTa rejects SDPA and falls back to eager."""
    try:
        return AutoModel.from_config(encoder_config)
    except (ValueError, RuntimeError):
        encoder_config._attn_implementation = "eager"
        return AutoModel.from_config(encoder_config)


def _classification_logits(cls_states, cls_mask, cls_groups, task_ids, group_mask, classifier):
    """One raw logit vector per classification group."""
    if cls_groups is None or task_ids is None:
        return None
    batch = cls_states.shape[0]
    rows = []
    for index in range(batch):
        sample = []
        for group in range(task_ids.shape[1]):
            if not bool(group_mask[index, group]) or int(task_ids[index, group]) != _CLASSIFICATION_TASK:
                continue
            states = _rows(cls_states[index], cls_mask[index], cls_groups[index], group)
            if states.numel() == 0:
                sample.append(states.new_zeros(0))
            else:
                sample.append(classifier(states).squeeze(-1))
        rows.append(sample)
    return rows


def _span_logits(
    text_states,
    text_mask,
    query_states,
    query_mask,
    query_groups,
    prompt_states,
    prompt_mask,
    prompt_groups,
    task_ids,
    group_mask,
    span_rep,
    count_pred,
    count_embed,
    max_width,
):
    """Pre-sigmoid span logits and instance counts, one entry per span group."""
    if query_groups is None or task_ids is None or prompt_states is None:
        return None, None
    span_rows = []
    count_rows = []
    for index in range(text_states.shape[0]):
        words = text_states[index][text_mask[index]]
        length = words.shape[0]
        rep = None
        if length:
            indices = _span_indices(length, max_width, words.device)
            rep = span_rep(words.unsqueeze(0), indices).squeeze(0)
        sample_spans = []
        sample_counts = []
        for group in range(task_ids.shape[1]):
            if not bool(group_mask[index, group]) or int(task_ids[index, group]) == _CLASSIFICATION_TASK:
                continue
            fields = _rows(query_states[index], query_mask[index], query_groups[index], group)
            prompt = _rows(prompt_states[index], prompt_mask[index], prompt_groups[index], group)
            if prompt.numel() == 0 or fields.numel() == 0 or rep is None:
                width = max_width
                sample_spans.append(text_states.new_zeros(0, fields.shape[0], length, width))
                sample_counts.append(0)
                continue
            predicted = int(count_pred(prompt[:1]).argmax(dim=-1).item())
            sample_counts.append(predicted)
            if predicted <= 0:
                sample_spans.append(text_states.new_zeros(0, fields.shape[0], length, max_width))
                continue
            projected = count_embed(fields, predicted)
            sample_spans.append(torch.einsum("lkd,bpd->bplk", rep, projected))
        span_rows.append(sample_spans)
        count_rows.append(sample_counts)
    return span_rows, count_rows


__all__ = [
    "Gliner2PreTrainedModel",
    "Gliner2Model",
    "Gliner2ForSchemaExtraction",
    "Gliner2ModelOutput",
    "Gliner2SchemaExtractionOutput",
]
