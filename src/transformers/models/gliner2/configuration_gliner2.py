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

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="fastino/GLiNER2.5-Decide")
@strict
class Gliner2BoundaryConfig(PreTrainedConfig):
    r"""
    Boundary-head settings. Only fields that change parameters or decoding are stored.

    boundary_dim (`int`, *optional*, defaults to `128`):
        Width of the boundary-token states.
    pair_dim (`int`, *optional*, defaults to `128`):
        Width of the start and end pair scorer.
    boundary_refinement_layers (`int`, *optional*, defaults to `1`):
        Number of boundary-state refinement blocks.
    boundary_ffn_multiplier (`float`, *optional*, defaults to `2.0`):
        Expansion ratio of the boundary feed-forward.
    start_top_k (`int`, *optional*, defaults to `16`):
        Start positions kept per query before pairing.
    end_top_k (`int`, *optional*, defaults to `16`):
        End positions kept per query before pairing.
    ends_per_start (`int`, *optional*, defaults to `8`):
        Ends retained for each start candidate.
    starts_per_end (`int`, *optional*, defaults to `8`):
        Starts retained for each end candidate.
    candidate_budget (`int`, *optional*, defaults to `128`):
        Maximum span candidates scored per query.
    training_candidate_budget (`int`, *optional*, defaults to `160`):
        Candidate cap used while training.
    max_gold_per_query (`int`, *optional*, defaults to `32`):
        Maximum gold spans packed per query.
    end_block_size (`int`, *optional*, defaults to `256`):
        Token block size of the end-position scan.
    bidirectional_proposals (`bool`, *optional*, defaults to `True`):
        Propose spans from both the start side and the end side.
    use_inside_evidence (`bool`, *optional*, defaults to `True`):
        Include inside-span features in the pair score.
    dropout (`float`, *optional*, defaults to `0.1`):
        Dropout probability on boundary and pair layers.
    export_mode (`str`, *optional*, defaults to `'auto'`):
        Proposal path: `"auto"`, `"streaming"`, or `"vectorized"`.
    vectorized_pair_elements (`int`, *optional*, defaults to `16777216`):
        Pair elements scored in one vectorized step.
    enable_span_content (`bool`, *optional*, defaults to `False`):
        Add a pooled content vector to each candidate.
    content_dim (`int`, *optional*, defaults to `64`):
        Width of the span-content vector.
    content_soft_max_pool (`bool`, *optional*, defaults to `False`):
        Use soft max-pooling for span content.
    enable_rotary_endpoints (`bool`, *optional*, defaults to `False`):
        Add rotary features to the start and end states.
    rotary_base (`float`, *optional*, defaults to `10000.0`):
        Base of the endpoint rotary embedding.
    boundary_attention_layers (`int`, *optional*, defaults to `0`):
        Local attention layers applied to boundary states.
    boundary_attention_heads (`int`, *optional*, defaults to `4`):
        Number of heads in the boundary attention layers.
    boundary_attention_window (`int`, *optional*, defaults to `0`):
        Window of boundary attention. `0` attends over the whole sequence.
    query_conditioned_inside_weight (`bool`, *optional*, defaults to `False`):
        Weight inside tokens with the query.
    endpoint_difference_features (`bool`, *optional*, defaults to `False`):
        Add start-minus-end features to the pair representation.
    reranker_endpoint_compat (`bool`, *optional*, defaults to `True`):
        Keep the published endpoint-feature layout.
    multihead_pair_compat_heads (`int`, *optional*, defaults to `8`):
        Heads in the pair compatibility projection.
    boundary_top_k_alpha (`float`, *optional*, defaults to `0.0`):
        Extra starts and ends added as the sequence grows.
    boundary_top_k_max (`int`, *optional*, defaults to `128`):
        Cap on the length-dependent boundary top-k.
    boundary_top_k_bucket (`int`, *optional*, defaults to `8`):
        Rounding bucket for the length-dependent top-k.
    candidate_pool (`Literal['per_query', 'shared']`, *optional*, defaults to `'per_query'`):
        `"per_query"` proposals, or one shared document pool.
    pool_boundary_top_k (`int`, *optional*, defaults to `64`):
        Boundary positions admitted to the shared pool.
    pool_size (`int`, *optional*, defaults to `384`):
        Maximum candidates stored in the shared pool.
    min_pool_per_query (`int`, *optional*, defaults to `8`):
        Minimum shared-pool slots reserved for each query.
    candidate_attention_layers (`int`, *optional*, defaults to `2`):
        Attention layers over candidates in the shared pool.
    candidate_attention_heads (`int`, *optional*, defaults to `4`):
        Heads in the shared-pool candidate attention.
    query_attention_layers (`int`, *optional*, defaults to `1`):
        Query-to-candidate attention layers.
    enable_abstention (`bool`, *optional*, defaults to `True`):
        Score a null class so a query can return nothing.
    enable_count_head (`bool`, *optional*, defaults to `True`):
        Predict how many spans each query has.
    enable_records (`bool`, *optional*, defaults to `False`):
        Score structured records on top of the candidates.
    enable_relations (`bool`, *optional*, defaults to `False`):
        Score typed relations between candidate spans.
    record_dim (`int`, *optional*, defaults to `128`):
        Width of the record-head states.
    record_instance_queries (`int`, *optional*, defaults to `8`):
        Latent record slots available to each structure.
    relation_heads_per_type (`int`, *optional*, defaults to `32`):
        Head candidates kept for each relation type.
    relation_tails_per_type (`int`, *optional*, defaults to `32`):
        Tail candidates kept for each relation type.
    relation_pair_cap (`int`, *optional*, defaults to `128`):
        Maximum typed pairs scored in one document.
    relation_argument_proposal_threshold (`float`, *optional*, defaults to `0.0`):
        Minimum score for a span to be a relation argument.
    directional_relation_states (`bool`, *optional*, defaults to `False`):
        Keep separate states for the relation head and tail.
    relation_biaffine_content (`bool`, *optional*, defaults to `False`):
        Add a biaffine content term to the relation score.
    pair_temperature (`float`, *optional*, defaults to `1.0`):
        Divisor applied to span-pair logits before the activation.
    relation_temperature (`float`, *optional*, defaults to `1.0`):
        Divisor applied to relation logits before the activation.
    record_temperature (`float`, *optional*, defaults to `1.0`):
        Divisor applied to record logits before the activation.
    overlap_policy (`str`, *optional*, defaults to `'flat'`):
        How overlapping spans are kept: `"flat"`, `"nested"`, or `"longest"`.
    abstention_threshold (`float`, *optional*, defaults to `0.5`):
        Null-class score above which a query returns nothing.
    record_anchor_proposal_threshold (`float`, *optional*, defaults to `0.5`):
        Score that can rescue a record anchor into the proposal set.
    record_anchor_threshold (`float`, *optional*, defaults to `0.5`):
        Score required to keep a record anchor.
    record_field_threshold (`float`, *optional*, defaults to `0.5`):
        Score cutoff for record fields.
    abstention_loss_weight (`float`, *optional*, defaults to `0.2`):
        Weight of the abstention loss.
    adaptive_threshold (`bool`, *optional*, defaults to `False`):
        Adjust the span cutoff using the count head.
    boundary_focal_clip (`float`, *optional*, defaults to `0.05`):
        Minimum probability used by the boundary focal loss.
    boundary_focal_gamma_negative (`float`, *optional*, defaults to `2.0`):
        Focal gamma on negative boundary labels.
    boundary_focal_gamma_positive (`float`, *optional*, defaults to `0.0`):
        Focal gamma on positive boundary labels.
    boundary_marginal_loss (`str`, *optional*, defaults to `"asymmetric_focal"`):
        Boundary objective, `"bce"` or `"asymmetric_focal"`.
    boundary_negative_weight (`float`, *optional*, defaults to `0.5`):
        Weight of positions that are not a boundary.
    classification_loss_weight (`float`, *optional*, defaults to `1.0`):
        Weight of the classification loss.
    classification_temperature (`float`, *optional*, defaults to `1.0`):
        Divisor applied to classification logits before the activation.
    consistency_loss_weight (`float`, *optional*, defaults to `0.1`):
        Weight of the proposal-consistency loss.
    consistency_warmup_steps (`int`, *optional*, defaults to `2000`):
        Steps before the consistency loss reaches its full weight.
    count_loss_weight (`float`, *optional*, defaults to `0.2`):
        Weight of the count-head loss.
    hard_negative_keep_all_when_absent (`bool`, *optional*, defaults to `True`):
        Keep every negative span when a query has no gold span.
    hard_negatives_per_positive (`int`, *optional*, defaults to `20`):
        Hard negatives sampled for each gold span.
    loss_reduction (`str`, *optional*, defaults to `"sum"`):
        How boundary losses are reduced across the batch.
    max_negative_queries_per_batch (`int`, *optional*, defaults to `64`):
        Maximum queries in a batch that have no gold spans.
    minimum_hard_negatives (`int`, *optional*, defaults to `16`):
        Minimum hard negatives kept for each query.
    negative_query_ratio (`float`, *optional*, defaults to `1.0`):
        Fraction of the batch reserved for queries with no gold.
    proposal_loss_weight (`float`, *optional*, defaults to `0.3`):
        Weight of the boundary-proposal loss.
    record_loss_weight (`float`, *optional*, defaults to `1.0`):
        Weight of the record loss.
    relation_loss_weight (`float`, *optional*, defaults to `1.0`):
        Weight of the relation loss.
    rerank_listwise_weight (`float`, *optional*, defaults to `0.3`):
        Weight of the listwise reranking loss.
    soft_iou_anneal_steps (`int`, *optional*, defaults to `20000`):
        Steps over which the soft-IoU loss reaches its full weight.
    soft_iou_aux_weight (`float`, *optional*, defaults to `0.2`):
        Weight of the soft-IoU auxiliary loss.
    """

    model_type = "gliner2_boundary"

    boundary_dim: int = 128
    pair_dim: int = 128
    boundary_refinement_layers: int = 1
    boundary_ffn_multiplier: float = 2.0
    start_top_k: int = 16
    end_top_k: int = 16
    ends_per_start: int = 8
    starts_per_end: int = 8
    candidate_budget: int = 128
    training_candidate_budget: int = 160
    max_gold_per_query: int = 32
    end_block_size: int = 256
    bidirectional_proposals: bool = True
    use_inside_evidence: bool = True
    dropout: float = 0.1
    export_mode: str = "auto"
    vectorized_pair_elements: int = 16_777_216
    enable_span_content: bool = False
    content_dim: int = 64
    content_soft_max_pool: bool = False
    enable_rotary_endpoints: bool = False
    rotary_base: float = 10000.0
    boundary_attention_layers: int = 0
    boundary_attention_heads: int = 4
    boundary_attention_window: int = 0
    query_conditioned_inside_weight: bool = False
    endpoint_difference_features: bool = False
    reranker_endpoint_compat: bool = True
    multihead_pair_compat_heads: int = 8
    boundary_top_k_alpha: float = 0.0
    boundary_top_k_max: int = 128
    boundary_top_k_bucket: int = 8
    candidate_pool: Literal["per_query", "shared"] = "per_query"
    pool_boundary_top_k: int = 64
    pool_size: int = 384
    min_pool_per_query: int = 8
    candidate_attention_layers: int = 2
    candidate_attention_heads: int = 4
    query_attention_layers: int = 1
    enable_abstention: bool = True
    enable_count_head: bool = True
    enable_records: bool = False
    enable_relations: bool = False
    record_dim: int = 128
    record_instance_queries: int = 8
    relation_heads_per_type: int = 32
    relation_tails_per_type: int = 32
    relation_pair_cap: int = 128
    relation_argument_proposal_threshold: float = 0.0
    directional_relation_states: bool = False
    relation_biaffine_content: bool = False
    pair_temperature: float = 1.0
    relation_temperature: float = 1.0
    record_temperature: float = 1.0
    overlap_policy: str = "flat"
    abstention_threshold: float = 0.5
    record_anchor_proposal_threshold: float = 0.5
    record_anchor_threshold: float = 0.5
    record_field_threshold: float = 0.5
    abstention_loss_weight: float = 0.2
    adaptive_threshold: bool = False
    boundary_focal_clip: float = 0.05
    boundary_focal_gamma_negative: float = 2.0
    boundary_focal_gamma_positive: float = 0.0
    boundary_marginal_loss: str = "asymmetric_focal"
    boundary_negative_weight: float = 0.5
    classification_loss_weight: float = 1.0
    classification_temperature: float = 1.0
    consistency_loss_weight: float = 0.1
    consistency_warmup_steps: int = 2000
    count_loss_weight: float = 0.2
    hard_negative_keep_all_when_absent: bool = True
    hard_negatives_per_positive: int = 20
    loss_reduction: str = "sum"
    max_negative_queries_per_batch: int = 64
    minimum_hard_negatives: int = 16
    negative_query_ratio: float = 1.0
    proposal_loss_weight: float = 0.3
    record_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    rerank_listwise_weight: float = 0.3
    soft_iou_anneal_steps: int = 20000
    soft_iou_aux_weight: float = 0.2

    def __post_init__(self, **kwargs):
        """Finish boundary-config initialization."""
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Reject an unknown candidate pool or a non-positive temperature."""
        if self.candidate_pool not in ("per_query", "shared"):
            raise ValueError(f"candidate_pool must be 'per_query' or 'shared', got {self.candidate_pool!r}")
        if self.pair_temperature <= 0 or self.relation_temperature <= 0 or self.record_temperature <= 0:
            raise ValueError("boundary temperatures must be > 0")


@auto_docstring(checkpoint="fastino/GLiNER2.5-Decide")
@strict
class Gliner2Config(PreTrainedConfig):
    r"""
    encoder_config (`Union[AutoConfig, dict]`, *optional*):
        Config of the text encoder. Built from this config alone, with no download.
    architecture (`str`, *optional*, defaults to `"span"`):
        `"span"` or `"boundary"`.
    max_width (`int`, *optional*, defaults to 8):
        Maximum span width, in words, for the span head.
    counting_layer (`str`, *optional*, defaults to `"count_lstm"`):
        `"count_lstm"` or `"count_lstm_v2"`.
    token_pooling (`str`, *optional*, defaults to `"first"`):
        Which subword of a word is gathered. Published checkpoints use `"first"`.
    max_len (`int`, *optional*, defaults to 2048):
        Maximum token length of one window. Published Decide checkpoints store `null`.
    span_mode (`str`, *optional*, defaults to `"markerV0"`):
        Span representation. Published checkpoints use marker prompts.
    model_name (`str`, *optional*):
        Encoder repository recorded by the original checkpoint.
    span_head (`dict`, *optional*):
        Legacy span-head view (`dropout`, `max_width`, `span_mode`).
    architecture_version (`int`, *optional*):
        Checkpoint architecture version.
    config_version (`int`, *optional*):
        Checkpoint config version.
    boundary_config (`Gliner2BoundaryConfig`, *optional*):
        Boundary-head settings. Required when `architecture="boundary"`.
    classification_temperature (`float`, *optional*, defaults to 1.0):
        Divisor applied to classification logits before the activation.

    Examples:

    ```python
    >>> from transformers import Gliner2Config, Gliner2ForSchemaExtraction

    >>> config = Gliner2Config()
    >>> model = Gliner2ForSchemaExtraction(config)
    ```
    """

    model_type = "gliner2"
    sub_configs_defaults = {
        "encoder_config": SubConfigSpec(config_class=AutoConfig, model_type="deberta-v2"),
        "boundary_config": SubConfigSpec(config_class=Gliner2BoundaryConfig, optional=True),
    }
    has_no_defaults_at_init = False

    encoder_config: dict | PreTrainedConfig | None = None
    architecture: Literal["span", "boundary"] = "span"
    max_width: int = 8
    counting_layer: Literal["count_lstm", "count_lstm_v2"] = "count_lstm"
    token_pooling: Literal["first"] = "first"
    span_mode: Literal["markerV0"] = "markerV0"
    max_len: int | None = 2048
    model_name: str | None = None
    span_head: dict | None = None
    architecture_version: int | None = None
    config_version: int | None = None
    boundary_config: Gliner2BoundaryConfig | dict | None = None
    classification_temperature: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        """Accept published head aliases and reject mixture-of-experts checkpoints."""
        if kwargs.pop("use_moe", False):
            raise ValueError("CountLSTMoE checkpoints are not supported")
        for key in ("model_type", "architectures", "transformers_version", "_attn_implementation_autoset"):
            kwargs.pop(key, None)
        head = kwargs.pop("boundary_head", None)
        if head is not None and self.boundary_config is None:
            self.boundary_config = head
        span_head = self.span_head or {}
        if isinstance(span_head, dict):
            if "max_width" in span_head:
                self.max_width = span_head["max_width"]
            if span_head.get("span_mode"):
                self.span_mode = span_head["span_mode"]
        if self.architecture == "boundary" and self.boundary_config is None:
            self.boundary_config = {}
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Reject an unsupported architecture, span mode, or temperature."""
        if self.architecture not in ("span", "boundary"):
            raise ValueError(f"architecture must be 'span' or 'boundary', got {self.architecture!r}")
        if self.counting_layer not in ("count_lstm", "count_lstm_v2"):
            raise ValueError(f"counting_layer must be 'count_lstm' or 'count_lstm_v2', got {self.counting_layer!r}")
        if self.span_mode != "markerV0":
            raise ValueError(f"span_mode must be 'markerV0', got {self.span_mode!r}")
        if self.token_pooling != "first":
            raise ValueError(f"token_pooling must be 'first', got {self.token_pooling!r}")
        if self.classification_temperature <= 0:
            raise ValueError("classification_temperature must be > 0")
        if self.max_width < 1:
            raise ValueError("max_width must be >= 1")
        if self.max_len is not None and self.max_len < 1:
            raise ValueError("max_len must be null or >= 1")


__all__ = ["Gliner2Config", "Gliner2BoundaryConfig"]
