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
        Value for `boundary_dim`.
    pair_dim (`int`, *optional*, defaults to `128`):
        Value for `pair_dim`.
    boundary_refinement_layers (`int`, *optional*, defaults to `1`):
        Value for `boundary_refinement_layers`.
    boundary_ffn_multiplier (`float`, *optional*, defaults to `2.0`):
        Value for `boundary_ffn_multiplier`.
    start_top_k (`int`, *optional*, defaults to `16`):
        Value for `start_top_k`.
    end_top_k (`int`, *optional*, defaults to `16`):
        Value for `end_top_k`.
    ends_per_start (`int`, *optional*, defaults to `8`):
        Value for `ends_per_start`.
    starts_per_end (`int`, *optional*, defaults to `8`):
        Value for `starts_per_end`.
    candidate_budget (`int`, *optional*, defaults to `128`):
        Value for `candidate_budget`.
    training_candidate_budget (`int`, *optional*, defaults to `160`):
        Value for `training_candidate_budget`.
    max_gold_per_query (`int`, *optional*, defaults to `32`):
        Value for `max_gold_per_query`.
    end_block_size (`int`, *optional*, defaults to `256`):
        Value for `end_block_size`.
    bidirectional_proposals (`bool`, *optional*, defaults to `True`):
        Value for `bidirectional_proposals`.
    use_inside_evidence (`bool`, *optional*, defaults to `True`):
        Value for `use_inside_evidence`.
    dropout (`float`, *optional*, defaults to `0.1`):
        Value for `dropout`.
    export_mode (`str`, *optional*, defaults to `'auto'`):
        Value for `export_mode`.
    vectorized_pair_elements (`int`, *optional*, defaults to `16777216`):
        Value for `vectorized_pair_elements`.
    enable_span_content (`bool`, *optional*, defaults to `False`):
        Value for `enable_span_content`.
    content_dim (`int`, *optional*, defaults to `64`):
        Value for `content_dim`.
    content_soft_max_pool (`bool`, *optional*, defaults to `False`):
        Value for `content_soft_max_pool`.
    enable_rotary_endpoints (`bool`, *optional*, defaults to `False`):
        Value for `enable_rotary_endpoints`.
    rotary_base (`float`, *optional*, defaults to `10000.0`):
        Value for `rotary_base`.
    boundary_attention_layers (`int`, *optional*, defaults to `0`):
        Value for `boundary_attention_layers`.
    boundary_attention_heads (`int`, *optional*, defaults to `4`):
        Value for `boundary_attention_heads`.
    boundary_attention_window (`int`, *optional*, defaults to `0`):
        Value for `boundary_attention_window`.
    query_conditioned_inside_weight (`bool`, *optional*, defaults to `False`):
        Value for `query_conditioned_inside_weight`.
    endpoint_difference_features (`bool`, *optional*, defaults to `False`):
        Value for `endpoint_difference_features`.
    reranker_endpoint_compat (`bool`, *optional*, defaults to `True`):
        Value for `reranker_endpoint_compat`.
    multihead_pair_compat_heads (`int`, *optional*, defaults to `8`):
        Value for `multihead_pair_compat_heads`.
    boundary_top_k_alpha (`float`, *optional*, defaults to `0.0`):
        Value for `boundary_top_k_alpha`.
    boundary_top_k_max (`int`, *optional*, defaults to `128`):
        Value for `boundary_top_k_max`.
    boundary_top_k_bucket (`int`, *optional*, defaults to `8`):
        Value for `boundary_top_k_bucket`.
    candidate_pool (`Literal['per_query', 'shared']`, *optional*, defaults to `'per_query'`):
        Value for `candidate_pool`.
    pool_boundary_top_k (`int`, *optional*, defaults to `64`):
        Value for `pool_boundary_top_k`.
    pool_size (`int`, *optional*, defaults to `384`):
        Value for `pool_size`.
    min_pool_per_query (`int`, *optional*, defaults to `8`):
        Value for `min_pool_per_query`.
    candidate_attention_layers (`int`, *optional*, defaults to `2`):
        Value for `candidate_attention_layers`.
    candidate_attention_heads (`int`, *optional*, defaults to `4`):
        Value for `candidate_attention_heads`.
    query_attention_layers (`int`, *optional*, defaults to `1`):
        Value for `query_attention_layers`.
    enable_abstention (`bool`, *optional*, defaults to `True`):
        Value for `enable_abstention`.
    enable_count_head (`bool`, *optional*, defaults to `True`):
        Value for `enable_count_head`.
    enable_records (`bool`, *optional*, defaults to `False`):
        Value for `enable_records`.
    enable_relations (`bool`, *optional*, defaults to `False`):
        Value for `enable_relations`.
    record_dim (`int`, *optional*, defaults to `128`):
        Value for `record_dim`.
    record_instance_queries (`int`, *optional*, defaults to `8`):
        Value for `record_instance_queries`.
    relation_heads_per_type (`int`, *optional*, defaults to `32`):
        Value for `relation_heads_per_type`.
    relation_tails_per_type (`int`, *optional*, defaults to `32`):
        Value for `relation_tails_per_type`.
    relation_pair_cap (`int`, *optional*, defaults to `128`):
        Value for `relation_pair_cap`.
    relation_argument_proposal_threshold (`float`, *optional*, defaults to `0.0`):
        Value for `relation_argument_proposal_threshold`.
    directional_relation_states (`bool`, *optional*, defaults to `False`):
        Value for `directional_relation_states`.
    relation_biaffine_content (`bool`, *optional*, defaults to `False`):
        Value for `relation_biaffine_content`.
    pair_temperature (`float`, *optional*, defaults to `1.0`):
        Value for `pair_temperature`.
    relation_temperature (`float`, *optional*, defaults to `1.0`):
        Value for `relation_temperature`.
    record_temperature (`float`, *optional*, defaults to `1.0`):
        Value for `record_temperature`.
    overlap_policy (`str`, *optional*, defaults to `'flat'`):
        Value for `overlap_policy`.
    abstention_threshold (`float`, *optional*, defaults to `0.5`):
        Value for `abstention_threshold`.
    record_anchor_proposal_threshold (`float`, *optional*, defaults to `0.5`):
        Value for `record_anchor_proposal_threshold`.
    record_anchor_threshold (`float`, *optional*, defaults to `0.5`):
        Value for `record_anchor_threshold`.
    record_field_threshold (`float`, *optional*, defaults to `0.5`):
        Value for `record_field_threshold`.
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

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    def validate_architecture(self):
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
        Maximum token length of one window.
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
    token_pooling: Literal["first", "mean", "max"] = "first"
    max_len: int = 2048
    boundary_config: Gliner2BoundaryConfig | dict | None = None
    classification_temperature: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.architecture == "boundary" and self.boundary_config is None:
            self.boundary_config = {}
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.architecture not in ("span", "boundary"):
            raise ValueError(f"architecture must be 'span' or 'boundary', got {self.architecture!r}")
        if self.counting_layer not in ("count_lstm", "count_lstm_v2"):
            raise ValueError(f"counting_layer must be 'count_lstm' or 'count_lstm_v2', got {self.counting_layer!r}")
        if self.classification_temperature <= 0:
            raise ValueError("classification_temperature must be > 0")
        if self.max_width < 1:
            raise ValueError("max_width must be >= 1")


__all__ = ["Gliner2Config", "Gliner2BoundaryConfig"]
