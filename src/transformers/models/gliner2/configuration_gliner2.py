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
from ...utils import auto_docstring, cached_file
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
    abstention_loss_weight (`float`, *optional*, defaults to `0.2`):
        Value for `abstention_loss_weight`.
    adaptive_threshold (`bool`, *optional*, defaults to `False`):
        Value for `adaptive_threshold`.
    boundary_focal_clip (`float`, *optional*, defaults to `0.05`):
        Value for `boundary_focal_clip`.
    boundary_focal_gamma_negative (`float`, *optional*, defaults to `2.0`):
        Value for `boundary_focal_gamma_negative`.
    boundary_focal_gamma_positive (`float`, *optional*, defaults to `0.0`):
        Value for `boundary_focal_gamma_positive`.
    boundary_marginal_loss (`str`, *optional*, defaults to `"asymmetric_focal"`):
        Value for `boundary_marginal_loss`.
    boundary_negative_weight (`float`, *optional*, defaults to `0.5`):
        Value for `boundary_negative_weight`.
    classification_loss_weight (`float`, *optional*, defaults to `1.0`):
        Value for `classification_loss_weight`.
    classification_temperature (`float`, *optional*, defaults to `1.0`):
        Value for `classification_temperature`.
    consistency_loss_weight (`float`, *optional*, defaults to `0.1`):
        Value for `consistency_loss_weight`.
    consistency_warmup_steps (`int`, *optional*, defaults to `2000`):
        Value for `consistency_warmup_steps`.
    count_loss_weight (`float`, *optional*, defaults to `0.2`):
        Value for `count_loss_weight`.
    hard_negative_keep_all_when_absent (`bool`, *optional*, defaults to `True`):
        Value for `hard_negative_keep_all_when_absent`.
    hard_negatives_per_positive (`int`, *optional*, defaults to `20`):
        Value for `hard_negatives_per_positive`.
    loss_reduction (`str`, *optional*, defaults to `"sum"`):
        Value for `loss_reduction`.
    max_negative_queries_per_batch (`int`, *optional*, defaults to `64`):
        Value for `max_negative_queries_per_batch`.
    minimum_hard_negatives (`int`, *optional*, defaults to `16`):
        Value for `minimum_hard_negatives`.
    negative_query_ratio (`float`, *optional*, defaults to `1.0`):
        Value for `negative_query_ratio`.
    proposal_loss_weight (`float`, *optional*, defaults to `0.3`):
        Value for `proposal_loss_weight`.
    record_loss_weight (`float`, *optional*, defaults to `1.0`):
        Value for `record_loss_weight`.
    relation_loss_weight (`float`, *optional*, defaults to `1.0`):
        Value for `relation_loss_weight`.
    rerank_listwise_weight (`float`, *optional*, defaults to `0.3`):
        Value for `rerank_listwise_weight`.
    soft_iou_anneal_steps (`int`, *optional*, defaults to `20000`):
        Value for `soft_iou_anneal_steps`.
    soft_iou_aux_weight (`float`, *optional*, defaults to `0.2`):
        Value for `soft_iou_aux_weight`.
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
        # Published configs request sdpa. This wrapper has no attention layers.
        if self._attn_implementation not in (None, "eager"):
            self._attn_implementation = "eager"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a published config, including `encoder_config/config.json`."""
        hub = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "proxies",
                "token",
                "local_files_only",
                "revision",
                "subfolder",
            )
            if key in kwargs
        }
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path, **dict(kwargs))
        has_encoder = isinstance(config_dict, dict) and config_dict.get("encoder_config") is not None
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if has_encoder:
            return loaded
        sidecar = _encoder_sidecar(pretrained_model_name_or_path, hub)
        if sidecar is None:
            return loaded
        config, unused = loaded if isinstance(loaded, tuple) else (loaded, None)
        config.encoder_config = AutoConfig.from_pretrained(sidecar, local_files_only=True)
        config.encoder_config._attn_implementation = "eager"
        if isinstance(loaded, tuple):
            return config, unused
        return config

    def to_dict(self) -> dict:
        """Write the Transformers config and the gliner2 checkpoint views."""
        output = super().to_dict()
        if self.boundary_config is not None:
            output["boundary_head"] = self.boundary_config.to_dict()
        if self.architecture == "span":
            output["span_head"] = self.span_head or {
                "dropout": 0.1,
                "max_width": self.max_width,
                "span_mode": self.span_mode,
            }
        return output

    def validate_architecture(self):
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


def _encoder_sidecar(pretrained_model_name_or_path, hub: dict) -> str | None:
    """Resolve the sibling encoder config published next to `config.json`."""
    try:
        return cached_file(
            pretrained_model_name_or_path,
            "encoder_config/config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **hub,
        )
    except OSError:
        return None


__all__ = ["Gliner2Config", "Gliner2BoundaryConfig"]
