# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Privacy Filter model configuration."""

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..gpt_oss.configuration_gpt_oss import GptOssConfig


PRIVACY_FILTER_SPAN_LABELS = (
    "O",
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)

PRIVACY_FILTER_NER_LABELS = ("O",) + tuple(
    f"{prefix}-{label}" for label in PRIVACY_FILTER_SPAN_LABELS if label != "O" for prefix in ("B", "I", "E", "S")
)


@auto_docstring(checkpoint="openai/privacy-filter")
@strict(accept_kwargs=True)
class PrivacyFilterConfig(GptOssConfig):
    r"""
    This is the configuration class to store the configuration of an [`PrivacyFilterModel`].

    Args:
        bidirectional_left_context (`int`, *optional*, defaults to 128):
            Number of tokens to the left visible to each query token.
        bidirectional_right_context (`int`, *optional*, defaults to 128):
            Number of tokens to the right visible to each query token.
        initial_context_length (`int`, *optional*, defaults to 4096):
            Original context length used for YaRN rotary embedding parameters.
        default_n_ctx (`int`, *optional*, defaults to 128000):
            Default runtime context length metadata saved with converted checkpoints.
        rope_theta (`float`, *optional*, defaults to 150000.0):
            Base period for rotary embeddings.
    """

    model_type = "privacy_filter"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_ep_plan = {}

    vocab_size: int = 200064
    hidden_size: int = 640
    intermediate_size: int = 640
    num_hidden_layers: int = 8
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    head_dim: int = 64
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 257
    bidirectional_left_context: int = 128
    bidirectional_right_context: int = 128
    initial_context_length: int = 4096
    max_position_embeddings: int = 131072
    default_n_ctx: int = 128000
    rope_theta: float = 150000.0
    rope_parameters: dict | None = None
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    classifier_dropout: float = 0.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0
    use_cache: bool = False
    layer_types: list[str] | None = None
    pad_token_id: int | None = 199999
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 199999
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": self.rope_theta,
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": self.initial_context_length,
            }
        else:
            self.rope_parameters.setdefault("rope_type", "yarn")
            self.rope_parameters.setdefault("rope_theta", self.rope_theta)
            if self.rope_parameters["rope_type"] in {"llama3", "longrope", "yarn"}:
                self.rope_parameters.setdefault("original_max_position_embeddings", self.initial_context_length)

        requested_num_labels = kwargs.pop("num_labels", len(PRIVACY_FILTER_NER_LABELS))
        if self.id2label is None and requested_num_labels == len(PRIVACY_FILTER_NER_LABELS):
            self.id2label = dict(enumerate(PRIVACY_FILTER_NER_LABELS))
        elif self.id2label is None:
            self.num_labels = requested_num_labels
        if self.label2id is None and self.id2label is not None:
            self.label2id = {label: idx for idx, label in self.id2label.items()}

        super().__post_init__(**kwargs)


__all__ = ["PRIVACY_FILTER_NER_LABELS", "PRIVACY_FILTER_SPAN_LABELS", "PrivacyFilterConfig"]
