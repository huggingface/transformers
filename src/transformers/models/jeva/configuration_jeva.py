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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


JEVA_TEMPLATE = r"""
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{{- (bos_token or "") ~ "Decision type: " ~ question.type ~ "\nQuestion: " ~ text(question.instructions) ~ "\nState:\n" ~ text(state) ~ "<|mark|>" -}}
{%- for option in question.options -%}
{%- set description = question.descriptions[loop.index0] -%}
{{- (bos_token or "") ~ "Option for " ~ question.type ~ ": " ~ text(option ~ ": " ~ text(description) if question.type == "choice" and description is not none else description if description is not none else option) ~ "<|mark|>" -}}
{%- endfor -%}
"""


@auto_docstring(checkpoint="LIQIIIII/Jeva-4B")
@strict
class JevaConfig(PreTrainedConfig):
    """
    Configuration of a Jeva encoder and matrix decision stack.

    text_config (`PreTrainedConfig` or `dict`, *optional*):
        Configuration of the shared text encoder. Defaults to a Qwen2 configuration.
    projection_dim (`int`, *optional*, defaults to `768`):
        Width of the state and option projections and decision stack.
    decision_depth (`int`, *optional*, defaults to `1`):
        Number of state-update blocks.
    decision_num_heads (`int`, *optional*, defaults to `8`):
        Number of heads in each decision block.
    decision_ffn_ratio (`float`, *optional*, defaults to `4.0`):
        Feed-forward expansion ratio. Set to zero to omit the feed-forward layers.
    decision_dropout (`float`, *optional*, defaults to `0.0`):
        Dropout in the decision feed-forward layers.
    decision_zero_init_ffn (`bool`, *optional*, defaults to `True`):
        Initialize feed-forward output projections to zero.
    decision_final_norm (`bool`, *optional*, defaults to `False`):
        Apply RMS normalization before reading the decision scores.
    normalize_state (`bool`, *optional*, defaults to `False`):
        Normalize the initial state representation.
    pooling (`str`, *optional*, defaults to `"last"`):
        Encoder pooling: `"last"`, `"mean"`, or `"cls"`.
    tau_set_init (`float`, *optional*, defaults to `0.15`):
        Initial state-update attention temperature.
    tau_dec_init (`float`, *optional*, defaults to `0.1`):
        Initial candidate-distribution temperature.
    tau_cov_init (`float`, *optional*, defaults to `0.1`):
        Initial coverage temperature.
    alpha_over_tau_init (`float`, *optional*, defaults to `0.6`):
        Initial ratio controlling the state-update step size.
    coverage_bias_init (`float`, *optional*, defaults to `0.0`):
        Initial coverage threshold.
    min_temperature (`float`, *optional*, defaults to `1e-4`):
        Lower bound of the learned temperatures.
    adapter_config (`dict`, *optional*):
        PEFT configuration for an adapter inside the shared encoder.
    router_aux_loss_coef (`float`, *optional*, defaults to `0.0`):
        Weight of the MoE load-balancing loss. Zero disables router-logit collection.
    """

    model_type = "jeva"
    sub_configs_defaults = {"text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen2")}

    text_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 768
    decision_depth: int = 1
    decision_num_heads: int = 8
    decision_ffn_ratio: float = 4.0
    decision_dropout: float = 0.0
    decision_zero_init_ffn: bool = True
    decision_final_norm: bool = False
    normalize_state: bool = False
    pooling: str = "last"
    tau_set_init: float = 0.15
    tau_dec_init: float = 0.1
    tau_cov_init: float = 0.1
    alpha_over_tau_init: float = 0.6
    coverage_bias_init: float = 0.0
    min_temperature: float = 1e-4
    adapter_config: dict | None = None
    router_aux_loss_coef: float = 0.0

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, PreTrainedConfig):
            kwargs.setdefault("attn_implementation", self.text_config._attn_implementation)
        super().__post_init__(**kwargs)
        if self.decision_depth < 1 or self.decision_num_heads < 1 or self.projection_dim % self.decision_num_heads:
            raise ValueError("decision_depth must be positive and decision_num_heads must divide projection_dim")
        if self.pooling not in {"last", "mean", "cls"}:
            raise ValueError("pooling must be last, mean, or cls")
        self.base_model_tp_plan = {
            f"model.{key}": value for key, value in (self.text_config.base_model_tp_plan or {}).items()
        }
        self.base_model_ep_plan = {
            f"model.{key}": value for key, value in (self.text_config.base_model_ep_plan or {}).items()
        }
        if self.adapter_config is not None:
            self.adapter_config = {
                key: list(value) if isinstance(value, set) else value for key, value in self.adapter_config.items()
            }


__all__ = ["JevaConfig"]
