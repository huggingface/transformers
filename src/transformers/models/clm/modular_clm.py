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
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionConfig, DecisionMixin, Question, process_scores, read_marks
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoConfig, AutoModel


# The independent state and candidate texts of `clm.schema.build_pairs`.
CLM_TEMPLATE = r"""
{%- macro text(value, indent=0) -%}
    {%- if value is none -%}
    {%- elif value is string -%}{{ value }}
    {%- elif value is boolean -%}{{ "true" if value else "false" }}
    {%- elif value is mapping or value is sequence -%}
        {%- set ns = namespace(lines=[]) -%}
        {%- if value is mapping -%}
        {%- for key, item in value.items() -%}
            {%- set nested = item and (item is mapping or (item is sequence and item is not string)) -%}
            {%- set ns.lines = ns.lines + [" " * indent ~ key ~ (":\n" ~ text(item, indent + 2) if nested else ": " ~ text(item))] -%}
        {%- endfor -%}
        {%- endif -%}
        {%- if value is not mapping -%}
            {%- for item in value -%}
                {%- set nested = item and (item is mapping or (item is sequence and item is not string)) -%}
                {%- set ns.lines = ns.lines + [" " * indent ~ ("-\n" ~ text(item, indent + 2) if nested else "- " ~ text(item))] -%}
            {%- endfor -%}
        {%- endif -%}
        {{- ns.lines|join("\n\n" if value is mapping and indent == 0 else "\n") -}}
    {%- else -%}{{ value }}
    {%- endif -%}
{%- endmacro -%}
{%- set context, instruction = text(state)|trim, text(question.instructions)|trim -%}
{{- (bos_token or "") ~ context ~ ("\n\n" if context and instruction else "") ~ instruction ~ "<|mark|>" -}}
{%- for option in question.options -%}
    {%- set description = question.descriptions[loop.index0] -%}
    {%- if question.type == "noul" -%}
        {%- if description is none or description == "" -%}
            {%- set description = (("Yes. This is true: " if option == "true" else "No. This is false: ") ~ instruction) if instruction else option -%}
        {%- endif -%}
        {%- set candidate = option ~ ": " ~ text(description) -%}
    {%- elif question.type == "choice" -%}
        {%- set candidate = option if description is none or description == "" else text(description) -%}
    {%- else -%}
        {%- set candidate = text(description) -%}
    {%- endif -%}
    {{- (bos_token or "") ~ candidate ~ "<|mark|>" -}}
{%- endfor -%}
"""


@auto_docstring(checkpoint="Contrastive-LM/CLM-v0.1-8B")
@strict
class ClmConfig(PreTrainedConfig):
    r"""
    projection_dim (`int`, *optional*, defaults to 512):
        Dimension of the normalized state and candidate projections.
    layer_norm (`bool`, *optional*, defaults to `True`):
        Apply layer normalization before the activation of each intermediate projection layer.
    residual (`bool`, *optional*, defaults to `False`):
        Add a residual connection around each intermediate projection layer.
    logit_scale_init_value (`float`, *optional*, defaults to 2.659260036932778):
        Initial log inverse temperature. The inverse temperature is capped at 100 when scoring candidates.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration of the shared text encoder. Defaults to Qwen3.
    """

    model_type = "clm"
    sub_configs_defaults = {"text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3")}

    hidden_size: int = 1536
    projection_dim: int = 512
    num_hidden_layers: int = 3
    hidden_act: str = "gelu"
    layer_norm: bool = True
    residual: bool = False
    logit_scale_init_value: float = 2.659260036932778
    initializer_range: float = 0.02
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be at least two")


class ClmProjectionHead(nn.Module):
    def __init__(self, config: ClmConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.text_config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers - 2)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_size) if config.layer_norm else nn.Identity()
            for _ in range(config.num_hidden_layers - 2)
        )
        self.output_projection = nn.Linear(config.hidden_size, config.projection_dim)
        self.activation = ACT2FN[config.hidden_act]
        self.residual = config.residual

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.input_projection(hidden_states))
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            residual = hidden_states
            hidden_states = self.activation(layer_norm(layer(hidden_states)))
            if self.residual:
                hidden_states = residual + hidden_states
        return nn.functional.normalize(self.output_projection(hidden_states), dim=-1)


@auto_docstring
@dataclass
class ClmDecisionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`):
        Mean of state-to-action and action-to-state InfoNCE losses, returned when `labels` are provided.
    logits (`torch.FloatTensor` of shape `(num_questions, num_candidates)`):
        Scaled cosine similarities between every state and every supplied candidate.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


@auto_docstring
class ClmPreTrainedModel(PreTrainedModel):
    config: ClmConfig
    base_model_prefix = "model"
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["ClmProjectionHead"]
    _keep_in_fp32_modules = ["state_head", "action_head", "logit_scale"]
    _keep_in_fp32_modules_strict = ["state_head", "action_head", "logit_scale"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ClmForDecision):
            init.constant_(module.logit_scale, self.config.logit_scale_init_value)


@auto_docstring(
    custom_intro="""
    CLM encodes the state and each candidate independently with a shared text model. Last-token pooled encoder
    embeddings are normalized and passed through separate state and action projection heads. Scaled cosine
    similarities score the candidates of each question. Each text ends with a `<|mark|>`: first the state and
    question, then each candidate. Unlike a joint prompt, the marked texts are encoded independently.
    """
)
class ClmForDecision(DecisionMixin, ClmPreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: ClmConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.state_head = ClmProjectionHead(config)
        self.action_head = ClmProjectionHead(config)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.post_init()

    @DecisionMixin.decision_config.getter
    def decision_config(self) -> DecisionConfig:
        if getattr(self, "_decision_config", None) is None:
            self._decision_config = DecisionConfig.from_pretrained(self.name_or_path)
            if self._decision_config.chat_template is None:
                self._decision_config.chat_template = CLM_TEMPLATE
                self._decision_config.labels = dict.fromkeys(("choice", "noul", "score"))
                self._decision_config.noul_options = ["false", "true"]
        return self._decision_config

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        group_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ClmDecisionOutput:
        r"""
        labels (`torch.LongTensor` of shape `(num_questions,)` or `torch.FloatTensor` of shape `(num_questions, num_candidates)`, *optional*):
            Matching candidate indices or target distributions over the in-batch candidate pool. The loss is the mean
            of state-to-candidate cross entropy and candidate-to-state cross entropy. The reverse direction includes
            only candidates with positive target mass, normalized across their matching states.
        group_ids (`torch.LongTensor` of shape `(num_questions,)`, *optional*):
            Group IDs for paired training batches with one matching candidate per state in the same order.
            Off-diagonal pairs from the same group are excluded as false negatives in both directions.
        """
        marks, num_options = marks[0], num_options[0]  # a batch of one prompt
        lengths = marks.diff(prepend=marks.new_zeros(1))
        # Independent encoder passes preserve the publisher's reusable candidate representations.
        positions = torch.arange(lengths.max(), device=input_ids.device)[None]
        attention_mask = positions < lengths[:, None]
        indices = (marks - lengths)[:, None] + positions
        input_ids = input_ids[0, indices.clamp(max=input_ids.shape[1] - 1)]
        input_ids = input_ids.masked_fill(~attention_mask, self.config.text_config.pad_token_id or 0)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
        hidden_states = outputs.last_hidden_state[torch.arange(len(lengths), device=input_ids.device), lengths - 1]
        hidden_states = hidden_states.float()
        hidden_states = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-12)
        state_indices = (num_options + 1).cumsum(0) - (num_options + 1)
        is_option = torch.ones(len(lengths), device=input_ids.device, dtype=torch.bool)
        is_option[state_indices] = False
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            states = self.state_head(hidden_states[state_indices].to(self.state_head.input_projection.weight.dtype))
            options = self.action_head(hidden_states[is_option].to(self.action_head.input_projection.weight.dtype))
            scale = self.logit_scale.exp().clamp(max=100.0)
            logits = scale * states @ options.T
            loss = None
            if labels is not None:
                targets = (
                    nn.functional.one_hot(labels, options.shape[0]).float() if labels.ndim == 1 else labels.float()
                )
                loss_logits = logits.float()
                if group_ids is not None:
                    same_group = group_ids[:, None] == group_ids[None, :]
                    same_group.fill_diagonal_(False)
                    loss_logits = loss_logits.masked_fill(same_group, -torch.inf)
                state_log_probs = loss_logits.log_softmax(-1).masked_fill(targets == 0, 0.0)
                state_loss = -(targets * state_log_probs).sum(-1).mean()
                reverse_targets = targets.T
                positive_candidates = reverse_targets.sum(-1) > 0
                reverse_targets = reverse_targets[positive_candidates]
                reverse_targets = reverse_targets / reverse_targets.sum(-1, keepdim=True)
                action_log_probs = loss_logits.T[positive_candidates].log_softmax(-1)
                action_log_probs = action_log_probs.masked_fill(reverse_targets == 0, 0.0)
                action_loss = -(reverse_targets * action_log_probs).sum(-1).mean()
                loss = (state_loss + action_loss) / 2
            return ClmDecisionOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.LongTensor,
        decision_questions: list[Question],
        decision_config: DecisionConfig | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Score each question's candidates with the same decision API as the other decision heads."""
        decision_config = decision_config or self.decision_config
        if any(question.allow_null for question in decision_questions):
            raise ValueError("This decision head does not predict null scores.")
        inputs, marks = read_marks({"input_ids": input_ids, **kwargs}, decision_config._sentinel_ids)
        num_options = torch.tensor([[len(question.options) for question in decision_questions]], device=self.device)
        question_types = [Question.questions_types.index(question.type) for question in decision_questions]
        output = self(
            marks=marks[None].to(self.device),
            num_options=num_options,
            question_type_ids=torch.tensor([question_types], device=self.device),
            **{key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()},
            return_dict=True,
        )
        option_question_ids = torch.repeat_interleave(num_options[0])
        scores = output.logits[option_question_ids, torch.arange(option_question_ids.shape[0], device=self.device)]
        return process_scores(
            decision_config, inputs["input_ids"], marks, decision_questions, scores.split(num_options[0].tolist())
        )


__all__ = ["ClmConfig", "ClmForDecision", "ClmPreTrainedModel"]
