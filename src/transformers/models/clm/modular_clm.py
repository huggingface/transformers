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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionConfig, DecisionMixin, Question, process_scores
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..auto import AutoConfig, AutoModel


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
    extra_config_files: ClassVar[tuple[str, ...]] = ()
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
    loss (`torch.FloatTensor` of shape `()`):
        Mean of state-to-action and action-to-state InfoNCE losses, returned when `labels` are provided.
    logits (`torch.FloatTensor` of shape `(num_questions, num_candidates)`):
        Scaled cosine similarities between every state and every supplied candidate.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


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
    # Trainer averages microbatch losses. Its automatic item count would count soft target entries.
    accepts_loss_kwargs = False

    def __init__(self, config: ClmConfig, model: PreTrainedModel | None = None):
        r"""
        model (`PreTrainedModel`, *optional*):
            An already loaded text encoder. Otherwise, initialize it from `config.text_config`.
        """
        super().__init__(config)
        self.model = model if model is not None else AutoModel.from_config(config.text_config)
        self.state_head = ClmProjectionHead(config)
        self.action_head = ClmProjectionHead(config)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.post_init()

    def post_init(self):
        super().post_init()
        self.decision_config = DecisionConfig(
            chat_template=CLM_TEMPLATE,
            labels=dict.fromkeys(("choice", "noul", "score")),
            noul_options=["false", "true"],
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        question_type_ids: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ClmDecisionOutput:
        r"""
        labels (`torch.LongTensor` of shape `(num_questions,)` or `torch.FloatTensor` of shape `(num_questions, num_candidates)`, *optional*):
            Matching candidate indices or target distributions over the in-batch candidate pool. The loss is the mean
            of state-to-candidate cross entropy and candidate-to-state cross entropy. The reverse direction includes
            only candidates with positive target mass, normalized across their matching states.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("CLM expects one packed prompt; pack multiple state-candidate pairs into that prompt.")
        marks, num_options = marks[0], num_options[0]  # a batch of one prompt
        lengths = marks.diff(prepend=marks.new_zeros(1))
        # Independent encoder passes preserve the publisher's reusable candidate representations.
        positions = torch.arange(lengths.max(), device=input_ids.device)[None]
        encoder_attention_mask = positions < lengths[:, None]
        indices = ((marks - lengths)[:, None] + positions).clamp(max=input_ids.shape[1] - 1)
        if attention_mask is not None:
            encoder_attention_mask &= attention_mask[0, indices].bool()
        input_ids = input_ids[0, indices]
        input_ids = input_ids.masked_fill(~encoder_attention_mask, self.config.text_config.pad_token_id or 0)
        outputs = self.model(input_ids=input_ids, attention_mask=encoder_attention_mask, use_cache=False, **kwargs)
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
            loss = self.loss_function(logits=logits, labels=labels, **kwargs)
        return ClmDecisionOutput(loss=loss, logits=logits)

    def prepare_decision_inputs(
        self,
        prompts: Sequence[tuple[dict, list[Question]]],
        *,
        targets: Sequence[Sequence[float]] | None = None,
        max_state_length: int | None = None,
        max_option_length: int | None = None,
        truncation: str = "error",
    ) -> dict:
        """Pack rendered CLM prompts into a single model input.

        Args:
            prompts (`Sequence[tuple[dict, list[Question]]]`):
                One marked state and its marked options per prompt, from [`DecisionConfig.prepare_inputs`].
            targets (`Sequence[Sequence[float]]`, *optional*):
                One target distribution per question, with zero null probability first. Supplying targets builds a
                pool of distinct candidates for the batch and returns labels and a mask of each question's options.
            max_state_length (`int`, *optional*):
                Maximum length of each state span, in tokens.
            max_option_length (`int`, *optional*):
                Maximum length of each option span, in tokens.
            truncation (`str`, *optional*, defaults to `"error"`):
                Use `"left"` or `"right"` to truncate overflowing spans on that side.
        """
        spans = self._split_decision_spans(prompts, max_state_length, max_option_length, truncation)
        segments, counts = [], []
        pool, option_columns = {}, []
        for state, options, question in spans:
            segments.append(state)
            if targets is None:
                segments.extend(options)
                counts.append(len(options))
                continue

            if question.allow_null:
                raise ValueError("CLM does not support nullable decisions.")
            start = len(pool)
            columns = []
            for option in options:
                key = tuple(option.tolist())
                if key not in pool:
                    pool[key] = len(pool)
                    segments.append(option)
                columns.append(pool[key])
            counts.append(len(pool) - start)
            option_columns.append(columns)

        input_ids = torch.cat(segments)[None]
        batch = {
            "input_ids": input_ids,
            "marks": input_ids.new_tensor([[len(segment) for segment in segments]]).cumsum(-1),
            "num_options": input_ids.new_tensor([counts]),
        }
        if targets is not None:
            labels = input_ids.new_zeros((len(spans), len(pool)), dtype=torch.float32)
            option_mask = torch.zeros_like(labels, dtype=torch.bool)
            for row, (target, columns) in enumerate(zip(targets, option_columns, strict=True)):
                if len(target) != len(columns) + 1 or target[0] != 0:
                    raise ValueError("Expected zero null probability followed by one probability per option.")
                indices = input_ids.new_tensor(columns)
                labels[row].scatter_add_(0, indices, labels.new_tensor(target[1:]))
                option_mask[row, indices] = True
            batch.update(labels=labels, option_mask=option_mask)
        return batch

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        decision_questions: list[Question],
        decision_config: DecisionConfig | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Score each question's candidates with the same decision API as the other decision heads."""
        decision_config = decision_config or self.decision_config
        # Override any caller preference because we read `.logits` below.
        kwargs["return_dict"] = True
        inputs = {"input_ids": input_ids, **kwargs}
        num_options = torch.tensor([[len(question.options) for question in decision_questions]], device=self.device)
        question_types = [Question.questions_types.index(question.type) for question in decision_questions]
        output = self(
            marks=marks[None].to(self.device),
            num_options=num_options,
            question_type_ids=torch.tensor([question_types], device=self.device),
            **{key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()},
        )
        option_question_ids = torch.repeat_interleave(num_options[0])
        scores = output.logits[option_question_ids, torch.arange(option_question_ids.shape[0], device=self.device)]
        return process_scores(
            decision_config, inputs["input_ids"], marks, decision_questions, scores.split(num_options[0].tolist())
        )


__all__ = ["ClmConfig", "ClmForDecision", "ClmPreTrainedModel"]
