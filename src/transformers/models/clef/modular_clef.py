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
import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...decision import DecisionHeadMixin, DecisionPrompt
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..cohere_asr.modeling_cohere_asr import CohereAsrDecoderLayer
from ..groupvit.modeling_groupvit import GroupViTMLP
from ..molmo2.modeling_molmo2 import Molmo2VisionAttention


@auto_docstring(checkpoint="Cloudflare/clef")
@strict
class ClefConfig(PreTrainedConfig):
    r"""
    backbone_hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden states of the backbone that the head reads.
    num_evidence_layers (`int`, *optional*, defaults to 2):
        Number of layers in which the options attend to the prompt.

    Example:
    ```python
    >>> from transformers import ClefConfig, ClefDecisionHead

    >>> head = ClefDecisionHead(ClefConfig(backbone_hidden_size=4096))
    ```"""

    model_type = "clef"

    backbone_hidden_size: int = 4096
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 4
    num_evidence_layers: int = 2
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    hidden_act: str = "gelu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


class ClefAttention(Molmo2VisionAttention):
    def __init__(self, config: ClefConfig, hidden_size: int | None = None):
        super().__init__(config, hidden_size)


class ClefMLP(GroupViTMLP):
    def __init__(
        self,
        config: ClefConfig,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        output_size: int | None = None,
    ):
        super().__init__(config, hidden_size, intermediate_size, output_size)


class ClefEvidenceLayer(GradientCheckpointingLayer):
    """Pre-norm cross-attention to the memory, which has a norm of its own, then a pre-norm MLP."""

    def __init__(self, config: ClefConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.memory_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_attn = ClefAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = ClefMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.cross_attn(
            self.input_layernorm(hidden_states), self.memory_layernorm(encoder_hidden_states), **kwargs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class ClefDecoderLayer(CohereAsrDecoderLayer):
    def __init__(self, config: ClefConfig):
        super().__init__(config)
        # Not causal: the questions attend to each other
        self.self_attn = ClefAttention(config)
        self.encoder_attn = ClefAttention(config)
        self.mlp = ClefMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            self.post_attention_layernorm(hidden_states), encoder_hidden_states, **kwargs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.final_layernorm(hidden_states))
        return residual + hidden_states


def segment_sum(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sum of the rows of `values` in each segment, accumulated in float32."""
    sums = values.new_zeros((num_segments, *values.shape[1:]), dtype=torch.float32)
    return sums.index_add_(0, segment_ids, values.float())


def segment_softmax(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of `logits` within each segment, in float32."""
    logits = logits.float()
    maxima = logits.new_full((num_segments,), float("-inf")).scatter_reduce(0, segment_ids, logits, reduce="amax")
    exponentials = (logits - maxima[segment_ids]).exp()
    return exponentials / segment_sum(exponentials, segment_ids, num_segments)[segment_ids]


@auto_docstring
class ClefPreTrainedModel(PreTrainedModel):
    config: ClefConfig
    main_input_name = "hidden_states"
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ClefDecisionHead):
            init.zeros_(module.prior_logit_scale)
            init.zeros_(module.joint_logit_scale)
            init.zeros_(module.residual_gate)


@auto_docstring(
    custom_intro="""
    Decision head of clef (Cloudflare/clef): all the options of all the questions are scored jointly. The template marks
    the span of each instruction and then of each of its options with a pair of `<|mark|>`. The options of all the
    questions are packed in one sequence: pooled from the hidden states and from the output embeddings of their tokens,
    they attend to the prompt and are summarized into the field of their question. The fields of the questions then
    attend to each other and to the prompt, and each option is scored against the field of its question.
    """
)
class ClefDecisionHead(ClefPreTrainedModel, DecisionHeadMixin):
    type_ids = {"noul": 0, "choice": 1, "score": 2}

    def __init__(self, config: ClefConfig):
        super().__init__(config)
        backbone_hidden_size, hidden_size = config.backbone_hidden_size, config.hidden_size
        self.hidden_norm = nn.LayerNorm(backbone_hidden_size)
        self.memory_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.question_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_question_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.global_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_context_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_lexical_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.type_embedding = nn.Embedding(len(self.type_ids), hidden_size)
        self.evidence_layers = nn.ModuleList(ClefEvidenceLayer(config) for _ in range(config.num_evidence_layers))
        self.option_summary_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(ClefDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.field_norm = nn.LayerNorm(hidden_size)
        self.option_norm = nn.LayerNorm(hidden_size)
        self.residual_scorer = ClefMLP(config, 4 * hidden_size, hidden_size, 1)
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.post_init()

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt: DecisionPrompt,
        lm_head: nn.Module | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, backbone_hidden_size)`):
            Final hidden states of the backbone for one prompt.
        prompt (`DecisionPrompt`):
            The prompt: its input ids, the positions recorded by `<|mark|>` and its questions.
        lm_head (`nn.Module`, *optional*):
            The output embeddings of the backbone.
        """
        device = hidden_states.device
        option_counts = [len(question.options) for question in prompt.questions]
        num_questions, num_options = len(option_counts), sum(option_counts)
        # The spans of the marks, in template order: the instruction of each question, then each of its options
        starts, ends = torch.tensor(prompt.marks, device=device).view(-1, 2).unbind(-1)
        lengths = ends - starts
        token_span_ids = torch.repeat_interleave(torch.arange(len(lengths), device=device), lengths)
        offsets = torch.arange(len(token_span_ids), device=device) - (lengths.cumsum(0) - lengths)[token_span_ids]
        token_positions = starts[token_span_ids] + offsets
        counts = torch.tensor(option_counts, device=device)
        option_question_ids = torch.repeat_interleave(torch.arange(num_questions, device=device), counts)
        question_spans = torch.arange(num_questions, device=device) + counts.cumsum(0) - counts
        option_spans = torch.arange(num_options, device=device) + option_question_ids + 1

        hidden_states = self.hidden_norm(hidden_states)
        global_state, memory = hidden_states[-1], self.memory_projection(hidden_states)[None]
        token_ids = torch.tensor(prompt.input_ids, device=device)[token_positions]
        span_states = segment_sum(hidden_states[token_positions], token_span_ids, len(lengths)) / lengths[:, None]
        span_lexical = segment_sum(lm_head.weight[token_ids], token_span_ids, len(lengths)) / lengths[:, None]
        questions = span_states[question_spans].to(hidden_states.dtype)
        lexical = span_lexical[option_spans].to(hidden_states.dtype)
        options = (
            self.option_context_projection(span_states[option_spans].to(hidden_states.dtype))
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[option_question_ids]
        )[None]
        for layer in self.evidence_layers:
            options = layer(options, encoder_hidden_states=memory, **kwargs)
        options = options[0]

        fields = self.question_projection(questions)
        routing_logits = (options * fields[option_question_ids]).sum(-1) / math.sqrt(options.shape[-1])
        routing_weights = segment_softmax(routing_logits, option_question_ids, num_questions)
        summaries = segment_sum(routing_weights[:, None] * options, option_question_ids, num_questions)
        type_ids = torch.tensor([self.type_ids[question.type] for question in prompt.questions], device=device)
        fields = (
            fields
            + self.option_summary_norm(summaries.to(fields.dtype))
            + self.global_projection(global_state)
            + self.type_embedding(type_ids)
        )[None]
        for layer in self.layers:
            fields = layer(fields, encoder_hidden_states=memory, **kwargs)
        fields = self.field_norm(fields[0])[option_question_ids]

        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        anchors = F.normalize(questions + global_state, dim=-1)[option_question_ids]
        prior = prior_scale * (F.normalize(lexical, dim=-1) * anchors).sum(-1)
        options = self.option_norm(options)
        features = torch.cat([fields, options, fields * options, (fields - options).abs()], dim=-1)
        joint = joint_scale * F.cosine_similarity(fields, options, dim=-1) + self.residual_scorer(features)[:, 0]
        scores = prior + torch.sigmoid(self.residual_gate) * joint
        return list(scores.split(option_counts))


__all__ = ["ClefConfig", "ClefDecisionHead", "ClefPreTrainedModel"]
