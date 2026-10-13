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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...decision import DecisionPrompt
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..clip.modeling_clip import CLIPMLP
from ..cohere_asr.modeling_cohere_asr import CohereAsrDecoderLayer
from ..molmo2.modeling_molmo2 import Molmo2VisionAttention


QUESTION_TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}


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
    def __init__(self, config: ClefConfig):
        super().__init__(config)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )


class ClefMLP(CLIPMLP):
    pass


class ClefResidualScorer(CLIPMLP):
    """Scores the features `[field, option, field * option, |field - option|]` of an option."""

    def __init__(self, config: ClefConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, 1)


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
    maxima = logits.new_full((num_segments,), torch.finfo(logits.dtype).min)
    maxima = maxima.scatter_reduce(0, segment_ids, logits, reduce="amax")
    exponentials = (logits - maxima[segment_ids]).exp()
    return exponentials / segment_sum(exponentials, segment_ids, num_segments)[segment_ids]


class ClefEmbeddings(nn.Module):
    """
    The memory of the prompt, the options, the fields of the questions and the lexical prior of the options, from the
    hidden states of the backbone and the output embeddings of the tokens of the marked spans, averaged over each span.
    """

    def __init__(self, config: ClefConfig):
        super().__init__()
        backbone_hidden_size, hidden_size = config.backbone_hidden_size, config.hidden_size
        self.hidden_norm = nn.LayerNorm(backbone_hidden_size)
        self.memory_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.question_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_question_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.global_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_context_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.option_lexical_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPE_IDS), hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lexical_states: torch.Tensor,
        token_positions: torch.Tensor,
        token_span_ids: torch.Tensor,
        option_question_ids: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(sequence_length, backbone_hidden_size)`):
                The hidden states of the backbone.
            lexical_states (`torch.Tensor` of shape `(num_tokens, backbone_hidden_size)`):
                The output embeddings of the tokens of the spans.
            token_positions (`torch.Tensor` of shape `(num_tokens,)`):
                The position of each token of the spans.
            token_span_ids (`torch.Tensor` of shape `(num_tokens,)`):
                The span of each of them: the instructions of the questions, then their options.
            option_question_ids (`torch.Tensor` of shape `(num_options,)`):
                The question of each option.
            type_ids (`torch.Tensor` of shape `(num_questions,)`):
                The type of each question.
        """
        num_questions, num_options = len(type_ids), len(option_question_ids)
        hidden_states = self.hidden_norm(hidden_states)
        global_state = hidden_states[-1]
        lengths = torch.bincount(token_span_ids, minlength=num_questions + num_options)[:, None]
        span_states = segment_sum(hidden_states[token_positions], token_span_ids, len(lengths)) / lengths
        span_lexical = segment_sum(lexical_states, token_span_ids, len(lengths)) / lengths
        questions, contexts = span_states.to(hidden_states.dtype).split([num_questions, num_options])
        lexical = span_lexical[num_questions:].to(hidden_states.dtype)
        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[option_question_ids]
        )
        # The cosine of the output embeddings of each option with its question and the end of the prompt
        anchors = nn.functional.normalize(questions + global_state, dim=-1)[option_question_ids]
        priors = (nn.functional.normalize(lexical, dim=-1) * anchors).sum(-1)
        field_embeddings = self.global_projection(global_state) + self.type_embedding(type_ids)
        memory = self.memory_projection(hidden_states)[None]
        return memory, options[None], self.question_projection(questions), field_embeddings, priors


class ClefEncoder(nn.Module):
    """
    The options attend to the prompt, each question summarizes its options into its field, then the fields attend to
    each other and to the prompt.
    """

    def __init__(self, config: ClefConfig):
        super().__init__()
        self.evidence_layers = nn.ModuleList(ClefEvidenceLayer(config) for _ in range(config.num_evidence_layers))
        self.option_summary_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(ClefDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.field_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        options: torch.Tensor,
        fields: torch.Tensor,
        field_embeddings: torch.Tensor,
        memory: torch.Tensor,
        option_question_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.evidence_layers:
            options = layer(options, encoder_hidden_states=memory, **kwargs)
        options = options[0]

        # The summary of the options of each question, weighted by their similarity with its field
        routing_logits = (options * fields[option_question_ids]).sum(-1) / math.sqrt(options.shape[-1])
        routing_weights = segment_softmax(routing_logits, option_question_ids, len(fields))
        summaries = segment_sum(routing_weights[:, None] * options, option_question_ids, len(fields))
        fields = (fields + self.option_summary_norm(summaries.to(fields.dtype)) + field_embeddings)[None]
        for layer in self.layers:
            fields = layer(fields, encoder_hidden_states=memory, **kwargs)
        return options, self.field_norm(fields[0])


class ClefScorer(nn.Module):
    """Scores each option against the field of its question, on top of its lexical prior."""

    def __init__(self, config: ClefConfig):
        super().__init__()
        self.option_norm = nn.LayerNorm(config.hidden_size)
        self.residual_scorer = ClefResidualScorer(config)
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, fields: torch.Tensor, options: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        options = self.option_norm(options)
        features = torch.cat([fields, options, fields * options, (fields - options).abs()], dim=-1)
        joint = (
            joint_scale * nn.functional.cosine_similarity(fields, options, dim=-1)
            + self.residual_scorer(features)[:, 0]
        )
        return prior_scale * priors + torch.sigmoid(self.residual_gate) * joint


@auto_docstring
class ClefPreTrainedModel(PreTrainedModel):
    config: ClefConfig
    main_input_name = "hidden_states"
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ClefScorer):
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
class ClefDecisionHead(ClefPreTrainedModel):
    def __init__(self, config: ClefConfig):
        super().__init__(config)
        self.embeddings = ClefEmbeddings(config)
        self.encoder = ClefEncoder(config)
        self.scorer = ClefScorer(config)
        self.post_init()

    def spans(self, prompt: DecisionPrompt, device: torch.device) -> tuple[torch.Tensor, ...]:
        """
        The tokens between each pair of marks of the prompt (around the instruction of each question, then around each
        of its options): their positions and the span of each, the instructions first and then the options. Also the
        question of each option and the type of each question.
        """
        option_counts = [len(question.options) for question in prompt.questions]
        num_questions, counts = len(option_counts), torch.tensor(option_counts, device=device)
        option_question_ids = torch.repeat_interleave(torch.arange(num_questions, device=device), counts)
        instruction_spans = torch.arange(num_questions, device=device) + counts.cumsum(0) - counts
        option_spans = torch.arange(sum(option_counts), device=device) + option_question_ids + 1
        spans = torch.tensor(prompt.marks, device=device).view(-1, 2)[torch.cat([instruction_spans, option_spans])]
        lengths = spans[:, 1] - spans[:, 0]
        token_span_ids = torch.repeat_interleave(torch.arange(len(spans), device=device), lengths)
        offsets = torch.arange(len(token_span_ids), device=device) - (lengths.cumsum(0) - lengths)[token_span_ids]
        type_ids = torch.tensor([QUESTION_TYPE_IDS[question.type] for question in prompt.questions], device=device)
        return spans[token_span_ids, 0] + offsets, token_span_ids, option_question_ids, type_ids

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
        token_positions, token_span_ids, option_question_ids, type_ids = self.spans(prompt, hidden_states.device)
        token_ids = torch.tensor(prompt.input_ids, device=hidden_states.device)[token_positions]
        memory, options, fields, field_embeddings, priors = self.embeddings(
            hidden_states, lm_head.weight[token_ids], token_positions, token_span_ids, option_question_ids, type_ids
        )
        options, fields = self.encoder(options, fields, field_embeddings, memory, option_question_ids, **kwargs)
        scores = self.scorer(fields[option_question_ids], options, priors)
        return list(scores.split([len(question.options) for question in prompt.questions]))


__all__ = ["ClefConfig", "ClefDecisionHead", "ClefPreTrainedModel"]
