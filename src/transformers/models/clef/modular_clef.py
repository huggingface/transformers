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
from ...decision import ForDecision, Question
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..clip.modeling_clip import CLIPMLP
from ..cohere_asr.modeling_cohere_asr import CohereAsrDecoderLayer
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


class ClefEmbeddings(nn.Module):
    """
    The memory of the prompt, the options, the fields of the questions with their embeddings, and the lexical prior of
    the options, from the hidden states of the backbone and the output embeddings of the tokens of the marked spans,
    averaged over each span.
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
        self.type_embedding = nn.Embedding(len(Question.questions_types), hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        output_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        # Each question marks the span of its instruction, then of each of its options
        spans_per_question = num_options + 1
        instructions = spans_per_question.cumsum(0) - spans_per_question
        is_option = torch.ones_like(marks[::2], dtype=torch.bool).index_fill(0, instructions, False)
        # The marks cut the prompt into segments, every other one is a span
        boundaries = torch.cat([marks.new_zeros(1), marks, marks.new_full((1,), input_ids.shape[-1])])
        segment_sizes = boundaries.diff().tolist()

        hidden_states = self.hidden_norm(hidden_states)
        span_states = [segment.float().mean(1) for segment in torch.split(hidden_states, segment_sizes, dim=1)[1::2]]
        span_lexical = [
            output_embeddings[segment].float().mean(1)
            for segment in torch.split(input_ids, segment_sizes, dim=1)[1::2]
        ]
        span_states, span_lexical = torch.stack(span_states, dim=1), torch.stack(span_lexical, dim=1)
        questions, contexts = span_states[:, instructions].to(dtype), span_states[:, is_option].to(dtype)
        lexical = span_lexical[:, is_option].to(dtype)
        global_state = hidden_states[:, -1:]
        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions).repeat_interleave(num_options, dim=1)
        )
        # The cosine of the output embeddings of each option with its question and the end of the prompt
        anchors = nn.functional.normalize(questions + global_state, dim=-1).repeat_interleave(num_options, dim=1)
        priors = (nn.functional.normalize(lexical, dim=-1) * anchors).sum(-1)
        fields = self.question_projection(questions)
        field_embeddings = self.global_projection(global_state) + self.type_embedding(question_type_ids)
        return self.memory_projection(hidden_states), options, fields, field_embeddings, priors


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
        memory: torch.Tensor,
        options: torch.Tensor,
        fields: torch.Tensor,
        field_embeddings: torch.Tensor,
        option_question_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`fields` are the projected questions, which route the options; `field_embeddings` are added to them after."""
        for layer in self.evidence_layers:
            options = layer(options, memory, **kwargs)

        routing_logits = (fields[:, :, None] * options[:, None]).sum(-1) / math.sqrt(fields.shape[-1])
        is_other_option = nn.functional.one_hot(option_question_ids, fields.shape[1]).T == 0
        routing_logits[:, is_other_option] = torch.finfo(fields.dtype).min
        summaries = routing_logits.softmax(-1, dtype=torch.float32) @ options.float()
        fields = fields + self.option_summary_norm(summaries.to(fields.dtype)) + field_embeddings
        for layer in self.layers:
            fields = layer(fields, encoder_hidden_states=memory, **kwargs)
        return options, self.field_norm(fields)


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
        logit_scales = torch.stack([self.prior_logit_scale, self.joint_logit_scale])
        prior_scale, joint_scale = logit_scales.clamp(max=math.log(100.0)).exp()
        options = self.option_norm(options)
        features = torch.cat([fields, options, fields * options, (fields - options).abs()], dim=-1)
        joint = (
            joint_scale * nn.functional.cosine_similarity(fields, options, dim=-1)
            + self.residual_scorer(features)[..., 0]
        )
        return prior_scale * priors + torch.sigmoid(self.residual_gate) * joint


@auto_docstring
class ClefPreTrainedModel(PreTrainedModel):
    config: ClefConfig
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

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        backbone: PreTrainedModel,
        backbone_kwargs: dict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        outputs = backbone.base_model(input_ids=input_ids[None], use_cache=False, **(backbone_kwargs or {}))
        hidden_states = outputs.last_hidden_state.to(self.dtype)
        # The output embeddings of the tokens of each option give its lexical prior
        output_embeddings = backbone.get_output_embeddings().weight
        option_question_ids = torch.repeat_interleave(num_options)
        memory, options, fields, field_embeddings, priors = self.embeddings(
            hidden_states, input_ids[None], marks, num_options, question_type_ids, output_embeddings
        )
        options, fields = self.encoder(memory, options, fields, field_embeddings, option_question_ids, **kwargs)
        scores = self.scorer(fields[:, option_question_ids], options, priors)
        return scores[0]


class ClefForDecision(ForDecision):
    """Clef: the model of a repo and its [`ClefDecisionHead`]."""

    config: ClefConfig


__all__ = ["ClefConfig", "ClefDecisionHead", "ClefForDecision", "ClefPreTrainedModel"]
