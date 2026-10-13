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
from torch import nn

from ...decision_utils import DecisionPrompt
from ...modeling_decision_heads import DecisionHeadMixin
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_clef import ClefConfig


class ClefEvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # The identity keeps the indices of the checkpoint (a dropout in training)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Identity(), nn.Linear(feedforward, width)
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # `memory_norm` twice, as `key is value` would switch torch to a packed key/value projection
        memory_key, memory_value = self.memory_norm(memory), self.memory_norm(memory)
        routed, _ = self.attention(self.query_norm(queries), memory_key, memory_value, need_weights=False)
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


@auto_docstring
class ClefPreTrainedModel(PreTrainedModel):
    config: ClefConfig
    main_input_name = "hidden_states"


@auto_docstring(
    custom_intro="""
    Decision head of clef (Cloudflare/clef): all the options of all the questions are scored jointly. The template marks
    the span of each instruction and then of each of its options with a pair of `<|mark|>`. Options are pooled from the
    hidden states and from the output embeddings of their tokens and routed through the prompt, and the questions attend
    to each other.
    """
)
class ClefDecisionHead(ClefPreTrainedModel, DecisionHeadMixin):
    type_ids = {"noul": 0, "choice": 1, "score": 2}

    def __init__(self, config: ClefConfig):
        super().__init__(config)
        hidden_size, width = config.hidden_size, config.width
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList(
            ClefEvidenceRoutingLayer(width, config.heads, config.feedforward) for _ in range(config.routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                width,
                config.heads,
                config.feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.post_init()

    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None = None
    ) -> list[torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, hidden_size)`):
            Final hidden states of the backbone for one prompt.
        prompt (`DecisionPrompt`):
            The prompt: its input ids, the positions recorded by `<|mark|>` and its questions.
        lm_head (`nn.Module`, *optional*):
            The output embeddings of the backbone.
        """
        spans = list(zip(prompt.marks[::2], prompt.marks[1::2]))
        input_ids = torch.tensor(prompt.input_ids, device=hidden_states.device)
        h = self.hidden_norm(hidden_states)
        memory, global_vector = self.memory_projection(h)[None], h[-1]
        questions, contexts, lexical, start = [], [], [], 0
        for question in prompt.questions:
            questions.append(h[slice(*spans[start])].mean(dim=0))
            option_spans = spans[start + 1 : start + 1 + len(question.options)]
            contexts.append(torch.stack([h[a:b].mean(dim=0) for a, b in option_spans]))
            lexical.append(
                torch.stack([lm_head.weight[input_ids[a:b]].to(h.dtype).mean(dim=0) for a, b in option_spans])
            )
            start += 1 + len(question.options)
        questions = torch.stack(questions)

        options = torch.cat(
            [
                self.option_context_projection(c)
                + self.option_lexical_projection(x)
                + self.option_question_projection(q)[None]
                for c, x, q in zip(contexts, lexical, questions)
            ]
        )[None]
        for layer in self.evidence_layers:
            options = layer(options, memory)
        options = torch.split(options[0], [len(question.options) for question in prompt.questions])

        fields = self.question_projection(questions)
        summaries = [
            (torch.softmax(o @ f / math.sqrt(o.shape[-1]), dim=0)[:, None] * o).sum(dim=0)
            for f, o in zip(fields, options)
        ]
        type_ids = torch.tensor([self.type_ids[question.type] for question in prompt.questions], device=h.device)
        fields = (
            fields + self.option_summary_norm(torch.stack(summaries)) + self.global_projection(global_vector)[None]
        )
        fields = (fields + self.type_embedding(type_ids))[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        scores = []
        for question, field, x, o in zip(questions, fields, lexical, options):
            prior = prior_scale * (F.normalize(x, dim=-1) @ F.normalize(question + global_vector, dim=-1))
            o = self.option_norm(o)
            f = field[None].expand_as(o)
            residual = self.residual_scorer(torch.cat([f, o, f * o, (f - o).abs()], dim=-1))[:, 0]
            scores.append(
                prior
                + torch.sigmoid(self.residual_gate) * (joint_scale * F.cosine_similarity(f, o, dim=-1) + residual)
            )
        return scores


__all__ = ["ClefDecisionHead", "ClefPreTrainedModel"]
