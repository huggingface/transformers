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
from ...decision_utils import DecisionPrompt
from ...modeling_decision_heads import DecisionHeadMixin
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..d_fine.modeling_d_fine import DFineMLP
from ..mbart.modeling_mbart import MBartAttention, MBartDecoderLayer


@auto_docstring(checkpoint="Cloudflare/clef")
@strict
class ClefConfig(PreTrainedConfig):
    r"""
    width (`int`, *optional*, defaults to 1024):
        Width of the head.
    routing_layers (`int`, *optional*, defaults to 2):
        Number of layers routing the options through the prompt.
    layers (`int`, *optional*, defaults to 4):
        Number of decoder layers over the questions.
    heads (`int`, *optional*, defaults to 16):
        Number of attention heads of the head.
    feedforward (`int`, *optional*, defaults to 4096):
        Dimension of the feed-forward layers of the head.

    Example:
    ```python
    >>> from transformers import ClefConfig, ClefDecisionHead

    >>> head = ClefDecisionHead(ClefConfig(hidden_size=4096))
    ```"""

    model_type = "clef"
    # The names read by the MBart decoder layers
    attribute_map = {
        "d_model": "width",
        "decoder_attention_heads": "heads",
        "decoder_ffn_dim": "feedforward",
        "attention_dropout": "dropout",
        "activation_dropout": "dropout",
    }

    hidden_size: int = 4096
    width: int = 1024
    routing_layers: int = 2
    layers: int = 4
    heads: int = 16
    feedforward: int = 4096
    activation_function: str = "gelu"
    dropout: float = 0.0
    initializer_range: float = 0.02


class ClefAttention(MBartAttention):
    pass


class ClefMLP(DFineMLP):
    pass


class ClefDecoderLayer(MBartDecoderLayer):
    def __init__(self, config: ClefConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        # Not causal: the questions attend to each other
        self.self_attn = ClefAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )


class ClefEvidenceRouter(MBartDecoderLayer):
    """An MBart decoder layer without self-attention, whose memory has a norm of its own."""

    def __init__(self, config: ClefConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.memory_norm = nn.LayerNorm(self.embed_dim)
        del self.self_attn
        del self.self_attn_layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): the options, of shape `(batch, num_options, embed_dim)`
            encoder_hidden_states (`torch.FloatTensor`): the memory, of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            hidden_states=self.encoder_attn_layer_norm(hidden_states),
            key_value_states=self.memory_norm(encoder_hidden_states),
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return residual + hidden_states


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
        self.type_embedding = nn.Embedding(len(self.type_ids), width)
        self.evidence_layers = nn.ModuleList(
            ClefEvidenceRouter(config, layer_idx) for layer_idx in range(config.routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(ClefDecoderLayer(config, layer_idx) for layer_idx in range(config.layers))
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = ClefMLP(4 * width, width, 1, num_layers=2, act=config.activation_function)
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
            options = layer(options, encoder_hidden_states=memory, **kwargs)
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
            fields = layer(fields, encoder_hidden_states=memory, **kwargs)
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


__all__ = ["ClefConfig", "ClefDecisionHead", "ClefPreTrainedModel"]
