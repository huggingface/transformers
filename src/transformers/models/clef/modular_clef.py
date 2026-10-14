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
        self.type_embedding = nn.Embedding(len(QUESTION_TYPE_IDS), hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        device, dtype = hidden_states.device, hidden_states.dtype
        num_options = torch.tensor([len(question.options) for question in prompt.questions], device=device)
        type_ids = torch.tensor([QUESTION_TYPE_IDS[question.type] for question in prompt.questions], device=device)
        # Each question marks the span of its instruction, then of each of its options
        is_instruction = torch.tensor([span == 0 for count in num_options for span in range(count + 1)], device=device)
        spans = list(zip(prompt.marks[::2], prompt.marks[1::2]))
        input_ids = torch.tensor([prompt.input_ids], device=device)

        hidden_states = self.hidden_norm(hidden_states)
        span_states = torch.stack([hidden_states[:, start:end].float().mean(1) for start, end in spans], dim=1)
        span_lexical = torch.stack(
            [lm_head.weight[input_ids[:, start:end]].float().mean(1) for start, end in spans], dim=1
        )
        questions, contexts = span_states[:, is_instruction].to(dtype), span_states[:, ~is_instruction].to(dtype)
        lexical = span_lexical[:, ~is_instruction].to(dtype)
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
        field_embeddings = self.global_projection(global_state) + self.type_embedding(type_ids)
        return self.memory_projection(hidden_states), options, (fields, field_embeddings), priors


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
        fields: tuple[torch.Tensor, torch.Tensor],
        option_question_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`fields` are the projected questions, which route the options, and the embeddings added to them after."""
        fields, field_embeddings = fields
        for layer in self.evidence_layers:
            options = layer(options, memory, **kwargs)

        # Each question summarizes its options, weighted by their similarity with its field (a product rather than a
        # matmul, which rounds differently in half precision)
        routing_logits = (fields[:, :, None] * options[:, None]).sum(-1) / math.sqrt(fields.shape[-1])
        question_ids = torch.arange(fields.shape[1], device=fields.device)
        routing_logits[:, question_ids[:, None] != option_question_ids] = torch.finfo(fields.dtype).min
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
        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
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
        num_options = [len(question.options) for question in prompt.questions]
        question_ids = torch.arange(len(num_options), device=hidden_states.device)
        option_question_ids = question_ids.repeat_interleave(torch.tensor(num_options, device=hidden_states.device))
        memory, options, fields, priors = self.embeddings(hidden_states[None], prompt, lm_head)
        options, fields = self.encoder(memory, options, fields, option_question_ids, **kwargs)
        scores = self.scorer(fields[:, option_question_ids], options, priors)
        return list(scores[0].split(num_options))


__all__ = ["ClefConfig", "ClefDecisionHead", "ClefPreTrainedModel"]
