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
import json
import math
from pathlib import Path

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionMixin, Question
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, cached_file
from ..auto import AutoConfig, AutoModel
from ..clip.modeling_clip import CLIPMLP
from ..cohere_asr.modeling_cohere_asr import CohereAsrDecoderLayer
from ..molmo2.modeling_molmo2 import Molmo2VisionAttention


# The prompt of clef (`joint_schema_model.encode_record`), for its checkpoints: one sequence per request
CLEF_TEMPLATE = r"""{#- Cloudflare/clef, clef-flash (joint_schema_model.encode_record): ONE sequence per request.
    Every text piece the original tokenizes on its own sits between sentinels; <|mark|> pairs give the spans the joint head
    pools: per question (instruction start, end) then (option start, end) for each option in question.options order
    (decision_config: noul_options = ["true", "false"], sort_choices = true). -#}
{%- macro r(x) %}{{ x if x is string else x|tojson(sort_keys=true, separators=(",", ":")) }}{% endmacro -%}
{{- "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n<|split|>" ~ r(state) ~ "<|split|>\n\nSCHEMA FIELDS:\n" -}}
{%- for q in questions -%}
{{- "<|split|>\nFIELD " ~ loop.index ~ "\nID: " ~ q.name ~ "\nTYPE: " ~ q.type ~ "\nINSTRUCTION: <|mark|>" ~ r(q.name if q.instructions is none or q.instructions == "" else q.instructions) ~ "<|mark|>\nALLOWED OPTIONS:\n" -}}
{%- for key in q.options -%}
{%- set d = q.descriptions[loop.index0] -%}
{%- if q.type == "noul" and d is none %}{% set d = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}[key] %}{% endif -%}
{{- "OPTION " ~ loop.index ~ ": <|mark|>" ~ r({"option_id": key} if d is none else {"option_id": key, "description": d}) ~ "<|mark|>\n" -}}
{%- endfor -%}
{{- "END FIELD\n" -}}
{%- endfor -%}
{{- "<|split|>\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:" -}}"""


def _read_json(pretrained_model_name_or_path, filename: str, **kwargs) -> dict | None:
    path = cached_file(pretrained_model_name_or_path, filename, _raise_exceptions_for_missing_entries=False, **kwargs)
    return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


@auto_docstring(checkpoint="Cloudflare/clef")
@strict
class ClefConfig(PreTrainedConfig):
    r"""
    backbone_hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden states of the backbone that the head reads.
    num_evidence_layers (`int`, *optional*, defaults to 2):
        Number of layers in which the options attend to the prompt.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Config of the model the head reads, the `config.json` of its repo.

    Example:
    ```python
    >>> from transformers import ClefConfig, ClefForDecision

    >>> model = ClefForDecision(ClefConfig(backbone_hidden_size=4096))
    ```"""

    model_type = "clef"
    sub_configs_defaults = {"text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3_5_text")}

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
    text_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = False

    # The files of a clef checkpoint that hold its head next to its weights, read with them: its weights and its config
    extra_files = ("joint_head.safetensors",)
    extra_config_files = ("joint_head_config.json",)

    @classmethod
    def decision_config(cls, pretrained_model_name_or_path, **kwargs):
        """The decision config of a clef checkpoint, or `None` for any other checkpoint."""
        from ...decision import DecisionConfig

        if _read_json(pretrained_model_name_or_path, "joint_head_config.json", **kwargs) is None:
            return None
        return DecisionConfig(noul_options=["true", "false"], sort_choices=True, chat_template=CLEF_TEMPLATE)

    def __post_init__(self, joint_head_config: dict | None = None, **kwargs):
        if joint_head_config:  # the config of the head of a clef checkpoint, whose config is `text_config`
            self.backbone_hidden_size, self.hidden_size = joint_head_config["hidden_size"], joint_head_config["width"]
            self.num_evidence_layers, self.num_hidden_layers = (
                joint_head_config["routing_layers"],
                joint_head_config["layers"],
            )
            self.num_attention_heads = self.num_key_value_heads = joint_head_config["heads"]
            self.intermediate_size = joint_head_config["feedforward"]
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)
        # The output embeddings of the model are its input embeddings when its config ties them
        self.tie_word_embeddings = self.tie_word_embeddings or self.text_config.tie_word_embeddings


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
    base_model_prefix = "model"
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ClefScorer):
            init.zeros_(module.prior_logit_scale)
            init.zeros_(module.joint_logit_scale)
            init.zeros_(module.residual_gate)


@auto_docstring(
    custom_intro="""
    Clef (Cloudflare/clef): the model of its repo, its language modeling head and a decision head that scores all the
    options of all the questions jointly. The template marks the span of each instruction and then of each of its
    options with a pair of `<|mark|>`. The options of all the questions are packed in one sequence: pooled from the
    hidden states and from the output embeddings of their tokens, they attend to the prompt and are summarized into the
    field of their question. The fields of the questions then attend to each other and to the prompt, and each option
    is scored against the field of its question.
    """
)
class ClefForDecision(DecisionMixin, ClefPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ClefConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        marks, num_options, question_type_ids = marks[0], num_options[0], question_type_ids[0]  # a batch of one prompt
        hidden_states = self.model(input_ids=input_ids, use_cache=False, **kwargs).last_hidden_state
        option_question_ids = torch.repeat_interleave(num_options)
        # The output embeddings of the tokens of each option give its lexical prior
        memory, options, fields, field_embeddings, priors = self.embeddings(
            hidden_states, input_ids, marks, num_options, question_type_ids, self.lm_head.weight
        )
        options, fields = self.encoder(memory, options, fields, field_embeddings, option_question_ids, **kwargs)
        scores = self.scorer(fields[:, option_question_ids], options, priors)
        return scores[0]


__all__ = ["ClefConfig", "ClefForDecision", "ClefPreTrainedModel"]
