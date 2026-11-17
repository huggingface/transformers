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
from collections.abc import Callable
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.utils.checkpoint import checkpoint

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionConfig, DecisionMixin, Question
from ...loss.loss_clef import check_clef_label_shapes, clef_label_errors, raise_label_error
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ..auto import AutoConfig, AutoModel
from ..auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ..clip.modeling_clip import CLIPMLP
from ..cohere_asr.modeling_cohere_asr import CohereAsrDecoderLayer
from ..molmo2.modeling_molmo2 import Molmo2VisionAttention, eager_attention_forward


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

# The same schema text and marks for backbones without a chat format (encoders): no ChatML turns nor system prompt,
# between the `[CLS]` and `[SEP]` of the tokenizer (or its beginning and end of sequence tokens)
CLEF_PLAIN_TEMPLATE = r"""{#- Clef's prompt (see CLEF_TEMPLATE) without the ChatML wrapper, for encoder backbones. -#}
{%- macro r(x) %}{{ x if x is string else x|tojson(sort_keys=true, separators=(",", ":")) }}{% endmacro -%}
{{- (cls_token or bos_token or "") ~ "STATE:\n<|split|>" ~ r(state) ~ "<|split|>\n\nSCHEMA FIELDS:\n" -}}
{%- for q in questions -%}
{{- "<|split|>\nFIELD " ~ loop.index ~ "\nID: " ~ q.name ~ "\nTYPE: " ~ q.type ~ "\nINSTRUCTION: <|mark|>" ~ r(q.name if q.instructions is none or q.instructions == "" else q.instructions) ~ "<|mark|>\nALLOWED OPTIONS:\n" -}}
{%- for key in q.options -%}
{%- set d = q.descriptions[loop.index0] -%}
{%- if q.type == "noul" and d is none %}{% set d = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}[key] %}{% endif -%}
{{- "OPTION " ~ loop.index ~ ": <|mark|>" ~ r({"option_id": key} if d is none else {"option_id": key, "description": d}) ~ "<|mark|>\n" -}}
{%- endfor -%}
{{- "END FIELD\n" -}}
{%- endfor -%}
{{- "<|split|>\n" ~ (sep_token or eos_token or "") -}}"""


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
    label_smoothing (`float`, *optional*, defaults to 0.0):
        Weight of the uniform distribution over the options in the target of the cross entropy of the loss.
    brier_weight (`float`, *optional*, defaults to 0.0):
        Weight of the Brier score in the loss.
    ordinal_weight (`float`, *optional*, defaults to 0.0):
        Weight of the ordinal term of the loss, the expected distance of the options to the right one on score questions.
    rlcd_weight (`float`, *optional*, defaults to 0.0):
        Weight of the RLCD policy term of the loss (Reinforcement Learning for Calibrated Decisions, from Laya), applied
        in training mode only.
    rlcd_sigma (`float`, *optional*, defaults to 0.4):
        Standard deviation of the noise added to the logits by the RLCD term.
    rlcd_samples (`int`, *optional*, defaults to 32):
        Number of noisy samples of the logits of each question in the RLCD term.
    rlcd_spherical_weight (`float`, *optional*, defaults to 0.75):
        Weight of the spherical score in the reward of the RLCD term.
    rlcd_rps_weight (`float`, *optional*, defaults to 1.0):
        Weight of the ranked probability score in the reward of the RLCD term, on score questions.
    rlcd_normalize_advantage (`bool`, *optional*, defaults to `False`):
        Whether the RLCD term divides its advantages by their standard deviation, as Laya's original objective does.
    head_type (`str`, *optional*, defaults to `"joint"`):
        The decision head: `"joint"` for the joint head of the checkpoints of clef, or `"cosine"` for a much smaller head
        that scores each option by the cosine of the projections of its span and of the span of its question.
    cosine_width (`int`, *optional*, defaults to 512):
        Width of the projections of the cosine head.
    cosine_initial_scale (`float`, *optional*, defaults to 12.0):
        Initial value of the temperature of the cosine of the cosine head, which is learned in log space.
    global_pooling (`str`, *optional*, defaults to `"last"`):
        Which attended token of each prompt is its global state, which the joint head reads: `"last"` for causal
        backbones, which have read the whole prompt there, or `"first"` (the `[CLS]` token) for bidirectional ones.
    prompt_style (`str`, *optional*, defaults to `"chatml"`):
        The template of the decision config of a new model: `"chatml"` for the ChatML prompt of the checkpoints of
        clef, or `"plain"` for the same schema text between the `[CLS]` and `[SEP]` of an encoder.

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
    label_smoothing: float | int = 0.0
    brier_weight: float | int = 0.0
    ordinal_weight: float | int = 0.0
    rlcd_weight: float | int = 0.0
    rlcd_sigma: float | int = 0.4
    rlcd_samples: int = 32
    rlcd_spherical_weight: float | int = 0.75
    rlcd_rps_weight: float | int = 1.0
    rlcd_normalize_advantage: bool = False
    head_type: str = "joint"
    cosine_width: int = 512
    cosine_initial_scale: float | int = 12.0
    global_pooling: str = "last"
    prompt_style: str = "chatml"

    # The files of a clef checkpoint that hold its head next to its weights, read with them: its weights and its config
    extra_files = ("joint_head.safetensors",)
    extra_config_files = ("joint_head_config.json",)

    @staticmethod
    def decision_config(head_configs: dict, text_config: dict):
        """The decision config of a clef checkpoint, from its config files, or `None` for any other checkpoint."""
        if "joint_head_config" not in head_configs:
            return None
        from ..auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING_NAMES

        model_type = text_config.get("model_type")
        if model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES and model_type != ClefConfig.model_type:
            raise ValueError(
                f"This folder ships a decision head in the layout of clef on a bidirectional backbone ({model_type}), "
                "which does not load as is: nothing records the pooling and prompt the head was trained with. Convert "
                "it with `src/transformers/models/clef/convert_clef_weights_to_hf.py --source <folder> --output "
                "<new folder> --global_pooling <first|last> --prompt_style <plain|chatml>`, then load the new folder."
            )
        return ClefConfig.prompt_decision_config("chatml")

    @staticmethod
    def prompt_decision_config(prompt_style: str = "chatml") -> DecisionConfig:
        """
        The decision config of clef with the template of `prompt_style`. Its head pools the spans of the options and
        reads no label token: it takes any number of options of every question type.
        """
        chat_template = CLEF_PLAIN_TEMPLATE if prompt_style == "plain" else CLEF_TEMPLATE
        return DecisionConfig(
            labels={"noul": None, "choice": None, "score": None},
            noul_options=["true", "false"],
            sort_choices=True,
            chat_template=chat_template,
        )

    def __post_init__(self, joint_head_config: dict | None = None, **kwargs):
        if joint_head_config:  # the config of the head of a clef checkpoint, whose config is `text_config`
            if joint_head_config.get("head_type") is not None:  # none for the joint head of clef
                raise ValueError(
                    f"`joint_head_config.json` declares a {joint_head_config['head_type']!r} head, which does not load "
                    "as is: convert the folder with `src/transformers/models/clef/convert_clef_weights_to_hf.py "
                    "--source <folder> --output <new folder>` (it reads `cosine_span_v1` cosine heads), then load the "
                    "new folder."
                )
            self.backbone_hidden_size, self.hidden_size = joint_head_config["hidden_size"], joint_head_config["width"]
            self.num_evidence_layers, self.num_hidden_layers = (
                joint_head_config["routing_layers"],
                joint_head_config["layers"],
            )
            self.num_attention_heads = self.num_key_value_heads = joint_head_config["heads"]
            self.intermediate_size = joint_head_config["feedforward"]
        # Floats, given as integers or not
        for name in (
            "label_smoothing",
            "brier_weight",
            "ordinal_weight",
            "rlcd_weight",
            "rlcd_sigma",
            "rlcd_spherical_weight",
            "rlcd_rps_weight",
            "cosine_initial_scale",
        ):
            setattr(self, name, float(getattr(self, name)))
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"`label_smoothing` must be in [0, 1), got {self.label_smoothing}.")
        for name in ("brier_weight", "ordinal_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be non-negative, got {getattr(self, name)}.")
        if self.global_pooling not in ("last", "first"):
            raise ValueError(f"`global_pooling` must be 'last' or 'first', got {self.global_pooling!r}.")
        if self.prompt_style not in ("chatml", "plain"):
            raise ValueError(f"`prompt_style` must be 'chatml' or 'plain', got {self.prompt_style!r}.")
        if self.head_type not in ("joint", "cosine"):
            raise ValueError(f"`head_type` must be 'joint' or 'cosine', got {self.head_type!r}.")
        if self.cosine_width <= 0:
            raise ValueError(f"`cosine_width` must be positive, got {self.cosine_width}.")
        if self.cosine_initial_scale <= 0:
            raise ValueError(f"`cosine_initial_scale` must be positive, got {self.cosine_initial_scale}.")
        if self.rlcd_weight < 0:
            raise ValueError(f"`rlcd_weight` must be non-negative, got {self.rlcd_weight}.")
        if self.rlcd_sigma <= 0:
            raise ValueError(f"`rlcd_sigma` must be positive, got {self.rlcd_sigma}.")
        if self.rlcd_samples < 2:
            raise ValueError(f"`rlcd_samples` must be at least 2, got {self.rlcd_samples}.")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)
        # The output embeddings of the model are its input embeddings when its config ties them
        self.tie_word_embeddings = self.tie_word_embeddings or self.text_config.tie_word_embeddings


def pool_spans(
    hidden_states: torch.Tensor,
    starts: torch.LongTensor,
    ends: torch.LongTensor,
    input_ids: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    The mean of `hidden_states` (`(batch_size, length, dim)`) over each span `[starts, ends)` (`(batch_size, num_spans)`),
    in float32, summed over the tokens of the span only: it does not depend on the other tokens, the padding nor the
    length of the sequence. Memory is linear in the number of tokens of the spans. An empty span gives zeros.

    With `input_ids` (`(batch_size, length)`), `hidden_states` is a table of embeddings (`(vocab_size, dim)`) and each
    span pools the rows of its tokens.
    """
    batch_size, num_spans = starts.shape
    lengths = (ends - starts).clamp(min=0).flatten()
    # The span of every token of every span, and its position in the sequence
    span_ids = torch.repeat_interleave(torch.arange(len(lengths), device=starts.device), lengths)
    first_tokens = lengths.cumsum(0) - lengths
    offsets = torch.arange(len(span_ids), device=starts.device) - first_tokens[span_ids]
    positions = starts.flatten()[span_ids] + offsets
    batch_ids = torch.div(span_ids, num_spans, rounding_mode="floor")
    if input_ids is None:
        tokens = hidden_states[batch_ids, positions].float()
    else:
        tokens = hidden_states[input_ids[batch_ids, positions]].float()
    # Summed relative to the first token of their span, which keeps the sums small when the states share a large offset
    zeros = torch.zeros(len(lengths), tokens.shape[-1], dtype=torch.float32, device=tokens.device)
    firsts = zeros.index_add(0, span_ids, tokens * (offsets == 0)[:, None])
    sums = zeros.index_add(0, span_ids, tokens - firsts[span_ids])
    return (firsts + sums / lengths.clamp(min=1)[:, None]).view(batch_size, num_spans, hidden_states.shape[-1])


def span_indices(
    marks: torch.LongTensor,
    num_options: torch.LongTensor,
    option_question_ids: torch.LongTensor,
    is_option: torch.BoolTensor,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    The `starts` and `ends` of the marked spans of each prompt, and the index among them of the instruction of each
    question and of each packed option (0 for padding questions and options). Each question marks the span of its
    instruction, then of each of its options; padding marks are -1.
    """
    starts, ends = marks.clamp(min=0).unflatten(-1, (-1, 2)).unbind(-1)
    spans_per_question = (num_options + 1) * (num_options > 0)
    instructions = spans_per_question.cumsum(-1) - spans_per_question
    first_options = num_options.cumsum(-1) - num_options
    positions = torch.arange(is_option.shape[-1], device=marks.device)
    option_spans = (
        instructions.gather(-1, option_question_ids) + 1 + positions - first_options.gather(-1, option_question_ids)
    )
    instructions = instructions.masked_fill(num_options == 0, 0)
    option_spans = option_spans.masked_fill(~is_option, 0)
    return starts, ends, instructions, option_spans


def padding_mask(is_valid: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor | None:
    """The additive attention mask of the keys `is_valid` (`(batch_size, num_keys)`), `None` when they all are."""
    if bool(is_valid.all()):
        return None
    mask = torch.zeros(is_valid.shape, dtype=dtype, device=is_valid.device)
    return mask.masked_fill(~is_valid, torch.finfo(dtype).min)[:, None, None]


class ClefAttention(Molmo2VisionAttention):
    """
    The attention of the head, which runs sdpa or eager whatever the model's `attn_implementation`: that selects the
    attention of the backbone over the prompt (a Flash Attention kernel, for example), and the head masks its padding
    with additive masks, which only these two take.
    """

    def __init__(self, config: ClefConfig, hidden_size: int | None = None):
        super().__init__(config, hidden_size)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_value_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(encoder_hidden_states).view(key_value_shape).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(key_value_shape).transpose(1, 2)

        implementation = self.config._attn_implementation
        if implementation not in ("eager", "sdpa"):  # the backbone's
            implementation = "sdpa"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


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
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.cross_attn(
            self.input_layernorm(hidden_states),
            self.memory_layernorm(encoder_hidden_states),
            attention_mask=attention_mask,
            **kwargs,
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


class ClefCosineHead(nn.Module):
    """
    A small head: each option is scored by the cosine of the projection of its span and of the projection of the span
    of its question, both averaged over the raw hidden states of the backbone, times a learned temperature. All in
    float32.
    """

    def __init__(self, config: ClefConfig):
        super().__init__()
        backbone_hidden_size, width = config.backbone_hidden_size, config.cosine_width
        self.question_projection = nn.Linear(backbone_hidden_size, width)
        self.question_norm = nn.LayerNorm(width)
        self.option_projection = nn.Linear(backbone_hidden_size, width)
        self.option_norm = nn.LayerNorm(width)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(config.cosine_initial_scale)))

    @staticmethod
    def project(spans: torch.Tensor, projection: nn.Linear, norm: nn.LayerNorm) -> torch.Tensor:
        spans = nn.functional.linear(spans.float(), projection.weight.float(), projection.bias.float())
        return nn.functional.layer_norm(spans, norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        option_question_ids: torch.LongTensor,
        is_option: torch.BoolTensor,
    ) -> torch.Tensor:
        batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None]
        starts, ends, instructions, option_spans = span_indices(marks, num_options, option_question_ids, is_option)
        span_states = pool_spans(hidden_states, starts, ends)
        # Float32 whatever the dtype of the weights (a model cast with `.to()`) and outside of autocast
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            questions = self.project(span_states[batch, instructions], self.question_projection, self.question_norm)
            options = self.project(span_states[batch, option_spans], self.option_projection, self.option_norm)
        questions = nn.functional.normalize(questions.float(), dim=-1)[batch, option_question_ids]
        options = nn.functional.normalize(options.float(), dim=-1)
        return (questions * options).sum(-1) * self.logit_scale.clamp(max=math.log(100.0)).exp()


class ClefEmbeddings(nn.Module):
    """
    The memory of the prompt, the options, the fields of the questions with their embeddings, and the lexical prior of
    the options, from the hidden states of the backbone and the output embeddings of the tokens of the marked spans,
    averaged over each span. The options of each prompt are packed in question order, `option_question_ids` gives the
    question of each and `is_option` whether it exists.
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
        self.global_pooling = config.global_pooling

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        output_embeddings: torch.Tensor,
        option_question_ids: torch.LongTensor,
        is_option: torch.BoolTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None]
        starts, ends, instructions, option_spans = span_indices(marks, num_options, option_question_ids, is_option)

        hidden_states = self.hidden_norm(hidden_states)
        span_states = pool_spans(hidden_states, starts, ends)
        span_lexical = pool_spans(output_embeddings, starts, ends, input_ids=input_ids)
        questions = span_states[batch, instructions].to(dtype)
        contexts, lexical = span_states[batch, option_spans].to(dtype), span_lexical[batch, option_spans].to(dtype)
        # The global state is the first or the last token each prompt attends to
        first = self.global_pooling == "first"
        if attention_mask is None:
            global_state = hidden_states[:, :1] if first else hidden_states[:, -1:]
        else:
            attended = attention_mask.long()  # `argmax` gives the first attended token
            if not first:  # the attended token of the highest position
                attended = attended * torch.arange(attended.shape[-1], device=marks.device)
            global_state = hidden_states[batch, attended.argmax(-1)[:, None]]
        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[batch, option_question_ids]
        )
        # The cosine of the output embeddings of each option with its question and the end of the prompt
        anchors = nn.functional.normalize(questions + global_state, dim=-1)[batch, option_question_ids]
        priors = (nn.functional.normalize(lexical, dim=-1) * anchors).sum(-1)
        fields = self.question_projection(questions)
        field_embeddings = self.global_projection(global_state) + self.type_embedding(question_type_ids)
        return self.memory_projection(hidden_states), options, fields, field_embeddings, priors


class ClefEncoder(nn.Module):
    """
    The options attend to the prompt, each question summarizes its options into its field, then the fields attend to
    each other and to the prompt. Padding tokens, options and questions are masked out of every attention and of the
    routing of the options.
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
        is_option: torch.BoolTensor,
        is_question: torch.BoolTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`fields` are the projected questions, which route the options; `field_embeddings` are added to them after."""
        memory_mask = None if attention_mask is None else padding_mask(attention_mask.bool(), memory.dtype)
        # Positional, so that the memory gets its gradients under reentrant activation checkpointing too
        for layer in self.evidence_layers:
            options = layer(options, memory, memory_mask, **kwargs)

        # Each question routes its own options only
        routing_logits = fields @ options.transpose(-1, -2) / math.sqrt(fields.shape[-1])
        questions = torch.arange(fields.shape[1], device=fields.device)
        is_own_option = (option_question_ids[:, None] == questions[:, None]) & is_option[:, None]
        # Masked in float32, where the softmax runs: the minimum of float64 would be -inf there, and a padding question
        # (none of whose options it owns) would get NaN weights instead of uniform ones
        routing_logits = routing_logits.float().masked_fill(~is_own_option, torch.finfo(torch.float32).min)
        summaries = routing_logits.softmax(-1) @ options.float()
        fields = fields + self.option_summary_norm(summaries.to(fields.dtype)) + field_embeddings
        field_mask = padding_mask(is_question, fields.dtype)
        for layer in self.layers:
            fields = layer(fields, field_mask, memory, memory_mask, **kwargs)
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
    # The backbone runs attention over the prompt: any implementation it supports (it checks it), while the attention of
    # the head runs sdpa or eager (see `ClefAttention`)
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True
    supports_gradient_checkpointing = True  # when its backbone does, see `ClefForDecision`

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ClefScorer):
            init.zeros_(module.prior_logit_scale)
            init.zeros_(module.joint_logit_scale)
            init.zeros_(module.residual_gate)
        elif isinstance(module, ClefCosineHead):
            init.constant_(module.logit_scale, math.log(self.config.cosine_initial_scale))


@dataclass
@auto_docstring(
    custom_intro="""
    The output of [`ClefForDecision`] for a batch of prompts, with `num_questions` questions in all (those of each
    prompt in turn) of at most `max_options` options.
    """
)
class ClefDecisionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
        The loss of the questions, see [`ClefForDecision`].
    logits (`torch.FloatTensor` of shape `(num_questions, max_options)`):
        The scores of the options of each question, in float32. The cells of the options a question does not have hold
        the minimum of the dtype.
    option_mask (`torch.BoolTensor` of shape `(num_questions, max_options)`):
        Whether each cell of `logits` is an option of its question.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    option_mask: torch.BoolTensor | None = None

    def split(self, split_size_or_sections: list[int]) -> tuple[torch.Tensor, ...]:
        """
        The scores of the options of each question, packed then split like a tensor of them: what the `decide` of
        [`DecisionMixin`] reads from the output of `forward`.
        """
        return self.logits[self.option_mask].split(split_size_or_sections)


@auto_docstring(
    custom_intro="""
    Clef (Cloudflare/clef): the model of its repo, its language modeling head and a decision head that scores all the
    options of all the questions jointly. The template marks the span of each instruction and then of each of its
    options with a pair of `<|mark|>`. The options of all the questions of a prompt are packed in one sequence: pooled
    from the hidden states and from the output embeddings of their tokens, they attend to the prompt and are summarized
    into the field of their question. The fields of the questions then attend to each other and to the prompt, and each
    option is scored against the field of its question.
    """
)
class ClefForDecision(DecisionMixin, ClefPreTrainedModel):
    accepts_loss_kwargs = False  # the loss is a mean over the questions of the batch, not over tokens
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ClefConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        # A backbone whose position embeddings have a padding index (the RoBERTa family) numbers the positions of its
        # tokens itself, from `input_ids`, after that index: positions counted from the attention mask would be shifted
        self._positions_from_input_ids = any(
            isinstance(module, nn.Embedding)
            and name.endswith("position_embeddings")
            and module.padding_idx is not None
            for name, module in self.model.named_modules()
        )
        if config.head_type == "cosine":  # no language modeling head, nothing to tie
            self._tied_weights_keys = {}
            self._keep_in_fp32_modules_strict = ["cosine_head"]  # float32 when loaded in half precision
            self.cosine_head = ClefCosineHead(config)
        else:
            if config.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                # The output embeddings of the causal language model, tied to its input embeddings when it ties them
                input_embeddings = self.model.get_input_embeddings()
                embeddings = next(
                    (name for name, module in self.model.named_modules() if module is input_embeddings), None
                )
                if embeddings is None:
                    raise ValueError(
                        f"The input embeddings of the backbone {type(self.model).__name__} are not one of its modules: "
                        "the language modeling head of the joint head cannot be tied to them. Use a backbone whose "
                        '`get_input_embeddings()` returns one of its modules, or `head_type="cosine"`.'
                    )
                self._tied_weights_keys = {"lm_head.weight": f"model.{embeddings}.weight"}
                self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
            else:  # an encoder without a language modeling head: the lexical prior reads its input embeddings
                self._tied_weights_keys = {}
            self.embeddings = ClefEmbeddings(config)
            self.encoder = ClefEncoder(config)
            self.scorer = ClefScorer(config)
        self.post_init()

    def post_init(self):
        """The decision config of a new model is clef's, in its `prompt_style`; `from_pretrained` reads the saved one."""
        super().post_init()
        self.decision_config = self.config.prompt_decision_config(self.config.prompt_style)

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func=checkpoint, every_n_layers=1
    ):
        """
        Activation checkpointing of the backbone, which it delegates to, and of the layers of the head. Raises when the
        backbone does not support it, rather than checkpointing the head alone.
        """
        if not self.model.supports_gradient_checkpointing:
            if enable:
                raise ValueError(
                    f"The backbone of {self.__class__.__name__} ({self.model.__class__.__name__}) does not support "
                    "activation checkpointing."
                )
        else:
            self.model._set_gradient_checkpointing(enable, gradient_checkpointing_func, every_n_layers)
        head_layers = [  # the evidence and decoder layers of the joint head
            module
            for name, module in self.named_modules()
            if isinstance(module, GradientCheckpointingLayer) and not name.startswith("model.")
        ]
        for index, module in enumerate(head_layers):
            module.gradient_checkpointing = enable and index % every_n_layers == 0
            module._gradient_checkpointing_func = gradient_checkpointing_func

    def lexical_embeddings(self) -> torch.Tensor:
        """The table of the embeddings of the tokens that gives the lexical prior of the options."""
        lm_head = getattr(self, "lm_head", None)
        return self.model.get_input_embeddings().weight if lm_head is None else lm_head.weight

    @staticmethod
    def _check_inputs(input_ids, marks, num_options, question_options, option_mask, given_option_mask, labels):
        """
        Raises a `ValueError` when the inputs disagree with each other, before anything indexes with them: the marks of a
        prompt that do not give a span within it to each instruction and option of its questions, an `option_mask`
        that disagrees with `num_options`, or labels that are not one per question or name an option their question
        does not have. Every check of the values runs on the device and costs a single synchronisation.
        """
        num_questions, length = len(question_options), input_ids.shape[-1]
        if option_mask.dim() != 2 or option_mask.shape[0] != num_questions:
            raise ValueError(
                f"`option_mask` has {option_mask.shape[0] if option_mask.dim() else 0} rows (shape "
                f"{tuple(option_mask.shape)}), one per question, but `num_options` has {num_questions} questions."
            )
        if labels is not None:
            check_clef_label_shapes(labels, option_mask)
        # Two marks per instruction and per option of each prompt, each a position in [0, length]
        needed = 2 * ((num_options + 1) * (num_options > 0)).sum(-1)
        in_use = torch.arange(marks.shape[-1], device=marks.device) < needed[:, None]
        valid_marks = (in_use & (marks >= 0) & (marks <= length)).sum(-1)
        errors = [(valid_marks != needed).any()]
        if given_option_mask:
            errors.append((option_mask.sum(-1) != question_options).any())
        if labels is not None:
            label_errors = clef_label_errors(labels, option_mask)
            errors.append(label_errors.any())
        if not bool(torch.stack(errors).any()):  # the synchronisation
            return
        # What is wrong, for the message: the slow path of an error only
        for row in (valid_marks != needed).nonzero()[:, 0].tolist():
            raise ValueError(
                f"The marks of prompt {row} do not fit its questions: prompt {row} has {int(valid_marks[row])} valid "
                f"marks (positions in [0, {length}]) of the {int(needed[row])} that its questions need, two per "
                f"instruction and per option, in {marks.shape[-1]} marks: {marks[row].tolist()}."
            )
        if given_option_mask:
            counts = option_mask.sum(-1)
            for index in (counts != question_options).nonzero()[:, 0].tolist():
                raise ValueError(
                    f"`option_mask` disagrees with `num_options`: question {index} has {int(counts[index])} options in "
                    f"`option_mask` and {int(question_options[index])} in `num_options`."
                )
        raise_label_error(labels, option_mask, label_errors)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        option_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ClefDecisionOutput:
        r"""
        marks (`torch.LongTensor` of shape `(batch_size, num_marks)`):
            The positions in `input_ids` of the marks of each prompt: per question, the start and end of the span of
            its instruction then of each of its options. Padded with `-1`.
        num_options (`torch.LongTensor` of shape `(batch_size, max_questions)`):
            The number of options of each question of each prompt, padded with `0`: a question is real iff it has
            options.
        question_type_ids (`torch.LongTensor` of shape `(batch_size, max_questions)`):
            The index of the type of each question in `Question.questions_types`, padded with `0`.
        labels (`torch.LongTensor` of shape `(num_questions,)` or `torch.FloatTensor` of shape `(num_questions, max_options)`, *optional*):
            For the real questions of all the prompts, in order: the index of the right option of each question,
            `-100` for a question to ignore, or the distribution of the right answer over its options.
        option_mask (`torch.BoolTensor` of shape `(num_questions, max_options)`, *optional*):
            The options each question has, from `num_options` by default.
        """
        # Positions that skip the padding, so that left and right padding give the same hidden states. The RoBERTa family
        # skips its padding tokens itself, and gets the same positions with and without a mask that way
        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None and attention_mask is not None and not self._positions_from_input_ids:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        # The options of each prompt packed in question order: the question of each, and whether it exists
        is_question = num_options > 0
        option_ends = num_options.cumsum(-1)
        max_options = int(option_ends[:, -1].max())
        if max_options == 0:
            raise ValueError(
                "The batch has no questions: `num_options` is 0 for every question of every prompt. Give each prompt at "
                "least one question with options."
            )
        # The options of each real question, prompt after prompt, padded to the widest question
        # One boolean indexing (one synchronisation) for the options and the types of the real questions
        question_options, question_types = torch.stack([num_options, question_type_ids], dim=-1)[is_question].unbind(
            -1
        )
        given_option_mask = option_mask is not None
        if option_mask is None:
            option_mask = (
                torch.arange(int(question_options.max()), device=num_options.device) < question_options[:, None]
            )
        self._check_inputs(input_ids, marks, num_options, question_options, option_mask, given_option_mask, labels)
        hidden_states = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, **kwargs
        ).last_hidden_state
        positions = torch.arange(max_options, device=num_options.device)
        is_option = positions < option_ends[:, -1:]
        positions = positions.expand(len(num_options), -1).contiguous()
        option_question_ids = torch.searchsorted(option_ends, positions, right=True).clamp(
            max=num_options.shape[-1] - 1
        )
        if self.config.head_type == "cosine":
            scores = self.cosine_head(hidden_states, marks, num_options, option_question_ids, is_option)[is_option]
        else:
            # The output embeddings of the tokens of each option give its lexical prior
            memory, options, fields, field_embeddings, priors = self.embeddings(
                hidden_states,
                input_ids,
                marks,
                num_options,
                question_type_ids,
                self.lexical_embeddings(),
                option_question_ids,
                is_option,
                attention_mask,
            )
            options, fields = self.encoder(
                memory,
                options,
                fields,
                field_embeddings,
                option_question_ids,
                is_option,
                is_question,
                attention_mask,
                **kwargs,
            )
            batch = torch.arange(len(fields), device=fields.device)[:, None]
            scores = self.scorer(fields[batch, option_question_ids], options, priors)[is_option].float()

        logits = scores.new_full(option_mask.shape, torch.finfo(scores.dtype).min).masked_scatter(option_mask, scores)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                option_mask,
                is_score=question_types == Question.questions_types.index("score"),
                label_smoothing=self.config.label_smoothing,
                brier_weight=self.config.brier_weight,
                ordinal_weight=self.config.ordinal_weight,
                # The policy term is random: out of training the loss is the supervised one, the same at every call
                rlcd_weight=self.config.rlcd_weight if self.training else 0.0,
                rlcd_sigma=self.config.rlcd_sigma,
                rlcd_samples=self.config.rlcd_samples,
                rlcd_spherical_weight=self.config.rlcd_spherical_weight,
                rlcd_rps_weight=self.config.rlcd_rps_weight,
                rlcd_normalize_advantage=self.config.rlcd_normalize_advantage,
                check_labels=False,  # in the single synchronisation of `_check_inputs`
            )
        return ClefDecisionOutput(loss=loss, logits=logits, option_mask=option_mask)


__all__ = ["ClefConfig", "ClefForDecision", "ClefPreTrainedModel"]
