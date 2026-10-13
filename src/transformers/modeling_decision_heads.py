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
"""Heads of decision models: they score the options of the questions of a [`DecisionPrompt`]."""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .decision_utils import DECISION_HEAD_NAME, DecisionConfig, DecisionPrompt, label_token_ids
from .utils import cached_file


class DecisionHead(nn.Module):
    """
    Base class of the heads of decision models. A head is built from `decision_config.head` and its weights live in
    `decision_head.safetensors`, so it can be pushed alone next to the original checkpoint of the backbone.

    Subclasses implement `score`, which gets the final hidden states of one prompt, `(seq_len, hidden_size)`.
    """

    def forward(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        """Run the backbone of `model` on `prompt`, and return one 1D tensor of option scores per question."""
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        hidden_states = model.base_model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        dtype = next(self.parameters(), hidden_states).dtype
        return self.score(hidden_states.to(dtype), prompt, model.get_output_embeddings())

    def score(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, config: DecisionConfig, **kwargs):
        """Build the head of `config.head` and load `decision_head.safetensors` (`kwargs` go to `cached_file`)."""
        head_kwargs = {key: value for key, value in config.head.items() if key != "type"}
        head = DECISION_HEADS[config.head["type"]](**head_kwargs)
        head.load_state_dict(load_file(cached_file(pretrained_model_name_or_path, DECISION_HEAD_NAME, **kwargs)))
        return head.eval()


class LabelHead(DecisionHead):
    """Scores the options with the logits of their single-token labels, at the last token before each `<|mark|>`."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        positions = torch.tensor(prompt.marks, device=model.device) - 1
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
        return [logits[i, label_token_ids(self.tokenizer, q)] for i, q in enumerate(prompt.questions)]


class VerbalizerHead(DecisionHead):
    """
    Linear scores of single-token labels at the token before the `<|mark|>` of each question, soft-capped like the
    logits of a language model head when `softcap` is set. The options of a question get the rows from
    `offsets[question.type]` on, in order.
    """

    def __init__(self, hidden_size: int, num_labels: int, offsets: dict[str, int], softcap: float | None = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_labels)
        self.offsets, self.softcap = offsets, softcap

    def score(self, hidden_states, prompt, lm_head):
        logits = self.proj(hidden_states[[mark - 1 for mark in prompt.marks]])
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return [row[self.offsets[q.type] :][: len(q.options)] for row, q in zip(logits, prompt.questions)]


class PointerHead(DecisionHead):
    """
    Scores each option by the dot product of the hidden states at the end of the option and at the end of the question,
    projected to `head_dim`, with optional layer norms and an MLP term. Each question has one `<|mark|>` after each of
    its options and one after itself.
    """

    def __init__(
        self, hidden_size: int, head_dim: int = 256, bias: bool = True, norm: bool = False, mlp: bool = False
    ):
        super().__init__()
        self.key = nn.Linear(hidden_size, head_dim, bias=bias)
        self.query = nn.Linear(hidden_size, head_dim, bias=bias)
        self.candidate_norm = nn.LayerNorm(hidden_size) if norm else nn.Identity()
        self.query_norm = nn.LayerNorm(hidden_size) if norm else nn.Identity()
        self.mlp = mlp
        if mlp:
            self.candidate_mlp = nn.Linear(hidden_size, head_dim)
            self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
            self.scalar = nn.Linear(head_dim, 1, bias=False)

    def score(self, hidden_states, prompt, lm_head):
        scores, start = [], 0
        for question in prompt.questions:
            end = start + len(question.options)
            candidates = self.candidate_norm(hidden_states[[mark - 1 for mark in prompt.marks[start:end]]])
            query = self.query_norm(hidden_states[prompt.marks[end] - 1])
            logits = self.key(candidates) @ self.query(query) / math.sqrt(self.key.out_features)
            if self.mlp:
                logits = logits + self.scalar(F.gelu(self.candidate_mlp(candidates) + self.query_mlp(query)))[:, 0]
            scores.append(logits)
            start = end + 1
        return scores


class OptionMarkerHead(DecisionHead):
    """
    Scores each option with an MLP on the hidden state of its `[MASK]` (the token before each `<|mark|>`, a last mark
    closing the block), each option seeing only the prompt and itself (wfzyx/von). A zero-shot `noul` question has a
    second block without the state, whose scores debias the first. The state is written once more after the last block,
    for the token count of the input-conditioned temperature.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        sliding_window: int = 64,
        calibration: dict[str, float] | None = None,
        temperature: float = 1.0,
        noul_prior: dict[str, float] | None = None,
        noul_band: list[float] | None = None,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.out_proj = nn.Linear(hidden_size // 2, 1)
        self.sliding_window, self.calibration, self.temperature = sliding_window, calibration, temperature
        self.noul_prior, self.noul_band = noul_prior, noul_band

    def blocks(self, prompt: DecisionPrompt) -> list[tuple[int, list[int], int]]:
        """`(start, [MASK] positions, end)` of each block."""
        per_block, marks = len(prompt.questions[0].options) + 1, prompt.marks
        if len(prompt.questions) != 1 or len(marks) not in (per_block, 2 * per_block):
            raise ValueError(f"Expected one question and {per_block} or {2 * per_block} marks, got {len(marks)}.")
        groups = [marks[i : i + per_block] for i in range(0, len(marks), per_block)]
        return [
            (group_start, [m - 1 for m in group[:-1]], group[-1])
            for group_start, group in zip([0, *marks[per_block - 1 :: per_block]], groups)
        ]

    def backbone_inputs(self, prompt: DecisionPrompt) -> dict[str, Any]:
        """Options attend to the prompt and to themselves, and their positions restart after the prompt."""
        blocks = self.blocks(prompt)
        length = blocks[-1][2]
        position_ids, option_ids = torch.arange(length), torch.full((length,), -1)
        block_ids = torch.zeros(length, dtype=torch.long)
        for block, (start, reads, end) in enumerate(blocks):
            block_ids[start:end] = block
            position_ids[start:end] = torch.arange(end - start)
            for k, (begin, stop) in enumerate(zip(reads, [*reads[1:], end - 1])):
                option_ids[begin:stop] = k
                position_ids[begin:stop] = torch.arange(reads[0] - start, reads[0] - start + stop - begin)
        query, key = option_ids[:, None], option_ids[None, :]
        allowed = ((query == -1) & (key == -1)) | ((query != -1) & ((key == -1) | (query == key)))
        allowed = (allowed & (block_ids[:, None] == block_ids[None, :])) | torch.eye(length, dtype=torch.bool)
        local = (position_ids[:, None] - position_ids[None, :]).abs() <= self.sliding_window
        return {
            "input_ids": torch.tensor([prompt.input_ids[:length]]),
            "position_ids": position_ids[None],
            "attention_mask": {
                "full_attention": allowed[None, None],
                "sliding_attention": (allowed & local)[None, None],
            },
        }

    def forward(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        inputs = self.backbone_inputs(prompt)
        inputs["attention_mask"] = {name: mask.to(model.device) for name, mask in inputs["attention_mask"].items()}
        inputs = {
            name: value.to(model.device) if name != "attention_mask" else value for name, value in inputs.items()
        }
        hidden_states = model.base_model(**inputs).last_hidden_state[0]
        return self.score(hidden_states.to(self.dense.weight.dtype), prompt, None)

    def option_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.norm(F.gelu(self.dense(self.input_norm(hidden_states)))))[:, 0]

    def score(self, hidden_states, prompt, lm_head):
        (question,), blocks = prompt.questions, self.blocks(prompt)
        logits = self.option_logits(hidden_states[blocks[0][1]])
        if len(blocks) == 2:  # zero-shot noul
            true, false = question.options.index("true"), question.options.index("false")
            null = self.option_logits(hidden_states[blocks[1][1]])
            correction = self.noul_prior["a"] * (null[true] - null[false]) + self.noul_prior["b"]
            logits = torch.stack([logits[i] - correction if i == true else logits[i] for i in range(2)])
        temperature = self.temperature
        if self.calibration:
            probabilities, n = torch.softmax(logits.float(), dim=-1), logits.numel()
            entropy = (
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item() / math.log(n) if n > 1 else 0.0
            )
            state_tokens = len(prompt.input_ids) - prompt.marks[-1]
            c = self.calibration
            features = {
                "bias": 1.0,
                "entropy": entropy,
                "log_tokens": math.log10(max(state_tokens, 1)) / 4,
                "n_options": n / 8,
            }
            temperature = min(c["hi"], max(c["lo"], sum(c.get(name, 0.0) * value for name, value in features.items())))
        logits = logits / max(temperature, 1e-4)
        if question.type == "noul" and self.noul_band:
            # von answers within a band around 0.5: p -> edge + slope * (p - 0.5), mirrored below 0.5
            edge, slope = self.noul_band
            true = question.options.index("true")
            p = min(1.0, max(0.0, torch.softmax(logits.float(), dim=-1)[true].item()))
            p = min(1.0, max(0.0, edge + slope * (p - 0.5) if p >= 0.5 else (1.0 - edge) - slope * (0.5 - p)))
            logits = torch.tensor([p if i == true else 1.0 - p for i in range(2)], device=logits.device).log()
        return [logits]


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # The identity keeps the checkpoint's indices (a dropout in training)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Identity(), nn.Linear(feedforward, width)
        )

    def forward(self, queries, memory):
        # `memory_norm` twice, as `key is value` would switch torch to a packed key/value projection
        routed, _ = self.attention(
            self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(DecisionHead):
    """
    Scores all the options of all the questions jointly (Cloudflare/clef): the template marks the span of each
    instruction and then of each of its options with a pair of `<|mark|>`. Options are pooled from the hidden states and
    from the output embeddings of their tokens, routed through the prompt, and the questions attend to each other.
    """

    type_ids = {"noul": 0, "choice": 1, "score": 2}

    def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int, feedforward: int):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                width, heads, feedforward, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def score(self, hidden_states, prompt, lm_head):
        spans = list(zip(prompt.marks[::2], prompt.marks[1::2]))
        input_ids = torch.tensor(prompt.input_ids, device=hidden_states.device)
        h = self.hidden_norm(hidden_states)
        memory, global_vector = self.memory_projection(h)[None], h[-1]
        pool = lambda x, spans: torch.stack([x[start:end].mean(dim=0) for start, end in spans])  # noqa: E731
        questions, contexts, lexical, start = [], [], [], 0
        for question in prompt.questions:
            questions.append(pool(h, spans[start : start + 1])[0])
            option_spans = spans[start + 1 : start + 1 + len(question.options)]
            contexts.append(pool(h, option_spans))
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


DECISION_HEADS = {
    "verbalizer": VerbalizerHead,
    "pointer": PointerHead,
    "option_marker": OptionMarkerHead,
    "joint_schema": JointSchemaHead,
}
