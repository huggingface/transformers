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
"""Helpers of the decision heads, the default head, and the processors of the option scores."""

import math

import torch

from .decision_utils import DecisionPrompt, label_token_ids


DECISION_HEAD_FOLDER = "decision_head"


class DecisionHeadMixin:
    """
    Decision heads are models named after the first model that used them, saved in the `decision_head` folder next to
    the original checkpoint of their backbone. `forward(hidden_states, prompt, lm_head)` scores the options of each
    question of the prompt.
    """

    def backbone_inputs(self, prompt: DecisionPrompt, model) -> dict:
        return {"input_ids": torch.tensor([prompt.input_ids], device=model.device), "use_cache": False}

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        hidden_states = model.base_model(**self.backbone_inputs(prompt, model)).last_hidden_state[0]
        return self(hidden_states.to(self.dtype), prompt, model.get_output_embeddings())


class LabelHead:
    """Default head: the logits of the single-token labels of the options, at the token before each `<|mark|>`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        positions = torch.tensor(prompt.marks, device=model.device) - 1
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
        return [logits[i, label_token_ids(self.tokenizer, q)] for i, q in enumerate(prompt.questions)]


class TemperatureProcessor:
    """Divides the scores of each question type by its temperature."""

    def __init__(self, temperature: dict[str, float]):
        self.temperature = temperature

    def __call__(self, scores: torch.Tensor, question, prompt: DecisionPrompt) -> torch.Tensor:
        return scores / self.temperature.get(question.type, 1.0)


class CalibratedTemperatureProcessor:
    """
    Temperature `clamp(bias + entropy * H + log_tokens * log10(n) / 4 + n_options * K / 8, lo, hi)`, with `H` the
    normalized entropy of the scores, `n` the number of tokens after the last `<|mark|>` (where the template writes the
    state) and `K` the number of options.
    """

    def __init__(self, lo: float, hi: float, **weights: float):
        self.lo, self.hi, self.weights = lo, hi, weights

    def __call__(self, scores: torch.Tensor, question, prompt: DecisionPrompt) -> torch.Tensor:
        n_options, p = scores.numel(), scores.softmax(-1)
        features = {
            "bias": 1.0,
            "entropy": -(p * p.clamp_min(1e-12).log()).sum() / math.log(max(n_options, 2)),
            "log_tokens": math.log10(max(len(prompt.input_ids) - prompt.marks[-1], 1)) / 4,
            "n_options": n_options / 8,
        }
        temperature = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return scores / temperature.clamp(self.lo, self.hi).clamp_min(1e-4)


class NoulBandProcessor:
    """Keeps the probability `p` of `true` within a band: `edge + slope * (p - 0.5)` above 0.5, mirrored below."""

    def __init__(self, edge: float, slope: float):
        self.edge, self.slope = edge, slope

    def __call__(self, scores: torch.Tensor, question, prompt: DecisionPrompt) -> torch.Tensor:
        if question.type != "noul":
            return scores
        true = question.options.index("true")
        p = scores.softmax(-1)[true]
        p = torch.where(p >= 0.5, self.edge + self.slope * (p - 0.5), 1 - self.edge - self.slope * (0.5 - p))
        return torch.stack([p.clamp(0, 1) if i == true else 1 - p.clamp(0, 1) for i in range(2)]).log()
