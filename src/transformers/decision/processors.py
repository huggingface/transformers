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
"""Weightless processors of the option scores of decision models, like the logits processors of `generate`."""

import math

import torch

from .template_utils import DecisionPrompt


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
        n_options, probabilities = scores.numel(), scores.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() / math.log(max(n_options, 2))
        log_tokens = math.log10(max(len(prompt.input_ids) - prompt.marks[-1], 1)) / 4
        features = {"bias": 1.0, "entropy": entropy, "log_tokens": log_tokens, "n_options": n_options / 8}
        temperature = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return scores / temperature.clamp(self.lo, self.hi).clamp_min(1e-4)


class NoulBandProcessor:
    """Keeps the probability `p` of `true` within a band: `edge + slope * (p - 0.5)` above 0.5, mirrored below."""

    def __init__(self, edge: float, slope: float):
        self.edge, self.slope = edge, slope

    def __call__(self, scores: torch.Tensor, question, prompt: DecisionPrompt) -> torch.Tensor:
        if question.type != "noul":
            return scores
        p_true = scores.softmax(-1)[question.options.index("true")]
        p_true = torch.where(
            p_true >= 0.5, self.edge + self.slope * (p_true - 0.5), 1 - self.edge - self.slope * (0.5 - p_true)
        ).clamp(0, 1)
        return torch.stack([p_true if option == "true" else 1 - p_true for option in question.options]).log()


__all__ = ["CalibratedTemperatureProcessor", "NoulBandProcessor", "TemperatureProcessor"]
