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
"""Inference helpers of decision heads, the default head of models without one, and the score processors."""

import inspect
import math

import torch

from .decision_utils import DecisionPrompt, label_token_ids


DECISION_HEAD_FOLDER = "decision_head"


class DecisionHeadMixin:
    """
    Decision heads are models (`models/<name>/modeling_<name>.py`, named after the first model that used them) whose
    `forward(hidden_states, prompt, lm_head)` scores the options of each question of a [`DecisionPrompt`]. They are saved
    in the `decision_head` folder of a repo, next to the original checkpoint of their backbone.
    """

    def backbone_inputs(self, prompt: DecisionPrompt, model) -> dict:
        """The inputs of the backbone of `model` for `prompt`."""
        return {"input_ids": torch.tensor([prompt.input_ids])}

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        """Run the backbone of `model` and then the head on `prompt`: one 1D tensor of option scores per question."""
        backbone = model.base_model
        inputs = {
            name: value.to(model.device)
            if torch.is_tensor(value)
            else {k: v.to(model.device) for k, v in value.items()}
            for name, value in self.backbone_inputs(prompt, model).items()
        }
        if "use_cache" in inspect.signature(backbone.forward).parameters:
            inputs["use_cache"] = False
        hidden_states = backbone(**inputs).last_hidden_state[0]
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
    Input-conditioned temperature `clamp(bias + entropy * H + log_tokens * log10(n) / 4 + n_options * K / 8, lo, hi)`,
    with `H` the normalized entropy of the scores, `n` the number of tokens after the last `<|mark|>` (where the template
    writes the state) and `K` the number of options.
    """

    def __init__(self, bias: float, entropy: float, log_tokens: float, n_options: float, lo: float, hi: float):
        self.bias, self.entropy, self.log_tokens, self.n_options, self.lo, self.hi = (
            bias,
            entropy,
            log_tokens,
            n_options,
            lo,
            hi,
        )

    def __call__(self, scores: torch.Tensor, question, prompt: DecisionPrompt) -> torch.Tensor:
        n_options, probabilities = scores.numel(), scores.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() / math.log(max(n_options, 2))
        log_tokens = math.log10(max(len(prompt.input_ids) - prompt.marks[-1], 1)) / 4
        temperature = (
            self.bias + self.entropy * entropy + self.log_tokens * log_tokens + self.n_options * n_options / 8
        )
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
        p = p.clamp(0, 1)
        return torch.stack([p if i == true else 1 - p for i in range(2)]).log()
