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
"""A decision model: a model and the decision head that reads it, from the files of one repo."""

from pathlib import Path

from torch import nn

from ..modeling_utils import PreTrainedModel
from ..models.auto import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from ..utils import cached_file
from .configuration_utils import DECISION_HEAD_CONFIG_NAME, DecisionConfig


def load_decision_head(
    model: PreTrainedModel, repo: str | Path, decision_config: DecisionConfig
) -> PreTrainedModel | None:
    """
    Sets `model` up as its decision config says (attention, LoRA in a subfolder) and returns the head of `repo`, whose
    config (`decision_head.json`) names its weights file. `None` for a model without a head, read from its logits.
    """
    if not decision_config.is_causal:
        model.config.get_text_config().is_causal = False
    if decision_config.adapter:
        model.load_adapter(repo, adapter_kwargs={"subfolder": decision_config.adapter})
    if not cached_file(repo, DECISION_HEAD_CONFIG_NAME, _raise_exceptions_for_missing_entries=False):
        return None
    config = AutoConfig.from_pretrained(repo, _configuration_file=DECISION_HEAD_CONFIG_NAME)
    return AutoModel.from_pretrained(repo, config=config).to(model.device).eval()


class ForDecision(nn.Module):
    """
    Base of the `XxxForDecision` models: the model of a repo and its decision head, called with the inputs of a
    [`DecisionPrompt`] (`prompt.model_inputs()`) to score its options.
    """

    def __init__(self, model: PreTrainedModel, head: PreTrainedModel):
        super().__init__()
        self.model, self.head = model, head

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> "ForDecision":
        # The classes the decision pipeline loads, in its order
        for auto_class in (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel):
            try:
                model = auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
                break
            except ValueError:  # not a model of this class
                continue
        decision_config = DecisionConfig.from_pretrained(pretrained_model_name_or_path)
        head = load_decision_head(model, pretrained_model_name_or_path, decision_config)
        if head is None:
            raise ValueError(f"{pretrained_model_name_or_path} has no `{DECISION_HEAD_CONFIG_NAME}`.")
        return cls(model, head)

    def forward(self, **inputs):
        return self.head(**inputs, backbone=self.model)


__all__ = ["ForDecision", "load_decision_head"]
