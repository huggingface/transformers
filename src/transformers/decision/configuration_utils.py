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
"""Settings of decision models, saved as `decision.json` and `decision_template.jinja`."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from huggingface_hub.dataclasses import strict

from ..utils import cached_file


# Next to the model files, without clashing with them: the config of the head is read with `_configuration_file`
DECISION_CONFIG_NAME, DECISION_TEMPLATE_NAME = "decision.json", "decision_template.jinja"
DECISION_HEAD_CONFIG_NAME = "decision_head.json"


@strict
@dataclass
class DecisionConfig:
    """
    Settings of a decision model, saved as `decision.json` (and its chat template as `decision_template.jinja`).

    Args:
        labels (`dict[str, list[str]]`, *optional*):
            Single-token labels of the options per question type, for heads that read them from the logits. Defaults to
            `Yes`/`No` for `noul` and `A`, `B`, `C`, ... otherwise.
        temperature (`dict[str, float]`, *optional*):
            Temperature per question type, applied to the option scores before the softmax. Defaults to 1.
        calibration (`dict[str, float]`, *optional*):
            Input-conditioned temperature (`bias`, `entropy`, `log_tokens`, `n_options`, `lo`, `hi`), see
            [`CalibratedTemperatureLogitsWarper`].
        noul_band (`list[float]`, *optional*):
            `[edge, slope]` of the band that keeps the probability of `noul` answers away from 0.5, see [`NoulBandLogitsWarper`].
        chat_template (`str`, *optional*):
            Jinja template rendering `state` with `question` (one prompt per question) or `questions` (one prompt for
            all of them). Defaults to a generic prompt in the tokenizer's chat template.
        noul_options (`list[str]`, *optional*, defaults to `["true", "false"]`):
            Order in which the template receives the two options of a `noul` question.
        sort_choices (`bool`, *optional*, defaults to `False`):
            Whether the template receives the options of a `choice` question sorted by key.
        sanitize_control_tokens (`bool`, *optional*, defaults to `True`):
            Whether control tokens in the request are encoded as text. Templates that escape them themselves disable it,
            to keep the input ids of their original code.
        adapter (`str`, *optional*):
            Subfolder of the LoRA that the pipeline loads on the model, for repos that don't keep it at their root.
    """

    labels: dict[str, list[str]] = field(default_factory=dict)
    temperature: dict[str, float | int] = field(default_factory=dict)
    calibration: dict[str, float | int] | None = None
    noul_band: list[float] | None = None
    chat_template: str | None = None
    noul_options: list[str] = field(default_factory=lambda: ["true", "false"])
    sort_choices: bool = False
    sanitize_control_tokens: bool = True
    adapter: str | None = None

    def validate_noul_options(self):
        if sorted(self.noul_options) != ["false", "true"]:
            raise ValueError(f"noul_options must order 'true' and 'false', got {self.noul_options!r}.")

    def get_logits_processor(self, question_type: str):
        """The [`LogitsProcessorList`] applied to the option scores of a question of type `question_type`."""
        from ..generation import (
            CalibratedTemperatureLogitsWarper,
            LogitsProcessorList,
            NoulBandLogitsWarper,
            TemperatureLogitsWarper,
        )

        processors = LogitsProcessorList([TemperatureLogitsWarper(float(self.temperature.get(question_type, 1.0)))])
        if self.calibration:
            processors.append(CalibratedTemperatureLogitsWarper(**self.calibration))
        if self.noul_band and question_type == "noul":
            processors.append(NoulBandLogitsWarper(*self.noul_band, true_index=self.noul_options.index("true")))
        return processors

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "DecisionConfig":
        config = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        # Settings this version doesn't know (e.g. older exports' `sort_choices`) are ignored
        return cls(**{key: value for key, value in config.items() if key in {f.name for f in fields(cls)}})

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, config_name: str = DECISION_CONFIG_NAME, **kwargs
    ) -> "DecisionConfig":
        config = cls.from_file(cached_file(pretrained_model_name_or_path, config_name, **kwargs))
        kwargs["_raise_exceptions_for_missing_entries"] = False
        if (template := cached_file(pretrained_model_name_or_path, DECISION_TEMPLATE_NAME, **kwargs)) is not None:
            config.chat_template = Path(template).read_text(encoding="utf-8")
        return config

    def save_pretrained(self, save_directory: str | Path):
        path, config = Path(save_directory), asdict(self)
        path.mkdir(parents=True, exist_ok=True)
        if (template := config.pop("chat_template")) is not None:
            (path / DECISION_TEMPLATE_NAME).write_text(template, encoding="utf-8")
        (path / DECISION_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


__all__ = ["DecisionConfig"]
