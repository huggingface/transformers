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
"""
`decide`, the `generate` of decision models: like the `GenerationMixin` of the models that generate.
"""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from ..configuration_utils import PreTrainedConfig
from ..utils import cached_file
from .configuration_utils import DecisionConfig
from .template_utils import Question


def batch_prompts(prompts: list[tuple[dict, list[Question]]]) -> dict[str, torch.Tensor]:
    """
    The prompts of a request as one right-padded batch, with `logits_to_keep` the mask of the positions read: the
    token before each mark, so the read positions of a batch are its questions in order.
    """
    inputs = [prompt for prompt, _ in prompts]
    reads = []
    for prompt in inputs:
        read = torch.zeros(prompt["input_ids"].shape[-1], dtype=torch.bool, device=prompt["input_ids"].device)
        read[prompt["marks"] - 1] = True
        reads.append(read)
    batch = {
        "attention_mask": pad_sequence([torch.ones_like(read, dtype=torch.long) for read in reads], batch_first=True),
        "logits_to_keep": pad_sequence(reads, batch_first=True),
    }
    for key in inputs[0].keys() - {"marks", "attention_mask"}:
        values = [prompt[key] for prompt in inputs]
        if values[0].shape[-1] == inputs[0]["input_ids"].shape[-1]:  # per token: `input_ids`, `mm_token_type_ids`...
            batch[key] = pad_sequence([value[0] for value in values], batch_first=True)
        else:  # ponytail: per image, already flat over images (`pixel_values`, `image_grid_thw`) like Qwen-VL
            batch[key] = torch.cat(values)
    return batch


def process_scores(decision_config, batch, questions, scores) -> list[list[torch.Tensor]]:
    """
    The logits processors of the decision config on the option scores of each question of each prompt of a batch from
    [`batch_prompts`], `scores` being those of all its questions in order.
    """
    scores, processed = iter(scores), []
    lengths = batch["attention_mask"].sum(-1).tolist()
    for input_ids, read, length, group in zip(batch["input_ids"], batch["logits_to_keep"], lengths, questions):
        tail = input_ids[int(read.nonzero()[-1]) + 1 : length]  # the tokens after the last mark: the state
        processed.append([decision_config.logits_processors[q.type](tail, next(scores).float()) for q in group])
    return processed


def _hub_kwargs(kwargs: dict) -> dict:
    return {key: kwargs[key] for key in ("cache_dir", "local_files_only", "revision", "token") if key in kwargs}


class DecisionMixin:
    """
    The `decide` of the `XxxForDecision` models, with their `decision_config`: the decision config of their repo, or the
    one of the checkpoint they read the head of, like `generation_config` for `generate`.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None, decision_config=None, **kwargs):
        """
        Also loads the checkpoints that ship a decision head next to their weights, in their own format: the config of
        the checkpoint is the `text_config` of the decision model, and the config files of its head
        (`config_class.extra_config_files`) give the rest of its config, in `__post_init__`.
        """
        if not isinstance(config, cls.config_class):  # none, or the config of the model the head ships next to
            head_config, decision_config = cls.checkpoint_config(pretrained_model_name_or_path, **kwargs)
            if head_config is None and config is not None:
                raise ValueError(f"{pretrained_model_name_or_path} has no {cls.__name__} head next to its weights.")
            config = head_config or config
        model = super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
        if decision_config is None:  # the decision config of the repo, read now like `generation_config`
            decision_config = DecisionConfig.from_pretrained(pretrained_model_name_or_path, **_hub_kwargs(kwargs))
        model.decision_config = decision_config
        return model

    @classmethod
    def checkpoint_config(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> tuple[PreTrainedConfig, DecisionConfig] | tuple[None, None]:
        """
        The config and decision config of the decision model of a checkpoint that ships its head next to its weights,
        read from its files, or `None`s for any other repo (e.g. one saved by `save_pretrained`, whose `config.json` is
        the one of the decision model).
        """
        hub_kwargs = _hub_kwargs(kwargs)
        head_configs = {}
        for filename in cls.config_class.extra_config_files:
            path = cached_file(
                pretrained_model_name_or_path, filename, _raise_exceptions_for_missing_entries=False, **hub_kwargs
            )
            if path is not None:
                head_configs[Path(filename).stem] = json.loads(Path(path).read_text(encoding="utf-8"))
        if not head_configs:
            return None, None
        text_config, _ = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **hub_kwargs)
        decision_config = cls.config_class.decision_config(head_configs, text_config)
        if decision_config is None or text_config.get("model_type") == cls.config_class.model_type:
            return None, None  # not a checkpoint of this decision model
        return cls.config_class(text_config=text_config, **head_configs), decision_config

    def post_init(self):
        """The default decision config, replaced by the one of the repo in `from_pretrained`, like `generation_config`."""
        super().post_init()
        self.decision_config = DecisionConfig()

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Also saves the decision config, like `generation_config`: the saved repo reloads as the same decision model."""
        super().save_pretrained(save_directory, *args, **kwargs)
        if kwargs.get("is_main_process", True):
            self.decision_config.save_pretrained(save_directory)

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.LongTensor,
        logits_to_keep: torch.BoolTensor,
        decision_questions: list[list[Question]],
        decision_config: DecisionConfig | None = None,
        **kwargs,
    ) -> list[list[torch.Tensor]]:
        """
        Scores the options of each question of a batch of prompts from [`batch_prompts`]: one forward pass, then the
        logits processors of the decision config. Returns the scores of each question of each prompt, which
        `Question.answer` turns into answers.
        """
        decision_config = decision_config or self.decision_config
        batch = {"input_ids": input_ids, "logits_to_keep": logits_to_keep, **kwargs}
        batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
        questions = [question for group in decision_questions for question in group]
        num_options = [len(question.options) for question in questions]
        question_types = [Question.questions_types.index(question.type) for question in questions]
        scores = self(
            num_options=torch.tensor(num_options, device=self.device),
            question_type_ids=torch.tensor(question_types, device=self.device),
            **batch,
        )
        return process_scores(decision_config, batch, decision_questions, scores.split(num_options))


__all__ = ["DecisionMixin"]
