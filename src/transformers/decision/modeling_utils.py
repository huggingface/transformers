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
"""Decision models: the model of a repo and a decision head, like a `ForCausalLM` is the model and its `lm_head`."""

from pathlib import Path

from ..modeling_utils import PreTrainedModel
from ..utils import cached_file
from .configuration_utils import DECISION_HEAD_CONFIG_NAME, DecisionConfig


class DecisionPreTrainedModel(PreTrainedModel):
    """
    Base of the `XxxForDecision` models. Their files can sit next to the ones of the model they read, in its repo: the
    config is then `decision_head.json`, which names the weights to load (`transformers_weights`, e.g. an index of the
    shards of the model and of the file of the head), and `decision.json` can name the subfolder of a LoRA to load.
    """

    # Like the composite models: the model they read is loaded with its own attention implementation, and checked for it
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args, **kwargs):
        if kwargs.get("config") is None and cached_file(
            pretrained_model_name_or_path, DECISION_HEAD_CONFIG_NAME, _raise_exceptions_for_missing_entries=False
        ):
            kwargs["config"] = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, _configuration_file=DECISION_HEAD_CONFIG_NAME
            )
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        try:
            adapter = DecisionConfig.from_pretrained(pretrained_model_name_or_path).adapter
        except OSError:
            adapter = None
        if adapter:
            model.load_adapter(pretrained_model_name_or_path, adapter_kwargs={"subfolder": adapter})
        return model


__all__ = ["DecisionPreTrainedModel"]
