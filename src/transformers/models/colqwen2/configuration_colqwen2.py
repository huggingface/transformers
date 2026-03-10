# Copyright 2025 The HuggingFace Inc. team.
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


from copy import deepcopy
from typing import Any

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="vidore/colqwen2-v1.0-hf")
class ColQwen2Config(PreTrainedConfig):
    r"""
    Example:

    ```python
    from transformers.models.colqwen2 import ColQwen2Config, ColQwen2ForRetrieval

    config = ColQwen2Config()
    model = ColQwen2ForRetrieval(config)
    ```
    """

    model_type = "colqwen2"
    sub_configs: dict[str, Any] = {"vlm_config": PreTrainedConfig}

    def __init__(
        self,
        vlm_config=None,
        embedding_dim: int = 128,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if vlm_config is None:
            vlm_config = CONFIG_MAPPING["qwen2_vl"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `Qwen2VLConfig` with default values."
            )
        elif isinstance(vlm_config, dict):
            vlm_config = deepcopy(vlm_config)
            if "model_type" not in vlm_config:
                raise KeyError(
                    "The `model_type` key is missing in the `vlm_config` dictionary. Please provide the model type."
                )
            vlm_config = CONFIG_MAPPING[vlm_config["model_type"]](**vlm_config)
        elif not isinstance(vlm_config, PreTrainedConfig):
            raise TypeError(
                f"Invalid type for `vlm_config`. Expected `PreTrainedConfig`, `dict`, or `None`, but got {type(vlm_config)}."
            )

        if not hasattr(vlm_config, "vocab_size"):
            vlm_config.vocab_size = vlm_config.get_text_config().vocab_size

        self.vlm_config = vlm_config
        self.embedding_dim = embedding_dim
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> PreTrainedConfig:
        return self.vlm_config.get_text_config(*args, **kwargs)


__all__ = ["ColQwen2Config"]
