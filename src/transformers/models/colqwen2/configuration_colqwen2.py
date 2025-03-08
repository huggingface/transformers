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
from typing import Any, Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class ColQwen2Config(PretrainedConfig):
    r"""
    Configuration class to store the configuration of a [`ColQ2en2ForRetrieval`]. It is used to instantiate an instance
    of `ColQwen2ForRetrieval` according to the specified arguments, defining the model architecture following the methodology
    from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vlm_config (`PretrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.

    Example:

    ```python
    from transformers.models.colqwen2 import ColQwen2Config, ColQwen2ForRetrieval

    config = ColQwen2Config()
    model = ColQwen2ForRetrieval(config)
    ```
    """

    model_type = "colqwen2"
    sub_configs: Dict[str, Any] = {"vlm_config": PretrainedConfig}

    def __init__(
        self,
        vlm_config=None,
        embedding_dim: int = 128,
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
            elif vlm_config["model_type"] != "qwen2_vl":
                raise ValueError(
                    f"Invalid model type for `vlm_config`. Expected `qwen2_vl`, but got {vlm_config['model_type']}."
                )
            vlm_config = CONFIG_MAPPING["qwen2_vl"](**vlm_config)
        elif isinstance(vlm_config, PretrainedConfig):
            if vlm_config.model_type != "qwen2_vl":
                raise ValueError(
                    f"Invalid model type for `vlm_config`. Expected `qwen2_vl`, but got {vlm_config.model_type}."
                )
            vlm_config = vlm_config
        else:
            raise TypeError(
                f"Invalid type for `vlm_config`. Expected `PretrainedConfig`, `dict`, or `None`, but got {type(vlm_config)}."
            )

        self.vlm_config = vlm_config
        self.embedding_dim = embedding_dim
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.vlm_config


__all__ = ["ColQwen2Config"]
