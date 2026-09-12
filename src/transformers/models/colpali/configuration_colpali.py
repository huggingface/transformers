# Copyright 2024 The HuggingFace Inc. team.
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
"""ColPali model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="vidore/colpali-v1.2")
@strict
class ColPaliConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    from transformers.models.colpali import ColPaliConfig, ColPaliForRetrieval

    config = ColPaliConfig()
    model = ColPaliForRetrieval(config)
    ```
    """

    model_type = "colpali"
    sub_configs_defaults = {
        "vlm_config": SubConfigSpec(config_class=AutoConfig, model_type="paligemma"),
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="gemma"),
    }

    vlm_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    embedding_dim: int = 128


__all__ = ["ColPaliConfig"]
