# Copyright 2025 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""ConvNeXT model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
class DINOv3ConvNextConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    Example:
    ```python
    >>> from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    >>> # Initializing a DINOv3ConvNext (tiny variant) style configuration
    >>> config = DINOv3ConvNextConfig()

    >>> # Initializing a model (with random weights)
    >>> model = DINOv3ConvNextModel(config)

    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    model_type = "dinov3_convnext"

    def __init__(
        self,
        num_channels: int = 3,
        hidden_sizes: list[int] | None = None,
        depths: list[int] | None = None,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,
        layer_scale_init_value: float = 1e-6,
        drop_path_rate: float = 0.0,
        image_size: int = 224,
        out_features: list[str] | None = None,
        out_indices: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)

    @property
    def num_stages(self) -> int:
        return len(self.hidden_sizes)


__all__ = ["DINOv3ConvNextConfig"]
