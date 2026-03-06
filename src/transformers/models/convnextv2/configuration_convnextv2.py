# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""ConvNeXTV2 model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/convnextv2-tiny-1k-224")
class ConvNextV2Config(BackboneConfigMixin, PreTrainedConfig):
    r"""
    num_stages (`int`, *optional*, defaults to 4):
        The number of stages in the model.

    Example:
    ```python
    >>> from transformers import ConvNeXTV2Config, ConvNextV2Model

    >>> # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
    >>> configuration = ConvNeXTV2Config()

    >>> # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
    >>> model = ConvNextV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "convnextv2"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=224,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["ConvNextV2Config"]
