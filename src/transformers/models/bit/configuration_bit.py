# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""BiT model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/bit-50")
class BitConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    layer_type (`str`, *optional*, defaults to `"preactivation"`):
        The layer to use, it can be either `"preactivation"` or `"bottleneck"`.
    embedding_dynamic_padding (`bool`, *optional*, defaults to `False`):
        Whether or not to make use of dynamic padding for the embedding layer.
    width_factor (`int`, *optional*, defaults to 1):
        The width factor for the model.
    global_padding (`str`, *optional*):
        Padding strategy to use for the convolutional layers. Can be either `"valid"`, `"same"`, or `None`.
    num_groups (`int`, *optional*, defaults to 32):
        Number of groups used for the `BitGroupNormActivation` layers.

    Example:
    ```python
    >>> from transformers import BitConfig, BitModel

    >>> # Initializing a BiT bit-50 style configuration
    >>> configuration = BitConfig()

    >>> # Initializing a model (with random weights) from the bit-50 style configuration
    >>> model = BitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bit"
    layer_types = ["preactivation", "bottleneck"]
    supported_padding = ["SAME", "VALID"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="preactivation",
        hidden_act="relu",
        global_padding=None,
        num_groups=32,
        drop_path_rate=0.0,
        embedding_dynamic_padding=False,
        output_stride=32,
        width_factor=1,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        if global_padding is not None:
            if global_padding.upper() in self.supported_padding:
                global_padding = global_padding.upper()
            else:
                raise ValueError(f"Padding strategy {global_padding} not supported")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.global_padding = global_padding
        self.num_groups = num_groups
        self.drop_path_rate = drop_path_rate
        self.embedding_dynamic_padding = embedding_dynamic_padding
        self.output_stride = output_stride
        self.width_factor = width_factor

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["BitConfig"]
