# Copyright 2024 the Fast authors and HuggingFace Inc. team.  All rights reserved.
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
"""TextNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="czczup/textnet-base")
@strict
class TextNetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    stem_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size for the initial convolution layer.
    stem_stride (`int`, *optional*, defaults to 2):
        The stride for the initial convolution layer.
    stem_num_channels (`int`, *optional*, defaults to 3):
        The num of channels in input for the initial convolution layer.
    stem_out_channels (`int`, *optional*, defaults to 64):
        The num of channels in out for the initial convolution layer.
    stem_act_func (`str`, *optional*, defaults to `"relu"`):
        The activation function for the initial convolution layer.
    conv_layer_kernel_sizes (`list[list[list[int]]]`, *optional*):
        A list of stage-wise kernel sizes. If `None`, defaults to:
        `[[[3, 3], [3, 3], [3, 3]], [[3, 3], [1, 3], [3, 3], [3, 1]], [[3, 3], [3, 3], [3, 1], [1, 3]], [[3, 3], [3, 1], [1, 3], [3, 3]]]`.
    conv_layer_strides (`list[list[int]]`, *optional*):
        A list of stage-wise strides. If `None`, defaults to:
        `[[1, 2, 1], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]]`.

    Examples:

    ```python
    >>> from transformers import TextNetConfig, TextNetBackbone

    >>> # Initializing a TextNetConfig
    >>> configuration = TextNetConfig()

    >>> # Initializing a model (with random weights)
    >>> model = TextNetBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "textnet"

    stem_kernel_size: int = 3
    stem_stride: int = 2
    stem_num_channels: int = 3
    stem_out_channels: int = 64
    stem_act_func: str = "relu"
    image_size: list[int] | tuple[int, int] | int = (640, 640)
    conv_layer_kernel_sizes: list | None = None
    conv_layer_strides: list | None = None
    hidden_sizes: list[int] | tuple[int, ...] = (64, 64, 128, 256, 512)
    batch_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.conv_layer_kernel_sizes is None:
            self.conv_layer_kernel_sizes = [
                [[3, 3], [3, 3], [3, 3]],
                [[3, 3], [1, 3], [3, 3], [3, 1]],
                [[3, 3], [3, 3], [3, 1], [1, 3]],
                [[3, 3], [3, 1], [1, 3], [3, 3]],
            ]
        if self.conv_layer_strides is None:
            self.conv_layer_strides = [[1, 2, 1], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]]

        self.depths = [len(layer) for layer in self.conv_layer_kernel_sizes]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, 5)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["TextNetConfig"]
