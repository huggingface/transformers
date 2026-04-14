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
"""Dilated Neighborhood Attention Transformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="shi-labs/dinat-mini-in1k-224")
@strict
class DinatConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    dilations (`list[list[int]]`, *optional*, defaults to `[[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]]`):
        Dilation value of each NA layer in the Transformer encoder.

    Example:

    ```python
    >>> from transformers import DinatConfig, DinatModel

    >>> # Initializing a Dinat shi-labs/dinat-mini-in1k-224 style configuration
    >>> configuration = DinatConfig()

    >>> # Initializing a model (with random weights) from the shi-labs/dinat-mini-in1k-224 style configuration
    >>> model = DinatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dinat"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 64
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 5)
    num_heads: list[int] | tuple[int, ...] = (2, 4, 8, 16)
    kernel_size: int = 7
    dilations: list | tuple | None = None
    mlp_ratio: float = 3.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    layer_scale_init_value: float = 0.0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_layers = len(self.depths)
        self.dilations = self.dilations or [[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]]

        # we set the hidden_size attribute in order to make Dinat work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["DinatConfig"]
