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
"""Swinv2 Transformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/swinv2-tiny-patch4-window8-256")
@strict
class Swinv2Config(BackboneConfigMixin, PreTrainedConfig):
    r"""
    window_size (`int`, *optional*, defaults to 7):
        Size of windows.
    pretrained_window_sizes (`list(int)`, *optional*, defaults to `[0, 0, 0, 0]`):
        Size of windows during pretraining.
    encoder_stride (`int`, *optional*, defaults to 32):
        Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import Swinv2Config, Swinv2Model

    >>> # Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
    >>> configuration = Swinv2Config()

    >>> # Initializing a model (with random weights) from the microsoft/swinv2-tiny-patch4-window8-256 style configuration
    >>> model = Swinv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swinv2"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 96
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    num_heads: list[int] | tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    pretrained_window_sizes: list[int] | tuple[int, ...] = (0, 0, 0, 0)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    encoder_stride: int = 32
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_layers = len(self.depths)
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        # we set the hidden_size attribute in order to make Swinv2 work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        super().__post_init__(**kwargs)


__all__ = ["Swinv2Config"]
