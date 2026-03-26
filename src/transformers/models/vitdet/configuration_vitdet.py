# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""VitDet model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/vitdet-base-patch16-224")
@strict
class VitDetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    pretrain_image_size (`int`, *optional*, defaults to 224):
        The size (resolution) of each image during pretraining.
    window_block_indices (`list[int]`, *optional*, defaults to `[]`):
        List of indices of blocks that should have window attention instead of regular global self-attention.
    residual_block_indices (`list[int]`, *optional*, defaults to `[]`):
        List of indices of blocks that should have an extra residual block after the MLP.
    use_relative_position_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to add relative position embeddings to the attention maps.
    window_size (`int`, *optional*, defaults to 0):
        The size of the attention window.

    Example:

    ```python
    >>> from transformers import VitDetConfig, VitDetModel

    >>> # Initializing a VitDet configuration
    >>> configuration = VitDetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VitDetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitdet"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 224
    pretrain_image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    window_block_indices: list[int] | tuple[int, ...] = ()
    residual_block_indices: list[int] | tuple[int, ...] = ()
    use_absolute_position_embeddings: bool = True
    use_relative_position_embeddings: bool = False
    window_size: int = 0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["VitDetConfig"]
