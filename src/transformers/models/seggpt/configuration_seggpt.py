# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""SegGpt model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BAAI/seggpt-vit-large")
@strict
class SegGptConfig(PreTrainedConfig):
    r"""
    mlp_dim (`int`, *optional*):
        The dimensionality of the MLP layer in the Transformer encoder. If unset, defaults to
        `hidden_size` * 4.
    pretrain_image_size (`int`, *optional*, defaults to 224):
        The pretrained size of the absolute position embeddings.
    use_relative_position_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to use relative position embeddings in the attention layers.
    merge_index (`int`, *optional*, defaults to 2):
        The index of the encoder layer to merge the embeddings.
    intermediate_hidden_state_indices (`list[int]`, *optional*, defaults to `[5, 11, 17, 23]`):
        The indices of the encoder layers which we store as features for the decoder.
    beta (`float`, *optional*, defaults to 0.01):
        Regularization factor for SegGptLoss (smooth-l1 loss).

    Example:

    ```python
    >>> from transformers import SegGptConfig, SegGptModel

    >>> # Initializing a SegGPT seggpt-vit-large style configuration
    >>> configuration = SegGptConfig()

    >>> # Initializing a model (with random weights) from the seggpt-vit-large style configuration
    >>> model = SegGptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seggpt"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, ...] = (896, 448)
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    mlp_dim: int | None = None
    drop_path_rate: float = 0.1
    pretrain_image_size: int | list[int] | tuple[int, int] = 224
    decoder_hidden_size: int = 64
    use_relative_position_embeddings: bool = True
    merge_index: int = 2
    intermediate_hidden_state_indices: list[int] | tuple[int, ...] = (5, 11, 17, 23)
    beta: float = 0.01

    def __post_init__(self, **kwargs):
        self.mlp_dim = int(self.hidden_size * 4) if self.mlp_dim is None else self.mlp_dim
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.merge_index > min(self.intermediate_hidden_state_indices):
            raise ValueError(
                f"Merge index must be less than the minimum encoder output index, but got {self.merge_index=} and {self.intermediate_hidden_state_indices=}"
            )


__all__ = ["SegGptConfig"]
