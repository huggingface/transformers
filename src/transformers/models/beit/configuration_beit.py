# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""BEiT model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/beit-base-patch16-224-pt22k")
@strict
class BeitConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token for masked image modeling.
    use_relative_position_bias (`bool`, *optional*, defaults to `False`):
        Whether to use T5-style relative position embeddings in the self-attention layers.
    use_shared_relative_position_bias (`bool`, *optional*, defaults to `False`):
        Whether to use the same relative position embeddings across all self-attention layers of the Transformer.
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
        CLS token, before applying the classification head.
    pool_scales (`tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
        Pooling scales used in Pooling Pyramid Module applied on the last feature map.
    use_auxiliary_head (`bool`, *optional*, defaults to `True`):
        Whether to use an auxiliary head during training.
    auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
        Weight of the cross-entropy loss of the auxiliary head.
    auxiliary_channels (`int`, *optional*, defaults to 256):
        Number of channels to use in the auxiliary head.
    auxiliary_num_convs (`int`, *optional*, defaults to 1):
        Number of convolutional layers to use in the auxiliary head.
    auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
        Whether to concatenate the output of the auxiliary head with the input before the classification layer.
    add_fpn (`bool`, *optional*, defaults to `False`):
        Whether to add a FPN as part of the backbone. Only relevant for [`BeitBackbone`].
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
        case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`. Only relevant for [`BeitBackbone`].

    Example:

    ```python
    >>> from transformers import BeitConfig, BeitModel

    >>> # Initializing a BEiT beit-base-patch16-224-pt22k style configuration
    >>> configuration = BeitConfig()

    >>> # Initializing a model (with random weights) from the beit-base-patch16-224-pt22k style configuration
    >>> model = BeitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "beit"

    vocab_size: int = 8192
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = False
    use_relative_position_bias: bool = False
    use_shared_relative_position_bias: bool = False
    layer_scale_init_value: float = 0.1
    drop_path_rate: float = 0.1
    use_mean_pooling: bool = True
    pool_scales: list[int] | tuple[int, ...] = (1, 2, 3, 6)
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False
    semantic_loss_ignore_index: int = 255
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    add_fpn: bool = False
    reshape_hidden_states: bool = True

    def __post_init__(self, **kwargs):
        if "segmentation_indices" in kwargs and kwargs.get("out_indices") is None:
            kwargs["out_indices"] = kwargs.pop("segmentation_indices")

        # backbone attributes
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )

        super().__post_init__(**kwargs)


__all__ = ["BeitConfig"]
