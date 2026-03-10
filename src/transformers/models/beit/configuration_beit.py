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

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/beit-base-patch16-224-pt22k")
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

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        use_mask_token=False,
        use_absolute_position_embeddings=False,
        use_relative_position_bias=False,
        use_shared_relative_position_bias=False,
        layer_scale_init_value=0.1,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        semantic_loss_ignore_index=255,
        out_features=None,
        out_indices=None,
        add_fpn=False,
        reshape_hidden_states=True,
        **kwargs,
    ):
        if "segmentation_indices" in kwargs and out_indices is None:
            out_indices = kwargs.pop("segmentation_indices")
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_bias = use_relative_position_bias
        self.use_shared_relative_position_bias = use_shared_relative_position_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.use_mean_pooling = use_mean_pooling
        # decode head attributes (semantic segmentation)
        self.pool_scales = pool_scales
        # auxiliary head attributes (semantic segmentation)
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.semantic_loss_ignore_index = semantic_loss_ignore_index

        # backbone attributes
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)
        self.add_fpn = add_fpn
        self.reshape_hidden_states = reshape_hidden_states


__all__ = ["BeitConfig"]
