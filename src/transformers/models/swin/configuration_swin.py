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
"""Swin Transformer model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/swin-tiny-patch4-window7-224")
class SwinConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    depths (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`):
        Depth of each layer in the Transformer encoder.
    num_heads (`list(int)`, *optional*, defaults to `[3, 6, 12, 24]`):
        Number of attention heads in each layer of the Transformer encoder.
    window_size (`int`, *optional*, defaults to 7):
        Size of windows.
    encoder_stride (`int`, *optional*, defaults to 32):
        Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import SwinConfig, SwinModel

    >>> # Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> configuration = SwinConfig()

    >>> # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> model = SwinModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["SwinConfig"]
