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
"""MGP-STR model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="alibaba-damo/mgp-str-base")
class MgpstrConfig(PreTrainedConfig):
    r"""
    max_token_length (`int`, *optional*, defaults to 27):
        The max number of output tokens.
    num_character_labels (`int`, *optional*, defaults to 38):
        The number of classes for character head .
    num_bpe_labels (`int`, *optional*, defaults to 50257):
        The number of classes for bpe head .
    num_wordpiece_labels (`int`, *optional*, defaults to 30522):
        The number of classes for wordpiece head .
    distilled (`bool`, *optional*, defaults to `False`):
        Model includes a distillation token and head as in DeiT models.
    drop_rate (`float`, *optional*, defaults to 0.0):
        The dropout probability for all fully connected layers in the embeddings, encoder.
    attn_drop_rate (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    output_a3_attentions (`bool`, *optional*, defaults to `False`):
        Whether or not the model should returns A^3 module attentions.

    Example:

    ```python
    >>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

    >>> # Initializing a Mgpstr mgp-str-base style configuration
    >>> configuration = MgpstrConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MgpstrForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mgp-str"

    def __init__(
        self,
        image_size=[32, 128],
        patch_size=4,
        num_channels=3,
        max_token_length=27,
        num_character_labels=38,
        num_bpe_labels=50257,
        num_wordpiece_labels=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        layer_norm_eps=1e-5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        output_a3_attentions=False,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_token_length = max_token_length
        self.num_character_labels = num_character_labels
        self.num_bpe_labels = num_bpe_labels
        self.num_wordpiece_labels = num_wordpiece_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.layer_norm_eps = layer_norm_eps
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.output_a3_attentions = output_a3_attentions
        self.initializer_range = initializer_range


__all__ = ["MgpstrConfig"]
