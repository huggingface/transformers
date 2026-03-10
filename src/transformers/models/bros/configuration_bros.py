# Copyright 2023-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
"""Bros model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="jinho8345/bros-base-uncased")
class BrosConfig(PreTrainedConfig):
    r"""
    dim_bbox (`int`, *optional*, defaults to 8):
        The dimension of the bounding box coordinates. (x0, y1, x1, y0, x1, y1, x0, y1)
    bbox_scale (`float`, *optional*, defaults to 100.0):
        The scale factor of the bounding box coordinates.
    n_relations (`int`, *optional*, defaults to 1):
        The number of relations for SpadeEE(entity extraction), SpadeEL(entity linking) head.

    Examples:

    ```python
    >>> from transformers import BrosConfig, BrosModel

    >>> # Initializing a BROS jinho8345/bros-base-uncased style configuration
    >>> configuration = BrosConfig()

    >>> # Initializing a model from the jinho8345/bros-base-uncased style configuration
    >>> model = BrosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bros"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dim_bbox=8,
        bbox_scale=100.0,
        n_relations=1,
        classifier_dropout_prob=0.1,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dim_bbox = dim_bbox
        self.bbox_scale = bbox_scale
        self.n_relations = n_relations
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        self.classifier_dropout_prob = classifier_dropout_prob


__all__ = ["BrosConfig"]
