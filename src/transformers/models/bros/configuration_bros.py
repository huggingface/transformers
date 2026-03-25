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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="jinho8345/bros-base-uncased")
@strict
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

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    dim_bbox: int = 8
    bbox_scale: float = 100.0
    n_relations: int = 1
    classifier_dropout_prob: float = 0.1
    is_decoder: bool = False
    add_cross_attention: bool = False

    def __post_init__(self, **kwargs):
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["BrosConfig"]
