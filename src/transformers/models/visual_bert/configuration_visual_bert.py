# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""VisualBERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uclanlp/visualbert-vqa-coco-pre")
@strict
class VisualBertConfig(PreTrainedConfig):
    r"""
    visual_embedding_dim (`int`, *optional*, defaults to 512):
        Dimensionality of the visual embeddings to be passed to the model.
    bypass_transformer (`bool`, *optional*, defaults to `False`):
        Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
        model directly concatenates the visual embeddings from [`VisualBertEmbeddings`] with text output from
        transformers, and then pass it to a self-attention layer.
    special_visual_initialize (`bool`, *optional*, defaults to `True`):
        Whether or not the visual token type and position type embedding weights should be initialized the same as
        the textual token type and positive type embeddings. When set to `True`, the weights of the textual token
        type and position type embeddings are copied to the respective visual embedding layers.

    Example:

    ```python
    >>> from transformers import VisualBertConfig, VisualBertModel

    >>> # Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
    >>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    >>> # Initializing a model (with random weights) from the visualbert-vqa-coco-pre style configuration
    >>> model = VisualBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "visual_bert"

    vocab_size: int = 30522
    hidden_size: int = 768
    visual_embedding_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    bypass_transformer: bool = False
    special_visual_initialize: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["VisualBertConfig"]
