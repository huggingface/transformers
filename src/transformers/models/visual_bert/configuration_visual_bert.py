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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="uclanlp/visualbert-vqa-coco-pre")
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

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        visual_embedding_dim=512,
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
        bypass_transformer=False,
        special_visual_initialize=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.visual_embedding_dim = visual_embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.bypass_transformer = bypass_transformer
        self.special_visual_initialize = special_visual_initialize


__all__ = ["VisualBertConfig"]
