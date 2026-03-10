# Copyright 2022 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""ERNIE model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nghuyong/ernie-3.0-base-zh")
class ErnieConfig(PreTrainedConfig):
    r"""
    task_type_vocab_size (`int`, *optional*, defaults to 3):
        The vocabulary size of the `task_type_ids` for ERNIE2.0/ERNIE3.0 model
    use_task_id (`bool`, *optional*, defaults to `False`):
        Whether or not the model support `task_type_ids`

    Examples:

    ```python
    >>> from transformers import ErnieConfig, ErnieModel

    >>> # Initializing a ERNIE nghuyong/ernie-3.0-base-zh style configuration
    >>> configuration = ErnieConfig()

    >>> # Initializing a model (with random weights) from the nghuyong/ernie-3.0-base-zh style configuration
    >>> model = ErnieModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie"

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
        task_type_vocab_size=3,
        use_task_id=False,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.task_type_vocab_size = task_type_vocab_size
        self.use_task_id = use_task_id
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


__all__ = ["ErnieConfig"]
