# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""XGLM model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/xglm-564M")
class XGLMConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import XGLMModel, XGLMConfig

    >>> # Initializing a XGLM facebook/xglm-564M style configuration
    >>> configuration = XGLMConfig()

    >>> # Initializing a model from the facebook/xglm-564M style configuration
    >>> model = XGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xglm"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=256008,
        max_position_embeddings=2048,
        d_model=1024,
        ffn_dim=4096,
        num_layers=24,
        attention_heads=16,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layerdrop=0.0,
        init_std=0.02,
        scale_embedding=True,
        use_cache=True,
        decoder_start_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.decoder_start_token_id = decoder_start_token_id

        super().__init__(**kwargs)


__all__ = ["XGLMConfig"]
