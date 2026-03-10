# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
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
"""CodeGen model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/codegen-2B-mono")
class CodeGenConfig(PreTrainedConfig):
    r"""
    n_ctx (`int`, *optional*, defaults to 2048):
        This attribute is used in `CodeGenModel.__init__` without any real effect.
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of dimensions in the embedding that Rotary Position Embedding is applied to.
    n_inner (`int`, *optional*):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd

    Example:

    ```python
    >>> from transformers import CodeGenConfig, CodeGenModel

    >>> # Initializing a CodeGen 6B configuration
    >>> configuration = CodeGenConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CodeGenModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "codegen"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_ctx=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["CodeGenConfig"]
