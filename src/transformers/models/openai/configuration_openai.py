# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""OpenAI GPT configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openai-community/openai-gpt")
class OpenAIGPTConfig(PreTrainedConfig):
    """
    afn (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers
    summary_type (`str`, *optional*, defaults to `"cls_index"`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        The dropout ratio to be used after the projection and activation.


    Examples:

    ```python
    >>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

    >>> # Initializing a GPT configuration
    >>> configuration = OpenAIGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OpenAIGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "openai-gpt"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=40478,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        afn="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.afn = afn
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["OpenAIGPTConfig"]
