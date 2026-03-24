# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
"""XLM configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="FacebookAI/xlm-mlm-en-2048")
@strict
class XLMConfig(PreTrainedConfig):
    r"""
    gelu_activation (`bool`, *optional*, defaults to `True`):
        Whether or not to use *gelu* for the activations instead of *relu*.
    sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
        Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
    causal (`bool`, *optional*, defaults to `False`):
        Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
        order to only attend to the left-side context instead if a bidirectional context.
    asm (`bool`, *optional*, defaults to `False`):
        Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
        layer.
    n_langs (`int`, *optional*, defaults to 1):
        The number of languages the model handles. Set to 1 for monolingual models.
    use_lang_emb (`bool`, *optional*, defaults to `True`):
        Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual
        models page](http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings) for information
        on how to use them.
    embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
        The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
    unk_index (`int`, *optional*, defaults to 3):
        The index of the unknown token in the vocabulary.
    mask_index (`int`, *optional*, defaults to 5):
        The index of the masking token in the vocabulary.
    is_encoder (`bool`, *optional*, defaults to `True`):
        Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
    summary_type (`string`, *optional*, defaults to "first"):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Used in the sequence classification and multiple choice models.
        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Used in the sequence classification and multiple choice models.
        The dropout ratio to be used after the projection and activation.
    start_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    end_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    mask_token_id (`int`, *optional*, defaults to 0):
        Model agnostic parameter to identify masked tokens when generating text in an MLM context.
    lang_id (`int`, *optional*, defaults to 1):
        The ID of the language used by the model. This parameter is used when generating text in a given language.

    Examples:

    ```python
    >>> from transformers import XLMConfig, XLMModel

    >>> # Initializing a XLM configuration
    >>> configuration = XLMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlm"
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # For backward compatibility
        "bos_index": "bos_token_id",
        "eos_index": "eos_token_id",
        "pad_index": "pad_token_id",
    }

    vocab_size: int = 30145
    emb_dim: int = 2048
    n_layers: int = 12
    n_heads: int = 16
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    gelu_activation: bool = True
    sinusoidal_embeddings: bool = False
    causal: bool = False
    asm: bool = False
    n_langs: int = 1
    use_lang_emb: bool = True
    max_position_embeddings: int = 512
    embed_init_std: float = 2048**-0.5
    layer_norm_eps: float = 1e-12
    init_std: float = 0.02
    unk_index: int = 3
    mask_index: int = 5
    is_encoder: bool = True
    summary_type: str = "first"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    start_n_top: int = 5
    end_n_top: int = 5
    mask_token_id: int | None = 0
    lang_id: int = 0
    pad_token_id: int | None = 2
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = True


__all__ = ["XLMConfig"]
