# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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
""" Bloom configuration"""
from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-350m": "https://huggingface.co/bigscience/bloom-350m/blob/main/config.json",
    "bigscience/bloom-760m": "https://huggingface.co/bigscience/bloom-760m/blob/main/config.json",
    "bigscience/bloom-1b3": "https://huggingface.co/bigscience/bloom-1b3/blob/main/config.json",
    "bigscience/bloom-2b5": "https://huggingface.co/bigscience/bloom-2b5/blob/main/config.json",
    "bigscience/bloom-6b3": "https://huggingface.co/bigscience/bloom-6b3/blob/main/config.json",
}


class BloomConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the Bloom model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BloomModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        skip_bias_add (`bool`, *optional*, defaults to `True`):
            If set to `True`, it will skip bias add for each linear layer in the transformer blocks
        skip_bias_add_qkv (`bool`, *optional*, defaults to `False`):
            If set to `True`, it will skip bias add for the first linear layer in the transformer blocks
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            If set to `True` and the `dtype` is set to `float16` it will scale the input of the Softmax function to
            `fp32`
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Precision that has been used for the model's training in Megatron. Please load the model in the correct
            precision by doing `model = BloomModel.from_pretrained(model_name, torch_dtype="auto")`.`
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.

    Example:

    ```python
    >>> from transformers import BloomModel, BloomConfig

    >>> # Initializing a Bloom configuration
    >>> configuration = BloomConfig()

    >>> # Initializing a model from the configuration
    >>> model = BloomModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "n_head": "num_attention_heads",
        "hidden_size": "n_embed",
        "dtype": "torch_dtype",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        pretraining_tp=1,  # TP rank used when training with megatron
        dtype="bfloat16",
        slow_but_exact=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dtype = dtype
        self.slow_but_exact = slow_but_exact

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
