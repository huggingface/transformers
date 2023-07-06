# coding=utf-8
# Copyright 2023 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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
""" Mpt configuration"""
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mosaicml/mpt-7b": "https://huggingface.co/mosaicml/mpt-7b/resolve/main/config.json",
}


class MptConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Mpt architecture
    [bigscience/mpt](https://huggingface.co/bigscience/mpt).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MptModel`]. Check [this
            discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
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
    >>> from transformers import MptConfig, MptModel

    >>> # Initializing a Mpt configuration
    >>> configuration = MptConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


"""A HuggingFace-style model configuration."""
from typing import Optional, Union

from transformers import PretrainedConfig


class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    def __init__(
        self,
        attn_type="multihead_attention",
        attn_pdrop=0,
        attn_impl="triton",
        normalise_query_key=False,
        clip_query_key_value=None,
        softmax_scale=None,  # TODO rename
        prefix_lm=False,  # TODO what is this
        attn_uses_sequence_id=False,
        alibi=False,
        alibi_bias_max=8,
        **kwargs,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.attn_impl = attn_impl
        self.normalise_query_key = normalise_query_key
        self.clip_query_key_value = clip_query_key_value
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max


class MptIntializerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    def __init__(
        self,
        name="kaiming_normal_",
        fan_mode="fan_in",
        init_nonlinearity="relu",
        init_div_is_residual=True,
        emb_init_std=None,
        emb_init_uniform_lim=None,
        init_std=None,
        init_gain=0.0,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.fan_mode = fan_mode
        self.init_nonlinearity = init_nonlinearity
        self.init_div_is_residual = init_div_is_residual
        self.emb_init_std = emb_init_std
        self.emb_init_uniform_lim = emb_init_uniform_lim
        self.init_std = init_std
        self.init_gain = init_gain


class MPTConfig(PretrainedConfig):
    model_type = "mpt"

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: MptAttentionConfig = None,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        init_config: MptIntializerConfig = None,
        **kwargs,
    ):
        """The MPT configuration class.
        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (int): The ratio of the up/down scale in the MLP.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict):  A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                    extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                    can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            multiquery_attention (bool): Whether to use multiquery attention implementation.
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.models.utils.param_init_fns.py for info on other param init config options
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        if "name" in kwargs:
            del kwargs["name"]
        if "loss_fn" in kwargs:
            del kwargs["loss_fn"]
        super().__init__(**kwargs)
        self._validate_config()

    def _validate_config(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if any((prob < 0 or prob > 1 for prob in [self.attn_config["attn_pdrop"], self.resid_pdrop, self.emb_pdrop])):
            raise ValueError(
                "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1"
            )
        if self.attn_config["attn_impl"] not in ["torch", "flash", "triton"]:
            raise ValueError(f"Unknown attn_impl={self.attn_config['attn_impl']}")
        if self.attn_config["prefix_lm"] and self.attn_config["attn_impl"] not in ["torch", "triton"]:
            raise NotImplementedError("prefix_lm only implemented with torch and triton attention.")
        if self.attn_config["alibi"] and self.attn_config["attn_impl"] not in ["torch", "triton"]:
            raise NotImplementedError("alibi only implemented with torch and triton attention.")
        if self.attn_config["attn_uses_sequence_id"] and self.attn_config["attn_impl"] not in ["torch", "triton"]:
            raise NotImplementedError("attn_uses_sequence_id only implemented with torch and triton attention.")
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError("model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!")
        if isinstance(self.logit_scale, str) and self.logit_scale != "inv_sqrt_d_model":
            raise ValueError(
                f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
            )
        if self.init_config.get("name", None) is None:
            raise ValueError(f"self.init_config={self.init_config!r} 'name' needs to be set.")
        if not self.learned_pos_emb and (not self.attn_config["alibi"]):
            raise ValueError(
                "Positional information must be provided to the model using either learned_pos_emb or alibi."
            )
