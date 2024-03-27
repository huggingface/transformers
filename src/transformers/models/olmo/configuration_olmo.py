# coding=utf-8
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" OLMo model configuration"""

from enum import Enum

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

OLMO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/OLMo": "https://huggingface.co/allenai/OLMo/resolve/main/config.json",
}


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


class OLMoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OLMoModel`]. It is used to instantiate an OLMo
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OLMo-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the OLMo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OLMoModel`]
        d_model (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        mlp_ratio (`int`, *optional*, defaults to 4):
            The ratio of the inner MLP dimensionality to ``d_model``.
            This is only used when ``mlp_hidden_size`` is not set.
        mlp_hidden_size (`Optional[int]`, *optional*, defaults to 22016):
            Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
        n_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        n_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        multi_query_attention (`bool`, *optional*, defaults to `False`):
            If `True`, the model will use Multi Query Attention (MQA). Otherwise the model will use Multi Head Attention (MHA).
        activation_type (`ActivationType`, *optional*, defaults to `swiglu`):
            The non-linear activation function in the decoder.
        max_sequence_length (`int`, *optional*, defaults to 2048):
            The maximum input sequence length supported by the model.
        block_type (`BlockType`, *optional*, defaults to `sequential`):
            The transformer block implementation.
        block_group_size (`int`, *optional*, defaults to 1):
            The number of blocks to group together into a single parent block.
            This has no affect on the number of parameters in the model and is only used to wrap groups
            of blocks together with a single FSDP wrapper during training.
        alibi (`bool`, *optional*, defaults to `False`):
            If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.
        alibi_bias_max (`float`, *optional*, defaults to 8.0):
            Maximum absolute value of ALiBi bias.
        rope (`bool`, *optional*, defaults to `True`):
            Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
        rope_full_precision (`bool`, *optional*, defaults to `False`):
            If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
            apply RoPE at the precision of the input.
        flash_attention (`bool`, *optional*, defaults to `True`):
            If ``True``, use ``FlashAttention``. This is ignored if ``use_pytorch_sdpa`` is ``False``.
        use_pytorch_sdpa (`bool`, *optional*, defaults to `True`):
            If ``True``, use Pytorch's ``F.scaled_dot_product_attention`` instead of a manual
            implementation for the same functionality.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability within the attention modules.
        attention_layer_norm (`bool`, *optional*, defaults to `False`):
            Apply layer norm to the keys and queries within the attention mechanism.
            This can help stabilize training.
        residual_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the MLP and attention output within each block.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for embeddings.
        layer_norm_type (`LayerNormType`, *optional*, defaults to `default`):
            The layernorm implementation to use.
        layer_norm_with_affine (`bool`, *optional*, defaults to `False`):
            Whether to include bias and weight parameters for the layer norms.
            This only affects layer norms that are immediately followed by a linear layer in the forward pass,
            so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
            to ``False``.
        attention_layer_norm_with_affine (`bool`, *optional*, defaults to `False`):
            Toggle affine transform for the QK norms.
        include_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to include bias parameters in linear layers.
            In PaLM, they got rid of all bias terms because they found that large
            models tend to have near 0 bias terms anyway.
        bias_for_layer_norm (`Optional[bool]`, *optional*, defaults to `False`):
            Whether or not to include bias parameters in layer norm.
            This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
            layer norm.
            When this is None, it inherits the setting from include_bias.
        scale_logits (`bool`, *optional*, defaults to `False`):
            If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
        weight_tying (`bool`, *optional*, defaults to `False`):
            Whether to tie output linear weights to the input embedding.
        init_device (`Optional[str]`, *optional*, defaults to `"cpu"`):
            The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
            See also `change_meta_init_to_cpu`.
        change_meta_init_to_cpu (`bool`, *optional*, defaults to `True`):
            If `change_meta_init_to_cpu` is `True` and `init_device` is set to "meta", then
            `init_device` is instead treated as "cpu". This changes the default init device of legacy
            OLMo checkpoints to "cpu".
        init_fn (`InitFnType`, *optional*, defaults to `mitchell`):
            The weight initialization strategy.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
            as "normal".
        init_cutoff_factor (`Optional[float]`, *optional*):
            A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
            as "normal". Setting this to None means values are not cutoff.
            Clip QKV to this value when set.
        clip_qkv (`Optional[float]`, *optional*):
            Clip QKV to this value when set.
        pad_token_id (`int`, *optional*, defaults to 1):
            The ID of the token to use for padding.
        eos_token_id (`int`, *optional*, defaults to 50279):
            The ID of the end-of-sentence special token.
        embedding_size (`Optional[int]`, *optional*):
            *Deprecated* Use `vocab_size` instead. This is kept to support legacy checkpoints.

    ```python
    >>> from transformers import OLMoModel, OLMoConfig

    >>> # Initializing a OLMo 7B style configuration
    >>> configuration = OLMoConfig()

    >>> # Initializing a model from the OLMo 7B style configuration
    >>> model = OLMoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "olmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "tie_word_embeddings": "weight_tying",
    }

    def __init__(
        self,
        vocab_size=50304,
        d_model=4096,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
        mlp_ratio=4,
        mlp_hidden_size: int | None = 22016,
        n_layers=32,
        n_heads=32,
        multi_query_attention=False,
        activation_type=ActivationType.swiglu,
        max_sequence_length=2048,
        block_type=BlockType.sequential,
        block_group_size=1,
        alibi=False,
        alibi_bias_max=8.0,
        rope=True,
        rope_full_precision=False,
        flash_attention=True,
        use_pytorch_sdpa=True,
        attention_dropout=0.0,
        attention_layer_norm=False,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        layer_norm_type=LayerNormType.default,
        layer_norm_with_affine=False,
        attention_layer_norm_with_affine=False,
        include_bias=False,
        bias_for_layer_norm=False,
        scale_logits=False,
        weight_tying=False,
        init_device="cpu",
        change_meta_init_to_cpu=True,
        init_fn=InitFnType.mitchell,
        init_std=0.02,
        init_cutoff_factor=None,
        clip_qkv=None,
        pad_token_id=1,
        eos_token_id=50279,
        embedding_size=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        if change_meta_init_to_cpu and init_device.lower() == "meta":
            init_device = "cpu"

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.use_cache = use_cache
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_size: int | None = mlp_hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.multi_query_attention = multi_query_attention
        self.activation_type = activation_type
        self.block_type = block_type
        self.block_group_size = block_group_size
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
        self.rope = rope
        self.rope_full_precision = rope_full_precision
        self.flash_attention = flash_attention
        self.use_pytorch_sdpa = use_pytorch_sdpa
        self.attention_dropout = attention_dropout
        self.attention_layer_norm = attention_layer_norm
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.layer_norm_type = layer_norm_type
        self.layer_norm_with_affine = layer_norm_with_affine
        self.attention_layer_norm_with_affine = attention_layer_norm_with_affine
        self.include_bias = include_bias
        self.bias_for_layer_norm = bias_for_layer_norm
        self.scale_logits = scale_logits
        self.weight_tying = weight_tying
        self.init_device = init_device
        self.init_fn = init_fn
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.clip_qkv = clip_qkv
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # OLMo considers 'vocab size' (the number of different tokens the tokenizer can produce)
        # and 'embedding size' (the number of embeddings) to be different and separately configurable.
        # HF does not. We set `embedding_size` to `None` so that HF OLMo models always have the
        # same vocab and embedding sizes, thus avoiding this problem.
        self.embedding_size = None
        if embedding_size is not None:
            self.vocab_size = embedding_size

        # This currently exists just to make a HF test pass
        self.init_std_mitchell = 1.0
