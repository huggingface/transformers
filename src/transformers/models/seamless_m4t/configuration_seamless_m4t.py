# coding=utf-8
# Copyright 2022 ylacombe and The HuggingFace Inc. team. All rights reserved.
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
""" SeamlessM4T model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "meta-private/m4t_large": "https://huggingface.co/meta-private/m4t_large/resolve/main/config.json",
    # See all SeamlessM4T models at https://huggingface.co/models?filter=seamless_m4t
}


class SeamlessM4TConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SeamlessM4TModel`].
    It is used to instantiate an SeamlessM4T model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the SeamlessM4T [meta-private/m4t_large](https://huggingface.co/meta-private/m4t_large) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the SeamlessM4T model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~SeamlessM4TModel`] or
            [`~TFSeamlessM4TModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~SeamlessM4TModel`] or
            [`~TFSeamlessM4TModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # Initializing a SeamlessM4T meta-private/m4t_large style configuration
    >>> configuration = SeamlessM4TConfig()

    >>> # Initializing a model from the meta-private/m4t_large style configuration
    >>> model = SeamlessM4TModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "seamless_m4t"

    def __init__(
        self,
        vocab_size=30522,
        # overall_config
        hidden_size=1024,
        use_text_encoder=True,
        use_conformer_adaptor=True,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
        # t2u config
        unit_vocabulary_size=10082,
        unit_pad_idx=1,
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
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


###################


class NllbMoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NllbMoeModel`]. It is used to instantiate an
    NLLB-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the NLLB-MoE
    [facebook/nllb-moe-54b](https://huggingface.co/facebook/nllb-moe-54b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the NllbMoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`NllbMoeModel`] or
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        second_expert_policy ( `str`, *optional*, default to `"all"`):
            The policy used for the sampling the probability of being sampled to a second expert for each token.
        normalize_router_prob_before_dropping (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the router probabilities before applying a mask based on the experts capacity
            (capacity dropping).
        batch_prioritized_routing (`bool`, *optional*, defaults to `True`):
            Whether or not to orders the tokens by their router probabilities before capacity dropping. This means that
            the tokens that have the highest probabilities will be routed before other tokens that might be further in
            the sequence.
        moe_eval_capacity_token_fraction (`float`, *optional*, defaults to 1.0):
            Fraction of tokens as capacity during validation, if set to negative, uses the same as training. Should be
            in range: (0.0, 1.0].
        num_experts (`int`, *optional*, defaults to 128):
            Number of experts for each NllbMoeSparseMlp layer.
        expert_capacity (`int`, *optional*, defaults to 64):
            Number of tokens that can be stored in each expert.
        encoder_sparse_step (`int`, *optional*, defaults to 4):
            Frequency of the sparse layers in the encoder. 4 means that one out of 4 layers will be sparse.
        decoder_sparse_step (`int`, *optional*, defaults to 4):
            Frequency of the sparse layers in the decoder. 4 means that one out of 4 layers will be sparse.
        router_dtype (`str`, *optional*, default to `"float32"`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
            *selective precision* discussion in [the paper](https://arxiv.org/abs/2101.03961).
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing. if `False`, the padding tokens are not routed to any
            experts.
        router_bias (`bool`, *optional*, defaults to `False`):
            Whether or not the classifier of the router should have a bias.
        moe_token_dropout (`float`, *optional*, defualt ot 0.2):
            Masking rate for MoE expert output masking (EOM), which is implemented via a Dropout2d on the expert
            outputs.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the router logits. Only set to `True` to get the auxiliary loss when training.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import NllbMoeModel, NllbMoeConfig

    >>> # Initializing a NllbMoe facebook/nllb-moe-54b style configuration
    >>> configuration = NllbMoeConfig()

    >>> # Initializing a model from the facebook/nllb-moe-54b style configuration
    >>> model = NllbMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "nllb-moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=128112,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.05,
        decoder_layerdrop=0.05,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        scale_embedding=True,
        router_bias=False,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        num_experts=128,
        expert_capacity=64,
        encoder_sparse_step=4,
        decoder_sparse_step=4,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        second_expert_policy="all",
        normalize_router_prob_before_dropping=False,
        batch_prioritized_routing=False,
        moe_eval_capacity_token_fraction=1.0,
        moe_token_dropout=0.2,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.decoder_sparse_step = decoder_sparse_step
        self.encoder_sparse_step = encoder_sparse_step
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype

        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.batch_prioritized_routing = batch_prioritized_routing
        self.second_expert_policy = second_expert_policy
        self.normalize_router_prob_before_dropping = normalize_router_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.moe_token_dropout = moe_token_dropout
        self.output_router_logits = output_router_logits
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
