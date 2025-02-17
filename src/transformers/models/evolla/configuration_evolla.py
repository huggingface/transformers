# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Evolla model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class EvollaSequenceCompressorConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 6):
            Depth of the transformer layers.
        heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each transformer layer.
        num_latents (`int`, *optional*, defaults to 64):
            Number of learned latent vectors.
        ff_mult (`int`, *optional*, defaults to 4):
            Multiplier for the hidden dimension of the feed-forward layers.
    """

    def __init__(self, depth: int = 6, heads: int = 8, num_latents=64, ff_mult=4, **kwargs):
        self.depth = depth
        self.heads = heads
        self.num_latents = num_latents
        self.ff_mult = ff_mult
        super().__init__(**kwargs)


class EvollaProteinEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 446):
            Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ESMModel`].
        mask_token_id (`int`, *optional*, defaults to 4):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*, defaults to 1):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 33):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `True`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
    """

    def __init__(
        self,
        vocab_size=446,
        mask_token_id=4,
        pad_token_id=1,
        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        position_embedding_type="rotary",
        use_cache=True,
        emb_layer_norm_before=False,
        token_dropout=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout

        super().__init__(mask_token_id=mask_token_id, pad_token_id=pad_token_id, **kwargs)


class EvollaProteinConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        protein_encoder_config (`EvollaProteinEncoderConfig`, *optional*):
            The configuration of the protein encoder.
        resampler_config (`EvollaSequenceCompressorConfig`, *optional*):
            The configuration of the resampler.
        protein_text_hidden_size (`int`, *optional*, defaults to 4096):
            The hidden size of the protein text representation.
    """

    sub_configs = {
        "protein_encoder_config": EvollaProteinEncoderConfig,
        "resampler_config": EvollaSequenceCompressorConfig,
    }

    def __init__(
        self,
        initializer_range=0.02,
        protein_encoder_config=None,
        resampler_config=None,
        protein_text_hidden_size=4096,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        if protein_encoder_config is None:
            self.protein_encoder_config = EvollaProteinEncoderConfig()
        elif isinstance(protein_encoder_config, dict):
            self.protein_encoder_config = EvollaProteinEncoderConfig(**protein_encoder_config)
        elif isinstance(protein_encoder_config, EvollaProteinEncoderConfig):
            self.protein_encoder_config = protein_encoder_config

        if resampler_config is None:
            self.resampler_config = EvollaSequenceCompressorConfig()
        elif isinstance(resampler_config, dict):
            self.resampler_config = EvollaSequenceCompressorConfig(**resampler_config)
        elif isinstance(resampler_config, EvollaSequenceCompressorConfig):
            self.resampler_config = resampler_config

        self.output_repr_dim = protein_text_hidden_size

        super().__init__(**kwargs)


class EvollaLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 128000):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 128009):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
    """

    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128009,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class EvollaSequenceAlignerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        ffn_mult (`int`, *optional*, defaults to 4): The factor to multiply the hidden size of the llama model by to get the
            hidden size of the feedforward layers in the sequence aligner.
        enable_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the feedforward layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1): The dropout ratio for the attention
            probabilities.
        num_add_layers (`int`, *optional*, defaults to 8): The number of additional layers to add to the sequence aligner.
    """

    def __init__(
        self,
        ffn_mult: int = 4,
        enable_bias: bool = True,
        attention_probs_dropout_prob: float = 0.1,
        num_add_layers: int = 8,
        **kwargs,
    ):
        self.ffn_mult = ffn_mult
        self.enable_bias = enable_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_add_layers = num_add_layers
        self.protein_encoder_dim = None
        self.structure_encoder_dim = None
        self.msa_encoder_dim = None
        super().__init__(**kwargs)


class EvollaLLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the part of the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        output_attentions (`bool`, *optional*, defaults to `False`): Whether or not to return the attentions tensors of all
            attention layers. See [`EvollaModel`] for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`): Whether or not to return the hidden states of all
            layers. See [`EvollaModel`] for more details.
        llama_config (`Dict`, *optional*): The configuration for the LLaMA model. If not provided, the default values
            will be used.
        sequence_aligner_config (`Dict`, *optional*): The configuration for the sequence aligner. If not provided, the
            default values will be used.
        protein_text_hidden_size (`int`, *optional*, defaults to 4096): The hidden size of the protein text encoder.
        quantization (`str`, *optional*, defaults to `"8bit"`): The quantization method to use. Can be one of `"8bit"` or
            `"4bit"`.
        initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices.
    """

    sub_configs = {"llama_config": EvollaLlamaConfig, "sequence_aligner_config": EvollaSequenceAlignerConfig}

    def __init__(
        self,
        output_attentions=False,
        output_hidden_states=False,
        llama_config=None,
        sequence_aligner_config=None,
        protein_text_hidden_size=4096,
        quantization="8bit",
        initializer_range=0.02,
        **kwargs,
    ):
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        if llama_config is None:
            self.llama_config = EvollaLlamaConfig()
        elif isinstance(llama_config, dict):
            self.llama_config = EvollaLlamaConfig(**llama_config)
        elif isinstance(llama_config, EvollaLlamaConfig):
            self.llama_config = llama_config

        if sequence_aligner_config is None:
            self.sequence_aligner_config = EvollaSequenceAlignerConfig()
        elif isinstance(sequence_aligner_config, dict):
            self.sequence_aligner_config = EvollaSequenceAlignerConfig(**sequence_aligner_config)
        elif isinstance(sequence_aligner_config, EvollaSequenceAlignerConfig):
            self.sequence_aligner_config = sequence_aligner_config

        self.sequence_aligner_config.protein_encoder_dim = protein_text_hidden_size

        self.quantization = quantization
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class EvollaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256): The llama vocabulary size.
        protein_text_hidden_size (`int`, *optional*, defaults to 4096): The hidden size of the protein text encoder.
        output_attentions (`bool`, *optional*, defaults to `False`): Whether or not to return the attentions tensors of all
            attention layers. See [`EvollaModel`] for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`): Whether or not to return the hidden states of all
            layers. See [`EvollaModel`] for more details.
        use_cache (`bool`, *optional*, defaults to `False`): Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if `config.is_decoder=True`.
        return_dict (`bool`, *optional*, defaults to `True`): Whether or not to return a [`~file_utils.ModelOutput`] instead of a
            plain tuple.
        generation_max_new_tokens (`int`, *optional*, defaults to 512): The maximum number of tokens to generate.
        generation_do_sample (`bool`, *optional*, defaults to `True`): Whether or not to use sampling ; use greedy
            decoding otherwise.
        generation_temperature (`float`, *optional*, defaults to 0.6): The value used to module the next token
            probabilities.
        generation_top_p (`float`, *optional*, defaults to 0.9): The cumulative probability for top-p sampling.
        generation_config (`Dict`, *optional*): The configuration for the generation parameters. If not provided, the
            default values will be used.
        protein_config (`Dict`, *optional*): The configuration for the protein text encoder. If not provided, the
            default values will be used.
        llm_config (`Dict`, *optional*): The configuration for the LLaMA model. If not provided, the default values
            will be used.
        initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import EvollaModel, EvollaConfig

    >>> # Initializing a Evolla evolla-10b style configuration
    >>> configuration = EvollaConfig()

    >>> # Initializing a model from the evolla-10b style configuration
    >>> model = EvollaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "EvollaModel"
    # sub_configs = {"protein_config": EvollaProteinConfig, "llm_config": EvollaLLMConfig}

    def __init__(
        self,
        vocab_size=128256,  # llama vocab size
        hidden_size=4096,  # llama hidden size
        intermediate_size=14336,  # llama intermediate size
        num_hidden_layers=32,  # llama num layers
        num_attention_heads=32,  # llama num heads
        num_key_value_heads=8,  # llama num key-value heads
        hidden_act="silu",  # llama activation function
        max_position_embeddings=8192,  # llama rope max length
        rms_norm_eps=1e-05,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128009,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        # head_dim=None,
        aligner_ffn_mult=4,
        aligner_enable_bias=True,
        aligner_attention_probs_dropout_prob=0.1,
        aligner_num_add_layers=8,
        protein_vocab_size=446,
        protein_mask_token_id=4,
        protein_pad_token_id=1,
        protein_hidden_size=1280,
        protein_num_hidden_layers=33,
        protein_num_attention_heads=20,
        protein_intermediate_size=5120,
        protein_hidden_dropout_prob=0.1,
        protein_attention_probs_dropout_prob=0.1,
        protein_max_position_embeddings=1026,
        protein_layer_norm_eps=1e-05,
        protein_position_embedding_type="rotary",
        protein_emb_layer_norm_before=False,
        protein_token_dropout=True,
        resampler_depth=6,
        resampler_dim_head=64,
        resampler_heads=8,
        resampler_num_latents=64,
        resampler_ff_mult=4,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        # protein_config=None,
        # llm_config=None,
        initializer_range=0.02,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # self.output_attentions = output_attentions
        # self.output_hidden_states = output_hidden_states
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        # self.pad_token_id = pad_token_id
        # self.bos_token_id = bos_token_id
        # self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.aligner_ffn_mult = aligner_ffn_mult
        self.aligner_enable_bias = aligner_enable_bias
        self.aligner_attention_probs_dropout_prob = aligner_attention_probs_dropout_prob
        self.aligner_num_add_layers = aligner_num_add_layers

        self.protein_vocab_size = protein_vocab_size
        self.protein_mask_token_id = protein_mask_token_id
        self.protein_pad_token_id = protein_pad_token_id
        self.protein_hidden_size = protein_hidden_size
        self.protein_num_hidden_layers = protein_num_hidden_layers
        self.protein_num_attention_heads = protein_num_attention_heads
        self.protein_intermediate_size = protein_intermediate_size
        self.protein_hidden_dropout_prob = protein_hidden_dropout_prob
        self.protein_attention_probs_dropout_prob = protein_attention_probs_dropout_prob
        self.protein_max_position_embeddings = protein_max_position_embeddings
        self.protein_layer_norm_eps = protein_layer_norm_eps
        self.protein_position_embedding_type = protein_position_embedding_type
        self.protein_emb_layer_norm_before = protein_emb_layer_norm_before
        self.protein_token_dropout = protein_token_dropout

        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head
        self.resampler_heads = resampler_heads
        self.resampler_num_latents = resampler_num_latents
        self.resampler_ff_mult = resampler_ff_mult

        # self.max_new_tokens = max_new_tokens
        # self.do_sample = do_sample
        # self.temperature = temperature
        # self.top_p = top_p
        # print(return_dict, self.return_dict)

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # if protein_config is None:
        #     self.protein_config = EvollaProteinConfig(protein_text_hidden_size=self.protein_text_hidden_size)
        # elif isinstance(protein_config, dict):
        #     protein_config.update({"protein_text_hidden_size": self.protein_text_hidden_size})
        #     self.protein_config = EvollaProteinConfig(**protein_config)
        # elif isinstance(protein_config, EvollaProteinConfig):
        #     self.protein_config = protein_config

        # if llm_config is None:
        #     self.llm_config = EvollaLLMConfig(protein_text_hidden_size=self.protein_text_hidden_size)
        # elif isinstance(llm_config, dict):
        #     llm_config.update({"protein_text_hidden_size": self.protein_text_hidden_size})
        #     self.llm_config = EvollaLLMConfig(**llm_config)
        # elif isinstance(llm_config, EvollaLLMConfig):
        #     self.llm_config = llm_config

        # if self.protein_config.resampler_config.output_repr_dim is None:
        #     self.protein_config.resampler_config.output_repr_dim = self.llm_config.llama_config.hidden_size

        # if self.llm_config.sequence_aligner_config.protein_encoder_dim is None:
        #     self.llm_config.sequence_aligner_config.protein_encoder_dim = self.protein_config.resampler_config.output_repr_dim

        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PretrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.


__all__ = ["EvollaConfig"]
