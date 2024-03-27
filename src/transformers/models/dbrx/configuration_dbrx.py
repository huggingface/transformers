"""Dbrx configuration."""
from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DBRX_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DbrxAttentionConfig(PretrainedConfig):
    """Configuration class for Dbrx Attention.

    [`DbrxAttention`] class. It is used to instantiate attention layers
    according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        clip_qkv (`float`, *optional*, defualts to None):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
        rope_theta (float): The base frequency for rope.
    """

    def __init__(
        self,
        attn_pdrop: float = 0,
        clip_qkv: Optional[float] = None,
        kv_n_heads: int = 1,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta

        for k in ['model_type']:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f'Found unknown {kwargs=}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                        **kwargs: Any) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
                                                  **kwargs)

        if config_dict.get('model_type') == 'dbrx':
            config_dict = config_dict['attn_config']

        if 'model_type' in config_dict and hasattr(
                cls,
                'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                +
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)


class DbrxFFNConfig(PretrainedConfig):
    """Configuration class for Dbrx FFN.

    [`DbrxFFN`] class. It is used to instantiate feedforward layers according to
    the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        ffn_act_fn (dict, optional): A dict specifying activation function for the FFN.
            The dict should have a key 'name' with the value being the name of
            the activation function along with any additional keyword arguments.
        ffn_hidden_size (int, optional): The hidden size of the feedforward network.
        moe_num_experts (int, optional): The number of experts in the mixture of experts layer.
        moe_top_k (int, optional): The number of experts to use in the mixture of experts layer.
        moe_jitter_eps (float, optional): The jitter epsilon for the mixture of experts layer.
        moe_loss_weight (float, optional): The loss weight for the mixture of experts layer.
        moe_normalize_expert_weights (float, optional): The normalization factor for the expert weights.
        uniform_expert_assignment (bool, optional): Whether to use uniform expert assignment.
            This should only be used for benchmarking purposes.
    """

    def __init__(
        self,
        ffn_act_fn: Optional[dict] = None,
        ffn_hidden_size: int = 3584,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_jitter_eps: Optional[float] = None,
        moe_loss_weight: float = 0.01,
        moe_normalize_expert_weights: Optional[float] = 1,
        uniform_expert_assignment: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        if ffn_act_fn is None:
            ffn_act_fn = {'name': 'silu'}
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        for k in ['model_type']:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f'Found unknown {kwargs=}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                        **kwargs: Any) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
                                                  **kwargs)

        if config_dict.get('model_type') == 'dbrx':
            config_dict = config_dict['ffn_config']

        if 'model_type' in config_dict and hasattr(
                cls,
                'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                +
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)


class DbrxConfig(PretrainedConfig):
    """Configuration class for Dbrx.

    [`DbrxModel`]. It is used to instantiate a Dbrx model according to the
    specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 6144):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 32768):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DbrxModel`].
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        ffn_config (`dict`, *optional*):
            A dictionary used to configure the model's FFN module.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.


    Example:
    ```python
    >>> from transformers import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = 'dbrx'
    attribute_map = {
        'num_attention_heads': 'n_heads',
        'hidden_size': 'd_model',
        'num_hidden_layers': 'n_layers',
        'max_position_embeddings': 'max_seq_len'
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        max_seq_len: int = 2048,
        vocab_size: int = 32000,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_config: Optional[DbrxAttentionConfig] = None,
        ffn_config: Optional[DbrxFFNConfig] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.05,
        **kwargs: Any,
    ):
        if attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        tie_word_embeddings = kwargs.pop('tie_word_embeddings', False)
        if tie_word_embeddings:
            raise ValueError(
                'tie_word_embeddings is not supported for Dbrx models.')

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )