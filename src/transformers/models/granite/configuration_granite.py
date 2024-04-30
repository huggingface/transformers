""" Granite model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class GraniteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraniteModel`]. It is used to instantiate an Granite
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Granite-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the Granite model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GraniteModel`]
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        n_embd (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to None):
            Number of key and value heads for each attention layer in the Transformer decoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder.
        attention_head_type (`str`, *optional*, defaults to `"mqa"`):
            The attention head mechanism used by the decoder, can be "mha", "mqa" or "gqa". To be used with `num_key_value_heads` appropriately.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        normalization_function (`str`, *optional*, defaults to `"layernorm"`):
            The normalization method for the transformer decoder. Can be one of ["layernorm", "rmsnorm"].
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by attention_multiplier.
        attention_multiplier (`float`, *optional*, defaults to None):
            Attention multiplier, `None` will set it to 1 / sqrt(head_dim).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to call the fused softmax in float32.
        scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to scale the attention softmax in float32.
        add_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias for linear layers in the model.
        position_embedding_type (`str`, *optional*, defaults to `"learned_absolute"`):
            The positional encoding method for the transformer model. Can be one of ["learned_absolute", "alibi", "rope"].
        rope_theta (`float`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.

    ```python
    >>> from transformers import GraniteModel, GraniteConfig

    >>> # Initializing a Granite granite-7b style configuration
    >>> configuration = GraniteConfig()

    >>> # Initializing a model from the granite-7b style configuration
    >>> model = GraniteModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        num_key_value_heads: int = None,
        n_inner: int = None,
        activation_function: str = "gelu_pytorch_tanh",
        attention_head_type: str = "mqa",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        attention_multiplier: float = None,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        attention_softmax_in_fp32: bool = True,
        scale_attention_softmax_in_fp32: bool = True,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_key_value_heads = num_key_value_heads
        self.n_inner = 4 * n_embd if n_inner is None else n_inner
        self.activation_function = activation_function
        self.attention_head_type = attention_head_type
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.normalization_function = normalization_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.attention_multiplier = attention_multiplier
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.position_embedding_type = position_embedding_type
        self.add_bias = add_bias
        self.rope_theta = rope_theta

        if self.attention_multiplier is not None:
            assert self.scale_attn_weights

        # for compatibility with some features
        self.multi_query = attention_head_type == "mqa"

        if attention_head_type == "mha":
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.n_head

            assert (
                self.n_head == self.num_key_value_heads
            ), "MultiHeadAttention should have same number of heads for query, keys and values"
        elif attention_head_type == "mqa":
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, "MultiQueryAttention should have 1 head for keys and values"
        elif attention_head_type == "gqa":
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert (
                self.n_head % self.num_key_value_heads == 0
            ), "GroupedQueryAttention should have more than 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({attention_head_type})")

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
