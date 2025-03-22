# configuration_arlow.py

from ...configuration_utils import PretrainedConfig


class ArlowConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`ArlowModel`], intended for multi-turn
    text-to-text causal language modeling. It is used to instantiate an ArlowGPT model according to the specified
    arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the ArlowGPT model. Defines the number of different tokens that can be represented by
            `input_ids`.
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimensionality of the MLP (feed-forward) layers.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length (in tokens) that this model might ever be used with.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer.
        num_key_value_heads (`int`, *optional*, defaults to 12):
            Number of key-value heads for attention (e.g., for Grouped Query Attention). If `num_key_value_heads`
            equals `num_attention_heads`, it behaves like standard multi-head attention.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer decoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (useful for speedy decoding in
            multi-turn or streaming scenarios).
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by RMS normalization layers.
        rope_theta (`float`, *optional*, defaults to 100000.0):
            The base period used by RoPE (rotary position embeddings).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input word embeddings and output embeddings.
        pad_token_id (`int`, *optional*):
            The padding token id. Must be set if padding is used, e.g., during batched training or generation.
        cross_attention (`bool`, *optional*, defaults to `True`):
            Whether cross-attention layers are included in the model. If `True`, the model can attend to encoder
            states in a text-to-text setup or a multi-turn conversational scenario where an external context is
            provided.
        use_cross_attention (`bool`, *optional*, defaults to `True`):
            Higher-level toggle for enabling cross-attention. If set to `False`, cross-attention will be skipped
            even if the layers are present.
        bos_token_id (`int`, *optional*, defaults to 1):
            Token id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            Token id of the end-of-sequence token.
        **kwargs:
            Additional key-value arguments passed to the base class `PretrainedConfig`.

    Example:
    ```python
    >>> from transformers import AutoTokenizer
    >>> from your_package.configuration_arlow import ArlowConfig

    >>> # Assume you've created a tokenizer with 131072 tokens
    >>> tokenizer = AutoTokenizer.from_pretrained("path/to/your-tokenizer")

    >>> config = ArlowConfig(
    ...     vocab_size=len(tokenizer),
    ...     hidden_size=2304,
    ...     intermediate_size=9216,
    ...     max_position_embeddings=2048,
    ...     num_attention_heads=12,
    ...     num_key_value_heads=12,
    ...     num_hidden_layers=28,
    ...     attention_dropout=0.1,
    ...     initializer_range=0.02,
    ...     hidden_act="silu",
    ...     use_cache=True,
    ...     rms_norm_eps=1e-6,
    ...     rope_theta=100000.0,
    ...     tie_word_embeddings=True,
    ...     pad_token_id=tokenizer.pad_token_id,
    ...     cross_attention=True,
    ...     use_cross_attention=True,
    ...     bos_token_id=1,
    ...     eos_token_id=2
    ... )
    >>> # You can now pass this config to your ArlowModel (multi-turn text-to-text for causal language modeling).
    ```

    """

    model_type = "arlow"

    # By default, frameworks that generate or cache key-value states
    # may store them under "past_key_values".
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2304,
        intermediate_size=9216,
        max_position_embeddings=2048,
        num_attention_heads=12,
        num_key_value_heads=12,
        num_hidden_layers=28,
        attention_dropout=0.1,
        initializer_range=0.02,
        hidden_act="silu",
        use_cache=True,
        rms_norm_eps=1e-6,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        pad_token_id=None,
        cross_attention=True,
        use_cross_attention=True,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.cross_attention = cross_attention
        self.use_cross_attention = use_cross_attention


__all__ = ["ArlowConfig"]
