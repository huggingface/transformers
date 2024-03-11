from ...modeling_utils import PretrainedConfig
from .modeling_granite.enums import AttentionHeadType, PositionEmbeddingType


class GraniteConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    model_type = "gpt_megatron"

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
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        attention_softmax_in_fp32: bool = True,
        scale_attention_softmax_in_fp32: bool = True,
        apply_residual_connection_post_layernorm: bool = False,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict = None,
        **kwargs,
    ):
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
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.position_embedding_type = position_embedding_type
        self.add_bias = add_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # just to test if attention head type and position embedding type is supported
        attention_head_type = AttentionHeadType(attention_head_type)
        position_embedding_type = PositionEmbeddingType(position_embedding_type)

        # for compatibility with some features
        self.multi_query = attention_head_type == AttentionHeadType.mqa

        if attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.n_head

            assert (
                self.n_head == self.num_key_value_heads
            ), "MultiHeadAttention should have same number of heads for query, keys and values"
        elif attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, "MultiQueryAttention should have 1 head for keys and values"
        elif attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert (
                self.n_head % self.num_key_value_heads == 0
            ), "GroupedQueryAttention should have more than 1 head for keys and values"

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
