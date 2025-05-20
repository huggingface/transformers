from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class Dots1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Dots1Model`]. It is used to instantiate a
    `dots.llm1` model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    [rednote-hilab/dots.llm1.base](https://huggingface.co/rednote-hilab/dots.llm1.base).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`Dots1Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, default=None):
            Number of shared experts. None means dense model.
        n_routed_experts (`int`, *optional*, default=None):
            Number of routed experts. None means dense model.
        num_experts_per_tok (`int`, *optional*, default=None):
            Number of selected experts. None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers at the beginning of the model before the first MoE layer.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method for computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux (`bool`, *optional*, defaults to True):
            Whether to compute auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, Multi
            Head Attention (MHA) is used. If `num_key_value_heads=1`, Multi Query Attention (MQA) is used. Otherwise,
            Grouped Query Attention (GQA) is used. If not specified, defaults to `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to "silu"):
            The non-linear activation function (function or string).
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length the model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to True):
            Whether or not the model should return the last key/values attentions. Only relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to None):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 151643):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental: tensor parallelism rank used during pretraining. This is necessary for exact reproducibility
            of pretraining results.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether to tie the input and output word embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary for scaling RoPE embeddings. Supports `{"type": strategy name, "factor": scaling factor}`.
        attention_bias (`bool`, *optional*, defaults to False):
            Whether to use a bias in the self-attention projections.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token (selected experts only within `topk_group` groups).
        use_sliding_window (`bool`, *optional*, defaults to False):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window for attention. If not specified, defaults to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.

    Examples:
        ```python
        >>> from transformers import Dots1Model, Dots1Config

        >>> # Initializing a Dots1 style configuration
        >>> configuration = Dots1Config()

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "dots1"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {  #  # TODO: only replicate attention layers when > first_k_dense_replace
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "local_colwise",
        "layers.*.mlp.experts.*.up_proj": "local_colwise",
        "layers.*.mlp.experts.*.down_proj": "local_rowwise",
        "layers.*.mlp.experts.*": "local",  # each expert is wrapped in a module list
        "layers.*.mlp.shared_experts.gate_proj": "local_colwise",
        "layers.*.mlp.shared_experts.up_proj": "local_colwise",
        "layers.*.mlp.shared_experts.down_proj": "local_rowwise",
        "layers.*.mlp.shared_experts": "local",
        "layers.*.mlp.gate_proj": "local_colwise",
        "layers.*.mlp.up_proj": "local_colwise",
        "layers.*.mlp.down_proj": "local_rowwise",
        "layers.*.mlp": "gather",  # This is the only moment where results are gathered
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=4608,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=62,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=151643,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        routed_scaling_factor=1.0,
        use_sliding_window=False,
        sliding_window=4096,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.n_group = n_group
        self.topk_group = topk_group
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
        self.routed_scaling_factor = routed_scaling_factor
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Dots1Config"]
