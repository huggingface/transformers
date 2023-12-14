import math
from ...configuration_utils import PretrainedConfig
from ...activations import ACT2FN


class SigmaMoEConfiguration(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        resid_pdrop: float,
        embd_pdrop: float,
        attention_dropout: float,
        activation: str,
        max_position_embeddings: float,
        initializer_range: float,
        layer_norm_eps: float,
        use_cache: bool,
        tie_word_embeddings: bool,
        rope_theta: float,
        rope_scaling: float,
        partial_rotary_factor: float,
        qk_layernorm: bool,
        n_experts: int,
        expert_size: int,
        top_k_experts: int,
        moe_dropout: float,
        selection_mode: str,
        activation_after_topk: bool,
        moe_bias: bool,
        v_dim: int,
        sinkhorn_n_iters: int,
        expert_dropout: float,
        weight_std_scale: float,
        routing_regularization: float = 0.001,
        num_sparse_hidden_layers: int = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.activation = ACT2FN[activation]
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm

        # MoE related
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.top_k_experts = top_k_experts
        self.moe_dropout = moe_dropout
        self.selection_mode = selection_mode
        self.activation_after_topk = activation_after_topk
        self.moe_bias = moe_bias
        self.v_dim = v_dim
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout
        self.weight_std_scale = weight_std_scale
        self.routing_regularization = routing_regularization
        self.num_sparse_hidden_layers = (
            self.num_hidden_layers
            if num_sparse_hidden_layers is None
            else num_sparse_hidden_layers
        )
        if self.num_sparse_hidden_layers > 0:
            self.sparse_step = math.ceil(
                self.num_hidden_layers / self.num_sparse_hidden_layers
            )
        else:
            # this will create no sparse layers
            self.sparse_step = -1

        self._rope_scaling_validation()

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )
