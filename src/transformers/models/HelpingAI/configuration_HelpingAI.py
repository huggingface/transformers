from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HelpingAIConfig(PretrainedConfig):
    model_type = "HelpingAI"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50281,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        head_dim=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        intermediate_size=6912,
        hidden_act="silu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_eps=1e-5,
        use_cache=False,
        bos_token_id=50278,
        eos_token_id=50279,
        pad_token_id=50279,
        tie_word_embeddings=False,
        rope_pct=0.25,
        rope_theta=10000,
        partial_rotary_factor=0.25,
        use_qkv_bias=False,
        output_router_logits=False,
        router_aux_loss_coef=0.02,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_pct = rope_pct
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.use_qkv_bias = use_qkv_bias
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
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
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")