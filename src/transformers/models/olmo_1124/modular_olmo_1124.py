from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import logging
from ..llama.modeling_llama import LlamaRMSNorm
from ..olmo.configuration_olmo import OlmoConfig


logger = logging.get_logger(__name__)


class Olmo1124Config(OlmoConfig):
    r"""
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon used by the rms normalization layers.
    """

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.rms_norm_eps = rms_norm_eps
        del self.clip_qkv


class Olmo1124RMSNorm(LlamaRMSNorm):
    pass


ALL_LAYERNORM_LAYERS.append(Olmo1124RMSNorm)
