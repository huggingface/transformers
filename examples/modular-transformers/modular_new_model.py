# Example where we only want to overwrite the defaults of an init

from transformers.models.gemma.configuration_gemma import GemmaConfig


class NewModelConfig(GemmaConfig):
    def __init__(
        self,
        vocab_size=256030,
        hidden_size=64,
        intermediate_size=90,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation=None,
        max_position_embeddings=1500,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(self, **kwargs)

    @property
    def num_heads(self):
        return self.num_attention_heads
