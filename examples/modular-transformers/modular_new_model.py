# Example where we only want to overwrite the defaults of an init

from transformers.models.gemma.configuration_gemma import GemmaConfig


class NewModelConfig(GemmaConfig):

    vocab_size: int = 256030
    hidden_size: int = 64
    intermediate_size: int = 90
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str | None = None
    max_position_embeddings: int = 1500
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_bidirectional_attention: bool = False

    @property
    def num_heads(self):
        return self.num_attention_heads
