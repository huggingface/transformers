from transformers.models.llama.modeling_llama import LlamaConfig
# Example where we only want to overwrite the defaults of an init?
class GemmaConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation=None,
        max_position_embeddings=8192,
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
    ):
        super().__init__(self)