from ...configuration_utils import PretrainedConfig

class Yuan2M32Config(PretrainedConfig):
    model_type = "yuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=135040,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        model_max_length=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=77185,
        bos_token_id=77185,
        eos_token_id=77185,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

