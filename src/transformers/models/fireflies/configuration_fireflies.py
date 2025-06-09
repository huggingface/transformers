from transformers.configuration_utils import PretrainedConfig

class FirefliesConfig(PretrainedConfig):
    model_type = "fireflies"

    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_heads=16,
        num_layers=24,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
