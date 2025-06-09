from transformers import PretrainedConfig

class FirefliesConfig(PretrainedConfig):
    model_type = "fireflies"

    def __init__(self, vocab_size=50257, d_model=1024, n_heads=19, num_layers=13, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers