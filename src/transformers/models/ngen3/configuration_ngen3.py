# src/transformers/models/ngen3/configuration_ngen3.py

from transformers import PretrainedConfig

class NGen3Config(PretrainedConfig):
    model_type = "ngen3"

    def __init__(self,
                 variant="120M",  # use "139M" or "120M" interchangeably
                 instruct=False,
                 use_moe=False,
                 num_experts=4,
                 vocab_size=50257,
                 dropout=0.1,
                 grad_clip=1.0,
                 grad_accum_steps=1,
                 warmup_iters=100,
                 **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.instruct = instruct
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.warmup_iters = warmup_iters

        # Set model size parameters based on variant.
        # For the 139M variant we use the same parameters as the 120M variant.
        if variant in ["120M", "139M"]:
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768
            self.block_size = 1024
            self.batch_size = 32
            self.learning_rate = 6e-4
            self.weight_decay = 0.1
            self.max_iters = 2000
            self.log_interval = 50
            self.eval_interval = 200
        elif variant == "7M":
            self.n_layer = 4
            self.n_head = 4
            self.n_embd = 128
            self.block_size = 128
            self.batch_size = 16
            self.learning_rate = 3e-4
            self.weight_decay = 0.1
            self.max_iters = 1000
            self.log_interval = 50
            self.eval_interval = 200
        else:
            # Default to 120M parameters if an unrecognized variant is passed.
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768
            self.block_size = 1024
            self.batch_size = 32
            self.learning_rate = 6e-4
            self.weight_decay = 0.1
            self.max_iters = 2000
            self.log_interval = 50
            self.eval_interval = 200
