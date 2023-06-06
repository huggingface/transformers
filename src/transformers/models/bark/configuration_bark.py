# TODO: look at how to config

from ...configuration_utils import PretrainedConfig

# TODO: PretrainedConfig -> PreTrainedConfig
# TODO: do I do two config for BarkModule and BarkModel ?

class BarkConfig(PretrainedConfig):
    r"""
    """
    
    
    model_type = "bark"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        block_size=1024,
        input_vocab_size=10_048,
        output_vocab_size=10_048,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        dropout=0.0,
        bias=True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster 
        n_codes_total=8, # for BarkFineAcousticsModel       
        n_codes_given=1, # for BarkFineAcousticsModel       
        
        use_return_dict=False, # TODO
        use_cache=True, # TODO
        #bos_token_id=50256, # TODO
        #eos_token_id=50256, # TODO
        **kwargs,
    ):
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given
        self.use_return_dict = use_return_dict
        self.use_cache = use_cache

        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

        super().__init__(**kwargs)

    