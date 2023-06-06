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
                
                
        initializer_range=0.02,
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
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    