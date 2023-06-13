from ...configuration_utils import PretrainedConfig
from ...utils import logging

import copy

logger = logging.get_logger(__name__)


class BarkModuleConfig(PretrainedConfig):
    r"""
    """
    
    
    model_type = "bark_module"
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


# WIP
class BarkConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Bark`].
    """

    model_type = "bark"
    is_composition = True

    def __init__(
        self,
        semantic_config=None,
        coarse_acoustics_config=None,
        fine_acoustics_config=None,
        **kwargs,
    ):
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the Semantic module with default values.")
            
        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the Coarse Acoustics module with default values.")

        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the Fine Acoustics module with default values.")

        self.semantic_config = BarkModuleConfig(**semantic_config)
        self.coarse_acoustics_config = BarkModuleConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkModuleConfig(**fine_acoustics_config)
        
        
        # TODO: check if right place and that is necessary
        self.text_encoding_offset = 10_048
        self.semantic_pad_token = 10_000
        self.text_pad_token = 129_595
        self.semantic_infer_token = 129_599
        self.coarse_semantic_pad_token = 12_048
        self.coarse_infer_token = 12_050
        self.context_window_size = 1024
        self.semantic_rate_hz = 49.9
        self.semantic_vocab_size = 10_000
        self.codebook_size = 1024
        self.n_coarse_codebooks = 2 # fixed for now
        self.n_fine_codebooks = 8 # fixed for now
        self.coarse_rate_hz = 75
        self.sample_rate = 24_000
     
    
        super().__init__(**kwargs)

    @classmethod
    def from_configs(cls, 
            semantic_config: BarkModuleConfig,
            coarse_acoustics_config: BarkModuleConfig,
            fine_acoustics_config: BarkModuleConfig,
            **kwargs):
        r"""
        Instantiate a [`BarkConfig`] (or a derived class) from bark modules configuration configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        """
        return cls(semantic_config=semantic_config.to_dict(), coarse_acoustics_config=coarse_acoustics_config.to_dict(), fine_acoustics_config=fine_acoustics_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["semantic_config"] = self.semantic_config.to_dict()
        output["coarse_acoustics_config"] = self.coarse_acoustics_config.to_dict()
        output["fine_acoustics_config"] = self.fine_acoustics_config.to_dict()
        
        output["model_type"] = self.__class__.model_type
        return output

