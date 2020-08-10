import os

from .modeling_utils import PretrainedConfig


BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {"blenderbot": "sshleifer/blenderbot-3B"}


class BlenderbotConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotModel`.
        It is used to instantiate an Blenderbot model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `blenderbot <https://huggingface.co/blenderbot>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.
        Args:
            embedding_size: (:obj:`int`, default to 512), dimension of the embeddings vector
            encoder_layers: (:obj:`int`, default to 8), number of layers in the encoder
            decoder_layers: (:obj:`int`, default to 8), number of layers in the decoder
            encoder_ffn_dim: (:obj:`int`, default to 2048), size of hidden layers in the FFN in the encoder
            decoder_ffn_dim: (:obj:`int`, default to 2048), size of hidden layers in the FFN in the decoder
            dropout: (:obj:`float`, default to 0.1), embedding dropout
            encoder_attention_heads:(:obj:`int`, default to 16),  number of multi heads attention in the encoder
            decoder_attention_heads:(:obj:`int`, default to 16),  number of multi heads attention in the decoder
            max_position_embeddings:(:obj:`int`, default to 512), size of the position embeddings
            activation: (:obj:`string`, default to 'gelu'), activation function to use
            attention_dropout: (:obj:`float`, default to 0.0), multi head attention dropout
            relu_dropout: (:obj:`float`, default to 0.0), relu dropout
            vocab_size: (:obj:`int`, default to 54944), the size of the vocabulary
            static_position_embeddings: (:obj:`boolean`, default to False),  if yes or no the positional embeddings will be learn,
            d_model: (:obj:`int`, default to 2560), hidden size dimention
            variant: (obj: str, default to 'prelayernorm') defines when to apply a layer norm
            scale_embedding: (obj: boolean, default to Faulse) if set to True, scale by sqrt(d_model)
            normalize_embedding: (obj: boolean, default to False) if set to True apply Layernorm after the embedding
            extra_pos_embeddings: (obj: int, default to 0) pos embedding offset

        Example::

            from transformers import BlenderbotModel, BlenderbotConfig
            # Initializing a Blenderbot configuration
            configuration = BlenderbotConfig()
            # Initializing a model from the configuration
            model = BlenderbotModel(configuration)
            # Accessing the model configuration
            configuration = model.config
        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """

    pretrained_config_archive_map = BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "blenderbot"

    def __init__(
        self,
        d_model=2560,
        dropout=0.1,
        encoder_ffn_dim=10240,
        encoder_layers=2,
        encoder_attention_heads=32,
        decoder_ffn_dim=10240,
        decoder_layers=4,
        decoder_attention_heads=32,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        vocab_size=8008,
        activation_dropout=0.0,
        init_std=0.02,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        unk_token_id=3,
        activation_function="gelu",
        normalize_before=True,
        add_final_layer_norm=False,
        scale_embedding=False,
        normalize_embedding=False,
        static_position_embeddings=False,
        is_encoder_decoder=True,
        variant="prelayernorm",
        extra_pos_embeddings=0,
        num_beams=10,
        min_length=20,
        length_penalty=0.65,  # add no_repeat_n_gram?
        **kwargs
    ):
        r"""
            :class:`~transformers.BlenderbotConfig` is the configuration class for `BlenderbotForConditionalGeneration`.

            Examples::

                >>> from transformers import BlenderbotForConditionalGeneration, BlenderbotConfig

                >>> config = BlenderbotConfig.from_pretrained('facebook/blenderbot-3B')
                >>> model = BlenderbotForConditionalGeneration(config)
        """
        if "hidden_size" in kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std 
        self.activation_function = activation_function
        self.variant = variant
        self.scale_embedding = scale_embedding  
        self.normalize_embedding = normalize_embedding  
        self.normalize_before = normalize_before  
        self.add_final_layer_norm = add_final_layer_norm

        self.static_position_embeddings = static_position_embeddings

        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout


        self.extra_pos_embeddings = extra_pos_embeddings

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
        
        
