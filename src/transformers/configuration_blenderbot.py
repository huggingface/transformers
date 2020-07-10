import os

from transformers.configuration_bart import BartConfig


BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {"blenderbot": "sshleifer/blenderbot-3B"}


class BlenderbotConfig(BartConfig):
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
            n_layers: (:obj:`int`, default to 8), number of layers
            ffn_size: (:obj:`int`, default to 2048), size of hidden layers in the FFN
            dropout: (:obj:`float`, default to 0.1), embedding dropout
            n_heads:(:obj:`int`, default to 16),  number of multi heads attention
            n_positions:(:obj:`int`, default to 512), size of the position embeddings
            activation: (:obj:`string`, default to 'gelu'), activation function to use
            attention_dropout: (:obj:`float`, default to 0.0), multi head attention dropout
            relu_dropout: (:obj:`float`, default to 0.0), relu dropout
            vocab_size: (:obj:`int`, default to 54944), the size of the vocabulary
            learn_positional_embedding: (:obj:`boolean`, default to True),  if yes or no the positional embeddings will be learn

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
        extra_pos_embeddings=0,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            d_model=d_model,
            init_std=init_std,
            max_position_embeddings=max_position_embeddings,
            static_position_embeddings=static_position_embeddings,
            vocab_size=vocab_size,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            decoder_layerdrop=decoder_layerdrop,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_function=activation_function,
            normalize_before=normalize_before,
            scale_embedding=scale_embedding,
            normalize_embedding=normalize_embedding,
            add_final_layer_norm=add_final_layer_norm,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.extra_pos_embeddings = extra_pos_embeddings
