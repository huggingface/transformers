from transformers.configuration_utils import PretrainedConfig
import os

BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "blenderbot": os.path.abspath(os.path.expanduser('blenderbot-90M-config.json'))
}


class BlenderbotConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotModel`.
        It is used to instantiate an Blenderbot model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `blenderbot <https://huggingface.co/blenderbot>`__ architecture.

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

    def __init__(self,
                 embedding_size=512,
                 n_layers=8,
                 ffn_size=2048,
                 dropout=0.1,
                 n_heads=16,
                 n_positions=512,
                 activation='gelu',
                 vocab_size=54944,
                 learn_positional_embeddings=True,
                 attention_dropout=0.0,
                 relu_dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.n_heads = n_heads
        self.n_positions = n_positions
        self.dropout = dropout
        self.activation = activation
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.vocab_size = vocab_size
        self.learn_positional_embeddings = learn_positional_embeddings

        @property
        def n_layers(self):
            return self.n_layers

        @property
        def n_heads(self):
            return self.n_heads

        @property
        def n_positions(self):
            return self.n_positions

        @property
        def ffn_size(self):
            return self.ffn_size

        @property
        def embedding_size(self):
            return self.embedding_size

