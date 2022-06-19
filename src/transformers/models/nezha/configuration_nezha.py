from transformers import PretrainedConfig


NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class NeZhaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an :class:`~transformers.NeZhaModel`. It is used to
    instantiate an NeZha model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the NeZha base architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, optional, defaults to 21128):
            Vocabulary size of the ALBERT model. Defines the different tokens that can be represented by the
            `inputs_ids` passed to the forward method of :class:`~transformers.NeZhaModel`.
        embedding_size (:obj:`int`, optional, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (:obj:`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, optional, defaults to 3072):
            The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, optional, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.NeZhaModel`.
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for attached classifiers.

    Example::

        from transformers import NeZhaConfig, NeZhaModel # Initializing an NeZha configuration nezha_config =
        NeZhaConfig()

        # Initializing a model from the NeZha-base style configuration model = NeZhaModel(nezha_config)

        # Accessing the model configuration configuration = model.config

    Attributes:
        pretrained_config_archive_map (Dict[str, str]):
            A dictionary containing all the available pre-trained checkpoints.
    """

    pretrained_config_archive_map = NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "nezha"

    def __init__(
        self,
        vocab_size=21128,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_relative_position=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
