""" PEGASUS model configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "pegasus-large": "placeholder",
    "pegasus-large-cnn_dailymail": "placeholder",
    "pegasus-large-newsroom": "placeholder",
    "pegasus-large-aeslc": "placeholder",
    "pegasus-large-big_patent": "placeholder",
    "pegasus-large-gigaword": "placeholder",
    "pegasus-large-reddit_tifu_long": "placeholder",
    "pegasus-large-wikihow_all": "placeholder",
    "pegasus-large-xsum": "placeholder",
    "pegasus-large-arxiv": "placeholder",
    "pegasus-large-pubmed": "placeholder",
    "pegasus-large-multi_news": "placeholder",
    "pegasus-large-billsum": "placeholder",
}


class PegasusConfig(PretrainedConfig):
    r"""
        :class:`~transformers.PegasusConfig` is the configuration class to store the configuration of a
        `PegasusModel`.
    """
    model_type = "pegasus"

    def __init__(
            self,
            vocab_size=96000,
            max_input_len=512,
            max_target_len=256,
            max_decode_len=256,
            hidden_size=1024,
            ffn_dim=4096,
            num_heads=16,
            num_encoder_layers=16,
            num_decoder_layers=16,
            dropout=0.1,
            is_encoder_decoder=True,
            pad_token_id=0,
            eos_token_id=1,
            **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.max_decode_len = max_decode_len
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

    @property
    def max_position_embeddings(self):
        return self.max_input_len

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_encoder_layers
