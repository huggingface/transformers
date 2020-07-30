""" PEGASUS model configuration """


import logging

from .configuration_bart import BartConfig
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


class PegasusConfig(BartConfig):
    r"""
        :class:`~transformers.PegasusConfig` is the configuration class to store the configuration of a
        `PegasusModel`.
    """
    model_type = "pegasus"

    def __init__(
        self,
        vocab_size=96000,
        max_position_embeddings=512,
        max_target_len=256,
        max_decode_len=256,
        d_model=1024,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_layers=16,
        decoder_layers=16,
        dropout=0.1,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            decoder_attention_heads=decoder_attention_heads,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=True,
            **kwargs,
        )
