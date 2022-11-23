""" ViViT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class ViViTConfig(PretrainedConfig):

    model_type = "vivit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_fast",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        video_size=(32, 224, 224),
        tubelet_size=(2, 16, 16),
        num_channels=3,
        qkv_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.video_size = video_size
        self.tubelet_size = tubelet_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
