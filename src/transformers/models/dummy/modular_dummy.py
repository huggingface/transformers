from torch import nn

from transformers.models.sam.configuration_sam import SamVisionConfig
from transformers.models.sam.modeling_sam import SamLayerNorm, SamVisionNeck


class DummySamLayerNorm(SamLayerNorm):
    pass


class DummySamVisionNeck(SamVisionNeck):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = DummySamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = DummySamLayerNorm(config.output_channels, data_format="channels_first")
