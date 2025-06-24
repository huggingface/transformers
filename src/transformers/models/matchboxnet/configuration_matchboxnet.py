from transformers.configuration_utils import PretrainedConfig
import json

MATCHBOXNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class MatchboxNetConfig(PretrainedConfig):
    model_type = "matchboxnet"
    pretrained_config_archive_map = MATCHBOXNET_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        input_channels=64,
        num_classes=30,
        B=3,
        R=2,
        C=64,
        kernel_sizes=None,
        target_sr=16000,
        n_mfcc=64,
        fixed_length=128,
        label2id=None,
        id2label=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.B = B
        self.R = R
        self.C = C
        self.kernel_sizes = kernel_sizes
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.fixed_length = fixed_length
        self.label2id = label2id or {}
        self.id2label = id2label or {}

    @classmethod
    def from_json_string(cls, json_string, **kwargs):
        """Instantiate a config from a JSON string."""
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Instantiate a config from a Python dict."""
        return super().from_dict(config_dict, **kwargs)
