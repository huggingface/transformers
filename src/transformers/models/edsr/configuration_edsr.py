from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from typing import Dict, List, Optional
import copy


logger = logging.get_logger(__name__)

EDSR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class EDSRConfig(PretrainedConfig):
    r"""
     This is the configuration class to store the configuration of a [`EDSRModel`]. It is used to instantiate an
    enhanced Deep Residual networks for super resolution according to the specified arguments, defining the model
    architecture. Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.
    Args:
        upscale (`int`, defaults to 2):
            The scale factor of the model, can be 2, 3, or 4
            upscale = 2 => Model outputs an image which is 2x bigger than the input image.
            upscale = 3 => Model outputs an image which is 3x bigger than the input image.
            upscale = 4 => Model outputs an image which is 4x bigger than the input image.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        num_res_block (`int`, *optional*, defaults to 16):
            Number of residual blocks.
        num_feature_maps (`int`, *optional*, defaults to 64):
            Number of feature maps.
        res_scale (`int`, *optional*, defaults to 1):
            Residual scaling.
        shift_mean (`bool`, *optional*, defaults to True):
            whether or not to subtract pixel mean from the input
        self_ensemble (`bool`, *optional*, defaults to True):
            whether or not to use self-ensemble method for test

    Example:
    ```python
    >>> from transformers import EDSRConfig, EDSRModel

    >>> configuration = EDSRConfig()
    >>> # Initializing a model (with random weights) from the microsoft/swin2sr_tiny_patch4_windows8_256 style configuration
    >>> model = EDSRModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        upscale: int = 2,
        num_channels: int = 3,
        hidden_act: str = "relu",
        num_res_block: int = 16,
        num_feature_maps: int = 64,
        res_scale: int = 1,
        rgb_range: int = 255,
        rgb_mean: tuple = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple = (1.0, 1.0, 1.0),
        shift_mean: bool = True,
        self_ensemble: bool = True,
        **kwargs,
    ):
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](
                # image_size=224,
                # in_channels=3,
                # patch_size=4,
                # embed_dim=96,
                # depths=[2, 2, 18, 2],
                # num_heads=[3, 6, 12, 24],
                # window_size=7,
                # drop_path_rate=0.3,
                # use_absolute_embeddings=False,
                # out_features=["stage1", "stage2", "stage3", "stage4"],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)


        self.backbone_config = backbone_config
        self.upscale = upscale
        self.num_channels = num_channels
        self.hidden_act = hidden_act
        self.num_res_block = num_res_block
        self.num_feature_maps = num_feature_maps
        self.res_scale = res_scale
        self.shift_mean = shift_mean
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.self_ensemble = self_ensemble

        super().__init__(**kwargs)

        # This is the configuration of EDSR Baseline x2 with 16 res blocks and 64 feature maps.
        # TODO: Implement to_dict method, example at DPT Config.
        # TODO: Remove Swin2sr references
        # TODO: push_to_hub example at mask2former
        # TODO: Meanshift to EDSRMeanshift, BasicBlock to EDSRBlock
        # TODO: Mean and std in config
    
    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`EDSRConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.

        Returns:
            [`EDSRConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
