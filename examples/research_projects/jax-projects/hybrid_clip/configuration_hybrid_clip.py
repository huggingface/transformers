import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HybridCLIPConfig(PretrainedConfig):
    r"""
    :class:`~transformers.CLIPConfig` is the configuration class to store the configuration of a
    :class:`~transformers.CLIPModel`. It is used to instantiate CLIP model according to the specified arguments,
    defining the text model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        text_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPTextConfig`.
        vision_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPVisionConfig`.
        projection_dim (:obj:`int`, `optional`, defaults to 512):
            Dimentionality of text and vision projection layers.
        kwargs (`optional`):
            Dictionary of keyword arguments.
    """

    model_type = "hybrid-clip"
    is_composition = True

    def __init__(self, text_config_dict=None, vision_config_dict=None, projection_dim=512, **kwargs):
        super().__init__(text_config_dict=text_config_dict, vision_config_dict=vision_config_dict, **kwargs)

        if text_config_dict is None:
            text_config_dict = {}
            logger.info("text_config_dict is None. Initializing the CLIPTextConfig with default values.")

        if vision_config_dict is None:
            vision_config_dict = {}
            logger.info("vision_config_dict is None. initializing the CLIPVisionConfig with default values.")

        text_model_type = text_config_dict.pop("model_type")
        vision_model_type = vision_config_dict.pop("model_type")

        from transformers import AutoConfig

        self.text_config = AutoConfig.for_model(text_model_type, **text_config_dict)

        #         if vision_model_type == "clip_vision_model":
        #             self.vision_config = CLIPVisionConfig(**vision_config_dict)
        #         else:
        #             self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config_dict).vision_config

        self.vision_config = CLIPVisionConfig(**vision_config_dict)

        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: PretrainedConfig, vision_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a :class:`~transformers.CLIPConfig` (or a derived class) from clip text model configuration and
        clip vision model configuration.

        Returns:
            :class:`CLIPConfig`: An instance of a configuration object
        """

        return cls(text_config_dict=text_config.to_dict(), vision_config_dict=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
