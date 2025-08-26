# coding=utf-8
""" MiniCPMV model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...models.siglip.configuration_siglip import SiglipVisionConfig
from ...utils import logging
from .. import Qwen3Config


logger = logging.get_logger(__name__)


class MiniCPMVisionConfig(PretrainedConfig):
    model_type = "minicpm_vision_encoder"

    def __init__(
        self,
        patch_size=14,
        max_slice_nums=9,
        scale_resolution=448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "minicpmv":
            config_dict = config_dict["slice_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)



class MiniCPM_V_4_5Config(Qwen3Config):
    model_type = "minicpm_v_4_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
        self,
        use_cache=True,
        query_num=64,
        image_size=448,
        drop_vision_last_layer=True,
        batch_vision_input=True,
        slice_config=None,
        vision_config=None,
        use_image_id=True,
        vision_batch_size=16,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size

        if slice_config is None:
            self.slice_config = MiniCPMVisionConfig(max_slice_nums=1)
        else:
            self.slice_config = MiniCPMVisionConfig(**slice_config)
        self.slice_mode = True

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if vision_config is None:
            self.vision_config = SiglipVisionConfig(**self.default_vision_config)
            #logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config

        self.patch_size = self.vision_config.patch_size

        super().__init__(**kwargs)


__all__ = ["MiniCPM_V_4_5Config"]
