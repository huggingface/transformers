# --------------------------------------------------------
# Eagle2
# Copyright (c) 2025 NVIDIA
# Licensed under The Apache License [see LICENSE for details]
# --------------------------------------------------------

import copy
import os
from transformers import AutoConfig, LlamaConfig, Qwen2Config, SiglipVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MultiBackboneChannelConcatenationVisionModelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MultiBackboneChannelConcatenationVisionModelConfig`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vision_path (str): Path to the vision model or its configuration.
        mm_vision_select_layer (int, optional): The layer to select from the vision model
                                                for multi-modal processing. Defaults to -2.
        grid_size (int, optional): The size of the grid for vision processing. Defaults to 32.
        **kwargs: Additional keyword arguments to be passed to the parent PretrainedConfig.
        
    """

    model_type = 'MOB'

    def __init__(
            self,
            vision_path,
            mm_vision_select_layer=-2,
            grid_size=32,
            input_image_size=1024,
            hidden_size='lazy_calculation',
            image_size=1024,
            freeze_backbones=None,
            moe_version_type=None,
            delay_load=False,
            convnext_img_size=1024,
            vision_tower_siglip_path=None,
            vision_tower_convnext_path='convnext_xxlarge.clip_laion2b_soup',
            normalize_type='siglip',
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.normalize_type = normalize_type
        self.vision_path = vision_path
        self.mm_vision_select_layer = mm_vision_select_layer
        self.grid_size = grid_size
        self.input_image_size = input_image_size
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.freeze_backbones = freeze_backbones
        self.moe_version_type = moe_version_type
        self.delay_load = delay_load
        self.convnext_img_size = convnext_img_size
        # other args. to make it compatable with eagle-next
        self.vision_tower_siglip_path = vision_tower_siglip_path
        self.vision_tower_convnext_path = vision_tower_convnext_path
        self.vision_tower = self.vision_path[4:] # remove `MOB:` prefix

        # asserts
        assert image_size == input_image_size, f"input_image_size ({input_image_size}) != image_size ({image_size})"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)

class Eagle2ChatConfig(PretrainedConfig):
    model_type = 'eagle_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            mlp_checkpoint=True,
            pre_feature_reduction=False,
            keep_aspect_ratio=False,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing Vision Encoders with default values.')

        if llm_config is None:
            llm_config = {}
            logger.info('llm_config is None. Initializing the LLM config with default values')

        if vision_config['model_type'] == 'siglip_vision_model':
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config['model_type'].startswith("MOB"):
            self.vision_config = MultiBackboneChannelConcatenationVisionModelConfig(**vision_config)
        else:
            raise ValueError('Unsupported model_type: {}'.format(vision_config['model_type']))

        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.mlp_checkpoint = mlp_checkpoint
        self.pre_feature_reduction = pre_feature_reduction
        self.keep_aspect_ratio = keep_aspect_ratio
        logger.info(f'keep_aspect_ratio: {self.keep_aspect_ratio}')
        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['keep_aspect_ratio'] = self.keep_aspect_ratio

        return output


__all__ = ["Ealge2ChatConfig", "MultiBackboneChannelConcatenationVisionModelConfig"]