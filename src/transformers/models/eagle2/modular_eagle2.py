


# --------------------------------------------------------
# Eagle2 
# Copyright (c) 2025 NVIDIA
# Licensed under The Apache License [see LICENSE for details]
# --------------------------------------------------------


import copy
import os
from transformers import AutoConfig, Qwen2Config, SiglipVisionConfig
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

class Eagle2TextConfig(Qwen2Config):
    def __init__(self, add_cross_attention, bad_words_ids, begin_suppress_tokens, chunk_size_feed_forward, cross_attention_hidden_size, decoder_start_token_id, diversity_penalty, early_stopping, encoder_no_repeat_ngram_size, exponential_decay_length_penalty, finetuning_task, forced_bos_token_id, forced_eos_token_id, id2label, is_decoder, is_encoder_decoder, label2id, length_penalty, max_length, min_length, no_repeat_ngram_size, num_beam_groups, num_beams, num_return_sequences, output_attentions, output_hidden_states, output_scores, prefix, problem_type, pruned_heads, remove_invalid_values, return_dict, return_dict_in_generate, sep_token_id, suppress_tokens, task_specific_params, tf_legacy_loss, tie_encoder_decoder, tokenizer_class, torchscript, typical_p, use_bfloat16, **kwargs):
        super().__init__(**kwargs)
        self.add_cross_attention = add_cross_attention
        self.bad_words_ids = bad_words_ids
        self.begin_suppress_tokens = begin_suppress_tokens
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.decoder_start_token_id = decoder_start_token_id
        self.diversity_penalty = diversity_penalty
        self.early_stopping = early_stopping
        self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.finetuning_task = finetuning_task
        self.forced_bos_token_id = forced_bos_token_id
        self.forced_eos_token_id = forced_eos_token_id
        self.id2label = id2label
        self.is_decoder = is_decoder
        self.is_encoder_decoder = is_encoder_decoder
        self.label2id = label2id
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.min_length = min_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_beam_groups = num_beam_groups
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_scores = output_scores
        self.prefix = prefix
        self.problem_type = problem_type
        self.pruned_heads = pruned_heads
        self.remove_invalid_values = remove_invalid_values
        self.return_dict = return_dict
        self.return_dict_in_generate = return_dict_in_generate
        self.sep_token_id = sep_token_id
        self.suppress_tokens = suppress_tokens
        self.task_specific_params = task_specific_params
        self.tf_legacy_loss = tf_legacy_loss
        self.tie_encoder_decoder = tie_encoder_decoder
        self.tokenizer_class = tokenizer_class
        self.torchscript = torchscript
        self.typical_p = typical_p
        self.use_bfloat16 = use_bfloat16


class Eagle2Config(PretrainedConfig):
    model_type = 'ealge2'
    sub_configs = {"text_config": Eagle2TextConfig, "vision_config": AutoConfig | MultiBackboneChannelConcatenationVisionModelConfig}

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

        
        self.llm_config = Eagle2TextConfig(**llm_config)

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

