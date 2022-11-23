# coding=utf-8
# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" AltCLIP model configuration """
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.clip.configuration_clip import CLIPConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BAAI/AltCLIP": "https://huggingface.co/BAAI/AltCLIP/resolve/main/config.json",
    # See all AltCLIP models at https://huggingface.co/models?filter=altclip
}


class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=512,pooler_fn='cls', **kwargs):
        super().__init__(pad_token_id, bos_token_id, eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn


class AltClipConfig(CLIPConfig):
    r"""
    This is the configuration class to store the configuration of a [`~AltClipModel`].
    It is used to instantiate an AltCLIP model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the AltCLIP [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the AltCLIP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~AltClipModel`] or
            [`~TFAltClipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~AltClipModel`] or
            [`~TFAltClipModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import AltClipModel, AltClipConfig

    >>> # Initializing a AltCLIP BAAI/AltCLIP style configuration
    >>> configuration = AltClipConfig()

    >>> # Initializing a model from the BAAI/AltCLIP style configuration
    >>> model = AltClipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "altclip"
    

    def __init__(
        self, 
        text_model_name=None,
        vision_model_name=None,
        text_config_dict=None, 
        vision_config_dict=None, 
        projection_dim=512, 
        logit_scale_init_value=2.6592, 
        **kwargs):
        super().__init__(text_config_dict, vision_config_dict, projection_dim, logit_scale_init_value, **kwargs)
        if text_config_dict is None:
            text_config_dict = {}
        # when reload the config from local, we need name to select which class should be instanced.
        self.text_config = RobertaSeriesConfig(**kwargs.pop('text_config'))
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name

    