# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BridgeTower model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
        "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
 }


class BridgeTowerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`bridgetowerModel`]. It is used to instantiate an
    bridgetower model according to the specified arguments=, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower
    [](https://huggingface.co/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args: TODO: TBD once config list is finalized

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a bridgetower  style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the  style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bridgetower"

    def __init__(
        self,      
        cache_dir='/tmp',
        classifier_drop_rate=0.1,
        cross_modal_transform_shared=True,
        downstream_fusion=False,
        downstream_fusion_layers=1,
        downstream_fusion_method='elmo',
        drop_rate=0.1,
        freeze_RoBERTa=False,
        freeze_ViT=False,
        freeze_layer_count_roberta=False,
        freeze_layer_count_vit=False,
        head_hidden_scale=2,
        hidden_size=768,
        image_size=288,
        input_image_embed_size=768,
        input_text_embed_size=768,
        link_tower_shared=False,
        link_tower_type='add',
        log_dir='log_dir',
        loss_names={'contras': 0,
                'irtr': 0,
                'itm': 0,
                'mlm': 0,
                'mpp': 0,
                'nlvr2': 0,
                'snli': 0,
                'vcr': 0,
                'vcr_qar': 0,
                'vqa': 1},
        max_text_len=50, #check 40 in configuration.py
        mlp_ratio=4,
        model_type='bridgetower',
        nlvr2_head_format='pair',
        num_attention_heads=12,
        num_hidden_layers=6,
        num_nodes=1,
        only_load_cross_modal_from_meter=False,
        patch_size=16,
        resolution_before=224,
        stop_gradient=False,
        task_head_layers=2,
        test_only=False,
        tokenizer='roberta-base',
        unfreeze_RoBERTa_attention=False,
        unfreeze_RoBERTa_embeddings=False,
        unfreeze_RoBERTa_encoder=False,
        unfreeze_RoBERTa_layernorm=False,
        unfreeze_ViT_attention=False,
        unfreeze_ViT_layernorm=False,
        vit='ViT-B/16',
        vit_layernorm_init_from_vit=False,
        vit_layernorm_shared=True,
        vit_remove_last=False,
        vocab_size=50265,
        vqav2_label_size=3129,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self.classifier_drop_rate = classifier_drop_rate
        self.cross_modal_transform_shared = cross_modal_transform_shared
        self.downstream_fusion = downstream_fusion
        self.downstream_fusion_layers = downstream_fusion_layers
        self.downstream_fusion_method = downstream_fusion_method
        self.drop_rate = drop_rate
        self.freeze_RoBERTa = freeze_RoBERTa
        self.freeze_ViT = freeze_ViT
        self.freeze_layer_count_roberta = freeze_layer_count_roberta
        self.freeze_layer_count_vit = freeze_layer_count_vit
        self.head_hidden_scale = head_hidden_scale
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.input_image_embed_size = input_image_embed_size
        self.input_text_embed_size = input_text_embed_size
        self.link_tower_shared = link_tower_shared
        self.link_tower_type = link_tower_type
        self.log_dir = log_dir
        self.loss_names = loss_names
        self.max_text_len = max_text_len
        self.mlp_ratio = mlp_ratio
        self.model_type = model_type
        self.nlvr2_head_format = nlvr2_head_format
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        self.only_load_cross_modal_from_meter = only_load_cross_modal_from_meter
        self.patch_size = patch_size
        self.resolution_before = resolution_before
        self.stop_gradient = stop_gradient
        self.task_head_layers = task_head_layers
        self.test_only = test_only
        self.tokenizer = tokenizer
        self.unfreeze_RoBERTa_attention = unfreeze_RoBERTa_attention
        self.unfreeze_RoBERTa_embeddings = unfreeze_RoBERTa_embeddings
        self.unfreeze_RoBERTa_encoder = unfreeze_RoBERTa_encoder
        self.unfreeze_RoBERTa_layernorm = unfreeze_RoBERTa_layernorm
        self.unfreeze_ViT_attention = unfreeze_ViT_attention
        self.unfreeze_ViT_layernorm = unfreeze_ViT_layernorm
        self.vit = vit
        self.vit_layernorm_init_from_vit = vit_layernorm_init_from_vit
        self.vit_layernorm_shared = vit_layernorm_shared
        self.vit_remove_last = vit_remove_last
        self.vocab_size = vocab_size
        self.vqav2_label_size = vqav2_label_size
