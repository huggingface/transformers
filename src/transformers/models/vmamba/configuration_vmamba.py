# coding=utf-8
# Copyright 2022 Derk Mus and The HuggingFace Inc. team. All rights reserved.
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
""" VMamba model configuration """


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vmamba": "https://huggingface.co/vmamba/resolve/main/config.json",
    # See all VMamba models at https://huggingface.co/models?filter=vmamba
}


class VMambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~VMambaModel`].
    It is used to instantiate an VMamba model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the VMamba [vmamba](https://huggingface.co/vmamba) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the VMamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~VMambaModel`] or
            [`~TFVMambaModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`~VMambaModel`] or
            [`~TFVMambaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import VMambaModel, VMambaConfig

    >>> # Initializing a VMamba vmamba style configuration
    >>> configuration = VMambaConfig()

    >>> # Initializing a model from the vmamba style configuration
    >>> model = VMambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vmamba"

    def __init__(
        self,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.use_checkpoint = False
        super().__init__(**kwargs)
