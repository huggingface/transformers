# coding=utf-8
# Copyright 2022 The Trajectory Transformers paper authors and The HuggingFace Inc. team. All rights reserved.
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
""" TrajectoryTransformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2": (
        "https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2/resolve/main/config.json"
    ),
    # See all TrajectoryTransformer models at https://huggingface.co/models?filter=trajectory_transformer
}


class TrajectoryTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TrajectoryTransformerModel`]. It is used to
    instantiate an TrajectoryTransformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    TrajectoryTransformer
    [CarlCochet/trajectory-transformer-halfcheetah-medium-v2](https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 100):
            Vocabulary size of the TrajectoryTransformer model. Defines the number of different tokens that can be
            represented by the `trajectories` passed when calling [`TrajectoryTransformerModel`]
        batch_size (`int`, *optional*, defaults to 256):
            Size of the batch of trajectories passed to the model.
        action_weight (`int`, *optional*, defaults to 5):
            Weight of the action in the loss function
        reward_weight (`int`, *optional*, defaults to 1):
            Weight of the reward in the loss function
        value_weight (`int`, *optional*, defaults to 1):
            Weight of the value in the loss function
        block_size (`int`, *optional*, defaults to 249):
            Size of the blocks in the trajectory transformer.
        action_dim (`int`, *optional*, defaults to 6):
            Dimension of the action space.
        observation_dim (`int`, *optional*, defaults to 17):
            Dimension of the observation space.
        transition_dim (`int`, *optional*, defaults to 25):
            Dimension of the transition space.
        n_layer (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_embd (`int`, *optional*, defaults to 128):
            Dimensionality of the embeddings and hidden states.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`TrajectoryTransformerModel`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        kaiming_initializer_range (`float, *optional*, defaults to 1):
            A coefficient scaling the negative slope of the kaiming initializer rectifier for EinLinear layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

    >>> # Initializing a TrajectoryTransformer CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
    >>> configuration = TrajectoryTransformerConfig()

    >>> # Initializing a model (with random weights) from the CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
    >>> model = TrajectoryTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "trajectory_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=100,
        batch_size=256,
        action_weight=5,
        reward_weight=1,
        value_weight=1,
        block_size=249,
        action_dim=6,
        observation_dim=17,
        transition_dim=25,
        n_layer=4,
        n_head=4,
        n_embd=128,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        learning_rate=0.0006,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        kaiming_initializer_range=1,
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=1,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.action_weight = action_weight
        self.reward_weight = reward_weight
        self.value_weight = value_weight
        self.max_position_embeddings = max_position_embeddings
        self.block_size = block_size
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.transition_dim = transition_dim
        self.learning_rate = learning_rate
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.kaiming_initializer_range = kaiming_initializer_range
        self.use_cache = use_cache
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
