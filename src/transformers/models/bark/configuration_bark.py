# coding=utf-8
# Copyright 2023 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" BARK model configuration"""

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


BARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "suno/bark": "https://huggingface.co/suno/bark/resolve/main/config.json",
}



class BarkModuleConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BartModule`]. It is used to instantiate Bark
    sub-models according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Bark
    [suno/bark](https://huggingface.co/suno/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        block_size (int, optional):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048). Defaults to 1024.
        input_vocab_size (_type_, optional): _description_.
        Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BarkModule`]. Defaults to 10_048 but should be carefully thought with
            regards to the chosen sub-model.
        output_vocab_size (_type_, optional):
        Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the:
            `output_ids` when passing forward a [`BarkModule`]. Defaults to 10_048 but should be carefully thought with
            regards to the chosen sub-model.
        num_layers (int, optional):
            Number of layers. Defaults to 12.
        num_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer architecture. Defaults to 12.
        hidden_size (int, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the architecture. Defaults to 768.
        dropout (float, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler. Defaults to
            0.0.
        bias (bool, optional):
            True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster. Defaults to True.
        use_cache (bool, optional):
            Whether or not the model should return the last key/values attentions (not used by all models). Defaults to
            True.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        n_codes_total (`int`, *optional*, defaults to 8):
            The total number of [`Encodec`] codebooks. Used in the fine acoustics sub-model.
        n_codes_given (`int`, *optional*, defaults to 8):
            The number of [`Encodec`] codebooks predicted in the coarse acoustics sub-model. Use in the acoustics
            sub-models.
    Example:

    ```python
    >>> from transformers import BarkModuleConfig, BarkModule

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkModuleConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkModule(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bark_module"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        block_size=1024,
        input_vocab_size=10_048,
        output_vocab_size=10_048,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        dropout=0.0,
        bias=True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        n_codes_total=8,  # for BarkFineAcousticsModel
        n_codes_given=1,  # for BarkFineAcousticsModel
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class BarkConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Bark`]. It is used to instantiate a Bark model
    according to the specified sub-models configurations, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Bark
    [suno/bark](https://huggingface.co/suno/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    semantic_config (BarkModuleConfig, optional):
        Configuration of the underlying semantic sub-model. Defaults to None.
    coarse_acoustics_config (BarkModuleConfig, optional):
        Configuration of the underlying coarse acoustics sub-model. Defaults to None.
    fine_acoustics_config (BarkModuleConfig, optional):
        Configuration of the underlying fine acoustics sub-model. Defaults to None.

    Example:

    ```python
    >>> from transformers import BarkModuleConfig, Bark, BarkConfig

    >>> # Initializing Bark sub-modules configurations.
    >>> semantic_config = BarkModuleConfig()
    >>> coarse_acoustics_config = BarkModuleConfig()
    >>> fine_acoustics_config = BarkModuleConfig()


    >>> # Initializing a Bark module style configuration
    >>> configuration = BarkConfig(semantic_config, coarse_acoustics_config, fine_acoustics_config)

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = Bark(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bark"
    is_composition = True

    def __init__(
        self,
        semantic_config: BarkModuleConfig = None,
        coarse_acoustics_config: BarkModuleConfig = None,
        fine_acoustics_config: BarkModuleConfig = None,
        **kwargs,
    ):
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the Semantic module with default values.")

        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info(
                "coarse_acoustics_config is None. initializing the Coarse Acoustics module with default values."
            )

        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the Fine Acoustics module with default values.")

        self.semantic_config = BarkModuleConfig(**semantic_config)
        self.coarse_acoustics_config = BarkModuleConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkModuleConfig(**fine_acoustics_config)

        # TODO: check if right place
        # some of these configs are linked to the config of the submodels
        self.text_encoding_offset = 10_048
        self.semantic_pad_token = 10_000
        self.text_pad_token = 129_595
        self.semantic_infer_token = 129_599
        self.coarse_semantic_pad_token = 12_048
        self.coarse_infer_token = 12_050
        self.context_window_size = 1024
        self.semantic_rate_hz = 49.9
        self.semantic_vocab_size = 10_000
        self.codebook_size = 1024
        self.n_coarse_codebooks = 2  # fixed for now
        self.n_fine_codebooks = 8  # fixed for now
        self.coarse_rate_hz = 75
        self.sample_rate = 24_000

        super().__init__(**kwargs)

    @classmethod
    def from_configs(
        cls,
        semantic_config: BarkModuleConfig,
        coarse_acoustics_config: BarkModuleConfig,
        fine_acoustics_config: BarkModuleConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`BarkConfig`] (or a derived class) from bark modules configuration configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        """
        return cls(
            semantic_config=semantic_config.to_dict(),
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["semantic_config"] = self.semantic_config.to_dict()
        output["coarse_acoustics_config"] = self.coarse_acoustics_config.to_dict()
        output["fine_acoustics_config"] = self.fine_acoustics_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output

