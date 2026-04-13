# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""BARK model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkSubModelConfig(PreTrainedConfig):
    r"""
    block_size (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    input_vocab_size (`int`, *optional*, defaults to 10_048):
        Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
        regards to the chosen sub-model.
    output_vocab_size (`int`, *optional*, defaults to 10_048):
        Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
        by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
        with regards to the chosen sub-model.
    bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the linear layers and layer norm layers.
    """

    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "input_vocab_size",
        "window_size": "block_size",
    }

    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    dropout: float | int = 0.0
    bias: bool = True
    initializer_range: float = 0.02
    use_cache: bool = True


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkSemanticConfig(BarkSubModelConfig):
    r"""
    block_size (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    input_vocab_size (`int`, *optional*, defaults to 10_048):
        Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
        regards to the chosen sub-model.
    output_vocab_size (`int`, *optional*, defaults to 10_048):
        Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
        by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
        with regards to the chosen sub-model.
    bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the linear layers and layer norm layers

    Example:

    ```python
    >>> from transformers import BarkSemanticConfig, BarkSemanticModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkSemanticConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkSemanticModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "semantic"
    base_config_key = "semantic_config"


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkCoarseConfig(BarkSubModelConfig):
    r"""
    block_size (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    input_vocab_size (`int`, *optional*, defaults to 10_048):
        Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
        regards to the chosen sub-model.
    output_vocab_size (`int`, *optional*, defaults to 10_048):
        Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
        by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
        with regards to the chosen sub-model.
    bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the linear layers and layer norm layers

    Example:

    ```python
    >>> from transformers import BarkCoarseConfig, BarkCoarseModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkCoarseConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkCoarseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "coarse_acoustics"
    base_config_key = "coarse_acoustics_config"


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkFineConfig(BarkSubModelConfig):
    r"""
    block_size (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    input_vocab_size (`int`, *optional*, defaults to 10_048):
        Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
        regards to the chosen sub-model.
    output_vocab_size (`int`, *optional*, defaults to 10_048):
        Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
        by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
        with regards to the chosen sub-model.
    bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the linear layers and layer norm layers
    n_codes_total (`int`, *optional*, defaults to 8):
        The total number of audio codebooks predicted. Used in the fine acoustics sub-model.
    n_codes_given (`int`, *optional*, defaults to 1):
        The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics
        sub-models.

    Example:

    ```python
    >>> from transformers import BarkFineConfig, BarkFineModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkFineConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkFineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fine_acoustics"
    base_config_key = "fine_acoustics_config"

    tie_word_embeddings: bool = True
    n_codes_total: int = 8
    n_codes_given: int = 1


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkConfig(PreTrainedConfig):
    r"""
    semantic_config ([`BarkSemanticConfig`], *optional*):
        Configuration of the underlying semantic sub-model.
    coarse_acoustics_config ([`BarkCoarseConfig`], *optional*):
        Configuration of the underlying coarse acoustics sub-model.
    fine_acoustics_config ([`BarkFineConfig`], *optional*):
        Configuration of the underlying fine acoustics sub-model.
    codec_config ([`AutoConfig`], *optional*):
        Configuration of the underlying codec sub-model.

    Example:

    ```python
    >>> from transformers import (
    ...     BarkSemanticConfig,
    ...     BarkCoarseConfig,
    ...     BarkFineConfig,
    ...     BarkModel,
    ...     BarkConfig,
    ...     AutoConfig,
    ... )

    >>> # Initializing Bark sub-modules configurations.
    >>> semantic_config = BarkSemanticConfig()
    >>> coarse_acoustics_config = BarkCoarseConfig()
    >>> fine_acoustics_config = BarkFineConfig()
    >>> codec_config = AutoConfig.from_pretrained("facebook/encodec_24khz")


    >>> # Initializing a Bark module style configuration
    >>> configuration = BarkConfig(
    ...     semantic_config, coarse_acoustics_config, fine_acoustics_config, codec_config
    ... )

    >>> # Initializing a model (with random weights)
    >>> model = BarkModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bark"
    sub_configs = {
        "semantic_config": BarkSemanticConfig,
        "coarse_acoustics_config": BarkCoarseConfig,
        "fine_acoustics_config": BarkFineConfig,
        "codec_config": AutoConfig,
    }
    semantic_config: dict | PreTrainedConfig | None = None
    coarse_acoustics_config: dict | PreTrainedConfig | None = None
    fine_acoustics_config: dict | PreTrainedConfig | None = None
    codec_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.semantic_config is None:
            self.semantic_config = BarkSemanticConfig()
            logger.info("`semantic_config` is `None`. Initializing the `BarkSemanticConfig` with default values.")
        elif isinstance(self.semantic_config, dict):
            self.semantic_config = BarkSemanticConfig(**self.semantic_config)

        if self.coarse_acoustics_config is None:
            self.coarse_acoustics_config = BarkCoarseConfig()
            logger.info(
                "`coarse_acoustics_config` is `None`. Initializing the `BarkCoarseConfig` with default values."
            )
        elif isinstance(self.coarse_acoustics_config, dict):
            self.coarse_acoustics_config = BarkCoarseConfig(**self.coarse_acoustics_config)

        if self.fine_acoustics_config is None:
            self.fine_acoustics_config = BarkFineConfig()
            logger.info("`fine_acoustics_config` is `None`. Initializing the `BarkFineConfig` with default values.")
        elif isinstance(self.fine_acoustics_config, dict):
            self.fine_acoustics_config = BarkFineConfig(**self.fine_acoustics_config)

        if self.codec_config is None:
            self.codec_config = CONFIG_MAPPING["encodec"]()
            logger.info("`codec_config` is `None`. Initializing the `codec_config` with default values.")
        elif isinstance(self.codec_config, dict):
            codec_model_type = self.codec_config.get("model_type", "encodec")
            self.codec_config = CONFIG_MAPPING[codec_model_type](**self.codec_config)

        super().__post_init__(**kwargs)


__all__ = ["BarkCoarseConfig", "BarkConfig", "BarkFineConfig", "BarkSemanticConfig"]
