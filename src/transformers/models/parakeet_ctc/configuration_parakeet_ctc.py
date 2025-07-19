# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..fastconformer.configuration_fastconformer import FastConformerConfig


logger = logging.get_logger(__name__)


class ParakeetCTCConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetCTC`]. It is used to instantiate a
    Parakeet CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size of the CTC head. Defines the number of different tokens that can be predicted by the model.
        blank_token_id (`int`, *optional*, defaults to 0):
            The id of the blank token used in CTC. Typically 0.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
            The reduction method for CTC loss. Can be "mean", "sum", or "none".
        ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
            Whether to set infinite losses to zero in CTC loss computation.
        encoder_config (`FastConformerConfig`, *optional*):
            Configuration for the FastConformer encoder.

    Example:
        ```python
        >>> from transformers import ParakeetCTC, ParakeetCTCConfig

        >>> # Initializing a ParakeetCTC configuration
        >>> configuration = ParakeetCTCConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetCTC(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the Parakeet CTC architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_ctc"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "encoder_config": FastConformerConfig,
    }

    def __init__(
        self,
        vocab_size=1024,
        blank_token_id=0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # CTC-specific parameters
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # FastConformer encoder configuration
        if encoder_config is None:
            self.encoder_config = FastConformerConfig()
            logger.info("encoder_config is None, using default FastConformer config.")
        elif isinstance(encoder_config, dict):
            self.encoder_config = FastConformerConfig(**encoder_config)
        elif isinstance(encoder_config, FastConformerConfig):
            self.encoder_config = encoder_config
        else:
            raise ValueError(
                f"encoder_config must be a dict, FastConformerConfig, or None, got {type(encoder_config)}"
            )


__all__ = ["ParakeetCTCConfig"]
