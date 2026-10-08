# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    audio_config (`Union[dict, PreTrainedConfig]`, *optional*):
        The config object or dictionary of the audio encoder. Defaults to [`OmniASRAudioConfig`].
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`OmniASRCTCForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`OmniASRCTCForCTC`].

    Example:

    ```python
    >>> from transformers import OmniASRCTCForCTC, OmniASRCTCConfig

    >>> # Initializing an OmniASR-CTC configuration
    >>> configuration = OmniASRCTCConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRCTCForCTC(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_ctc"
    sub_configs_defaults = {
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="omniasr_audio"),
    }

    vocab_size: int = 10288
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    initializer_range: float = 0.02
    audio_config: dict | PreTrainedConfig | None = None
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.initializer_range != self.audio_config.initializer_range:
            raise ValueError(
                f"`initializer_range` ({self.initializer_range}) must match `audio_config.initializer_range` "
                f"({self.audio_config.initializer_range})."
            )


__all__ = ["OmniASRCTCConfig"]
