# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..nemotron_h import NemotronHConfig
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig
from ..radio.configuration_radio import RadioConfig


__all__ = ["NemotronH_Omni_Reasoning_V3_Config"]

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
@strict
class NemotronH_Omni_Reasoning_V3_Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `RadioConfig`, *optional*):
        Configuration for the RADIO vision encoder. Defaults to a default [`RadioConfig`].
    llm_config (`dict` or `NemotronHConfig`, *optional*):
        Configuration for the NemotronH language model. Defaults to a default [`NemotronHConfig`].
    sound_config (`dict` or `ParakeetEncoderConfig`, *optional*):
        Configuration for the optional Parakeet sound encoder. `None` disables the audio branch.
    force_image_size (`int`, *optional*):
        Fixed input image resolution (in pixels) the vision tower expects.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Pixel-shuffle spatial downsample ratio applied to the vision features.
    ps_version (`str`, *optional*, defaults to `"v1"`):
        Pixel-shuffle implementation version (`"v2"` swaps height/width back; `"v1"` is the legacy
        transposed variant).
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        Hidden size of the vision-to-LLM MLP projector.
    vit_hidden_size (`int`, *optional*, defaults to 1280):
        Hidden size of the RADIO vision features.
    video_pruning_rate (`float`, *optional*, defaults to 0.0):
        Efficient-Video-Sampling token pruning rate; `0.0` disables pruning.
    video_temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of frames collapsed into a single temporal patch by the video embedder.
    img_context_token_id (`int`, *optional*):
        Token id used as the image-context placeholder in `input_ids`.
    video_context_token_id (`int`, *optional*):
        Token id used as the video-context placeholder in `input_ids`.
    sound_context_token_id (`int`, *optional*):
        Token id used as the audio-context placeholder in `input_ids`.
    """

    model_type = "nemotron_h_omni"
    is_composition = True

    vision_config: dict | RadioConfig | None = None
    llm_config: dict | NemotronHConfig | None = None
    sound_config: dict | ParakeetEncoderConfig | None = None
    force_image_size: int | None = None
    downsample_ratio: float = 0.5
    ps_version: str = "v1"  # Pixel shuffle version
    projector_hidden_size: int = 4096
    vit_hidden_size: int = 1280
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    # Vision token settings
    img_context_token_id: int | None = None
    video_context_token_id: int | None = None
    # Sound/audio token settings
    sound_context_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = RadioConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = RadioConfig()
        # The vision tower needs the temporal patch size to build its video patch projection.
        self.vision_config.video_temporal_patch_size = self.video_temporal_patch_size

        # Handle both cases: when loading from JSON (llm_config is dict) and when called
        # internally by transformers (llm_config is None).
        if isinstance(self.llm_config, dict):
            self.llm_config = NemotronHConfig(**self.llm_config)
        elif self.llm_config is None:
            self.llm_config = NemotronHConfig()

        # This checkpoint's sound_config omits `attention_bias`/`scale_input`, which are `False` for
        # its Parakeet variant, so supply them before building the encoder config.
        if isinstance(self.sound_config, dict):
            sound_config = {"attention_bias": False, "scale_input": False, **self.sound_config}
            sound_config.pop("model_type", None)
            self.sound_config = ParakeetEncoderConfig(**sound_config)

        super().__post_init__(**kwargs)

        # `attn_implementation` flows in through `**kwargs` (the base `PreTrainedConfig` stores it as
        # `self._attn_implementation`, as for every other model); propagate it to the language model.
        self.llm_config._attn_implementation = self._attn_implementation

    # vLLM's `NemotronH_Nano_VL_V2` implementation reads the language-model sub-config as
    # `config.text_config`. Our HF config stores it as `config.llm_config`; expose an alias so the
    # same config object loads under both loaders without having to duplicate the dict on disk.
    @property
    def text_config(self):
        return self.llm_config
