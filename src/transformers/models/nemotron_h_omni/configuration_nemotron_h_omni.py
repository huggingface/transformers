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

from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...utils import auto_docstring, logging
from ..nemotron_h import NemotronHConfig
from ..radio.configuration_radio import RadioConfig


__all__ = ["NemotronH_Omni_Reasoning_V3_Config", "SoundConfig"]

logger = logging.get_logger(__name__)


@strict
class SoundConfig(PretrainedConfig):
    """Configuration for the sound/audio model (Parakeet encoder + projection)."""

    model_type = "parakeet"

    # Parakeet encoder config
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    conv_kernel_size: int = 31
    feat_in: int = 80  # Mel features
    subsampling_factor: int = 8
    # Projection config
    projection_hidden_size: int = 20480
    projection_bias: bool = True
    # Audio processing
    sampling_rate: int = 16000


@auto_docstring(checkpoint="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
@strict
class NemotronH_Omni_Reasoning_V3_Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `RadioConfig`, *optional*):
        Configuration for the RADIO vision encoder. Defaults to a default [`RadioConfig`].
    llm_config (`dict` or `NemotronHConfig`, *optional*):
        Configuration for the NemotronH language model. Defaults to a default [`NemotronHConfig`].
    sound_config (`dict` or `SoundConfig`, *optional*):
        Configuration for the optional Parakeet sound encoder. `None` disables the audio branch.
    force_image_size (`int`, *optional*):
        Fixed input image resolution (in pixels) the vision tower expects.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Pixel-shuffle spatial downsample ratio applied to the vision features.
    template (`str`, *optional*):
        Conversation template name (legacy; the chat template is carried by the processor).
    ps_version (`str`, *optional*, defaults to `"v1"`):
        Pixel-shuffle implementation version.
    image_tag_type (`str`, *optional*, defaults to `"internvl"`):
        Image-tag convention used when expanding image placeholders.
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        Hidden size of the vision-to-LLM MLP projector.
    vit_hidden_size (`int`, *optional*, defaults to 1280):
        Hidden size of the RADIO vision features.
    video_pruning_rate (`float`, *optional*, defaults to 0.0):
        Efficient-Video-Sampling token pruning rate; `0.0` disables pruning.
    video_temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of frames collapsed into a single temporal patch by the video embedder.
    patch_size (`int`, *optional*, defaults to 16):
        Vision patch size in pixels.
    img_context_token_id (`int`, *optional*):
        Token id used as the image-context placeholder in `input_ids`.
    img_context_token (`str`, *optional*, defaults to `"<image>"`):
        Textual image-context placeholder token.
    video_context_token_id (`int`, *optional*):
        Token id used as the video-context placeholder in `input_ids`.
    video_context_token (`str`, *optional*, defaults to `"<video>"`):
        Textual video-context placeholder token.
    sound_context_token_id (`int`, *optional*):
        Token id used as the audio-context placeholder in `input_ids`.
    sound_context_token (`str`, *optional*, defaults to `"<audio>"`):
        Textual audio-context placeholder token.
    """

    model_type = "nemotron_h_omni"
    is_composition = True

    vision_config: dict | RadioConfig | None = None
    llm_config: dict | NemotronHConfig | None = None
    sound_config: dict | SoundConfig | None = None
    force_image_size: int | None = None
    downsample_ratio: float = 0.5
    template: str | None = None  # TODO move out of here and into the tokenizer
    ps_version: str = "v1"  # Pixel shuffle version
    image_tag_type: str = "internvl"  # TODO: into the tokenizer too?
    projector_hidden_size: int = 4096
    vit_hidden_size: int = 1280
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    # Vision token settings
    patch_size: int = 16
    img_context_token_id: int | None = None
    img_context_token: str = "<image>"
    video_context_token_id: int | None = None
    video_context_token: str = "<video>"
    # Sound/audio token settings
    sound_context_token_id: int | None = None
    sound_context_token: str = "<audio>"

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = RadioConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = RadioConfig()

        # Handle both cases: when loading from JSON (llm_config is dict) and when called
        # internally by transformers (llm_config is None).
        if isinstance(self.llm_config, dict):
            self.llm_config = NemotronHConfig(**self.llm_config)
        elif self.llm_config is None:
            self.llm_config = NemotronHConfig()

        # Sound/audio model configuration is optional; leave it as `None` to disable the audio branch.
        if isinstance(self.sound_config, dict):
            self.sound_config = SoundConfig(**self.sound_config)

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
