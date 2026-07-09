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
from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...utils import auto_docstring, logging
from ..nemotron_h import NemotronHConfig
from ..radio.configuration_radio import RadioConfig


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3_Config", "SoundConfig"]

logger = logging.get_logger(__name__)


class SoundConfig(PretrainedConfig):
    """Configuration for the sound/audio model (Parakeet encoder + projection)."""

    model_type = "parakeet"

    def __init__(
        self,
        # Parakeet encoder config
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 24,
        intermediate_size: int = 4096,
        conv_kernel_size: int = 31,
        feat_in: int = 80,  # Mel features
        subsampling_factor: int = 8,
        # Projection config
        projection_hidden_size: int = 20480,
        projection_bias: bool = True,
        # Audio processing
        sampling_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.feat_in = feat_in
        self.subsampling_factor = subsampling_factor
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate


@auto_docstring(checkpoint="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
class NemotronH_Nano_Omni_Reasoning_V3_Config(PreTrainedConfig):
    model_type = "nemotron_h_nano_omni"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        sound_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version="v1",
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        video_pruning_rate: float = 0.0,
        video_temporal_patch_size: int = 2,
        # Vision token settings
        patch_size: int = 16,
        img_context_token_id: int | None = None,
        img_context_token: str = "<image>",
        video_context_token_id: int | None = None,
        video_context_token: str = "<video>",
        # Sound/audio settings
        sound_context_token_id: int | None = None,
        sound_context_token: str = "<audio>",
        **kwargs,
    ):
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
        attn_implementation (`str`, *optional*, defaults to `"flash_attention_2"`):
            Attention implementation to use across the model and its sub-models.
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
        super().__init__(**kwargs)

        if vision_config is not None:
            self.vision_config = RadioConfig(**vision_config)
        else:
            self.vision_config = RadioConfig()

        # Handle both cases: when loading from JSON (llm_config is dict) and when called internally by transformers (llm_config is None)
        if llm_config is not None:
            self.llm_config = NemotronHConfig(**llm_config)
        else:
            self.llm_config = NemotronHConfig()

        # Sound/audio model configuration
        if sound_config is not None:
            self.sound_config = SoundConfig(**sound_config)
        else:
            self.sound_config = None  # Sound model is optional

        # Assign configuration values
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template  # TODO move out of here and into the tokenizer
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type  # TODO: into the tokenizer too?
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.video_pruning_rate = video_pruning_rate
        self.video_temporal_patch_size = video_temporal_patch_size

        # Vision token settings
        self.patch_size = patch_size
        self.img_context_token_id = img_context_token_id
        self.img_context_token = img_context_token
        self.video_context_token_id = video_context_token_id
        self.video_context_token = video_context_token

        # Sound/audio token settings
        self.sound_context_token_id = sound_context_token_id
        self.sound_context_token = sound_context_token

        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = (
            self._attn_implementation is not None and "flash_attention" in self._attn_implementation
        )
        self.llm_config._attn_implementation = self._attn_implementation

    # vLLM's `NemotronH_Nano_VL_V2` implementation reads the language-model sub-config as
    # `config.text_config`. Our HF config stores it as `config.llm_config`; expose an alias so the
    # same config object loads under both loaders without having to duplicate the dict on disk.
    @property
    def text_config(self):
        return self.llm_config
