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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


__all__ = ["NemotronH_Omni_Reasoning_V3_Config"]

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
@strict
class NemotronH_Omni_Reasoning_V3_Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `RadioConfig`, *optional*):
        Configuration for the RADIO vision encoder. Defaults to a default [`RadioConfig`].
    text_config (`dict` or `NemotronHConfig`, *optional*):
        Configuration for the NemotronH language model. Defaults to a default [`NemotronHConfig`].
    audio_config (`dict` or `ParakeetEncoderConfig`, *optional*):
        Configuration for the optional Parakeet sound encoder. `None` disables the audio branch.
    force_image_size (`int`, *optional*):
        Fixed input image resolution (in pixels) the vision tower expects.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Pixel-shuffle spatial downsample ratio applied to the vision features.
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        Hidden size of the vision-to-LLM MLP projector.
    vision_hidden_size (`int`, *optional*, defaults to 1280):
        Hidden size of the RADIO vision features.
    video_pruning_rate (`float`, *optional*, defaults to 0.0):
        Efficient-Video-Sampling token pruning rate; `0.0` disables pruning.
    video_temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of frames collapsed into a single temporal patch by the video embedder.
    image_token_id (`int`, *optional*):
        Token id used as the image-context placeholder in `input_ids`.
    video_token_id (`int`, *optional*):
        Token id used as the video-context placeholder in `input_ids`.
    audio_token_id (`int`, *optional*):
        Token id used as the audio-context placeholder in `input_ids`.
    """

    model_type = "nemotron_h_omni"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(
            config_class=AutoConfig, model_type="radio", init_kwargs={"video_temporal_patch_size": 2}
        ),
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="nemotron_h"),
        "audio_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="parakeet_encoder",
            init_kwargs={"attention_bias": False, "scale_input": False},
            optional=True,
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    force_image_size: int | None = None
    downsample_ratio: float = 0.5
    projector_hidden_size: int = 4096
    vision_hidden_size: int = 1280
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    # Vision token settings
    image_token_id: int | None = None
    video_token_id: int | None = None
    # Sound/audio token settings
    audio_token_id: int | None = None

    # Released checkpoints predate the rename to the standard field names and store these values
    # under the names below; map them across so those configs keep loading.
    _legacy_field_names = {
        "img_context_token_id": "image_token_id",
        "video_context_token_id": "video_token_id",
        "sound_context_token_id": "audio_token_id",
        "llm_config": "text_config",
        "sound_config": "audio_config",
        "vit_hidden_size": "vision_hidden_size",
    }

    def __post_init__(self, **kwargs):
        for legacy_name, name in self._legacy_field_names.items():
            legacy_value = kwargs.pop(legacy_name, None)
            # only fall back to the legacy value when the current field is still at its default,
            # so a config carrying both names keeps the new one
            if legacy_value is not None and getattr(self, name) == getattr(type(self), name, None):
                setattr(self, name, legacy_value)

        # Backwards compatibility: released checkpoints omit `attention_bias`/`scale_input` from
        # `audio_config`, and their Parakeet variant expects `False` for both, whereas
        # `ParakeetEncoderConfig` defaults them to `True`. Supply them for configs that predate the
        # fields; an explicit value in the checkpoint still wins.
        if isinstance(self.audio_config, dict):
            audio_config = dict(self.audio_config)  # copy: the caller may reuse the dict
            # the checkpoint stores `model_type: parakeet`, which is not a registered type; let the
            # sub-config class supply its own
            audio_config.pop("model_type", None)
            self.audio_config = {"attention_bias": False, "scale_input": False, **self.audio_config}

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        # The vision tower builds its video patch projection from its own copy of this value, so a
        # mismatch would silently give the two towers different temporal packing.
        if self.vision_config.video_temporal_patch_size != self.video_temporal_patch_size:
            raise ValueError(
                f"`vision_config.video_temporal_patch_size` "
                f"({self.vision_config.video_temporal_patch_size}) and `video_temporal_patch_size` "
                f"({self.video_temporal_patch_size}) must be equal."
            )
