# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig
from ..minicpmv2_6.configuration_minicpmv2_6 import MiniCPMV2_6Config, MiniCPMV2_6VisionConfig
from ..minicpmv2_6.modeling_minicpmv2_6 import (
    MiniCPMV2_6ForConditionalGeneration,
    MiniCPMV2_6Model,
    MiniCPMV2_6PreTrainedModel,
    MiniCPMV2_6VisionTransformer,
)


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4_5")
@strict
class MiniCPMV4_5VisionConfig(MiniCPMV2_6VisionConfig):
    model_type = "minicpmv4_5_vision"


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4_5")
@strict
class MiniCPMV4_5Config(MiniCPMV2_6Config):
    r"""
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of learned latent queries of the resampler, i.e. the number of tokens each image slice becomes.
    resampler_n_heads (`int`, *optional*, defaults to 32):
        Number of attention heads of the resampler's cross-attention.
    image_token_id (`int`, *optional*, defaults to 128244):
        Token id of the image placeholder (`<unk>`), repeated `resampler_n_latents` times per image slice.

    Example:

    ```python
    >>> from transformers import MiniCPMV4_5Config, MiniCPMV4_5ForConditionalGeneration

    >>> configuration = MiniCPMV4_5Config()
    >>> model = MiniCPMV4_5ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmv4_5"
    sub_configs_defaults = {
        "vision_config": MiniCPMV4_5VisionConfig,
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen3",
            init_kwargs={
                "vocab_size": 151748,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "bos_token_id": 151643,
                "eos_token_id": 151645,
            },
        ),
    }

    resampler_n_heads: int = 32


class MiniCPMV4_5PreTrainedModel(MiniCPMV2_6PreTrainedModel):
    pass


class MiniCPMV4_5VisionTransformer(MiniCPMV2_6VisionTransformer):
    pass


class MiniCPMV4_5Model(MiniCPMV2_6Model):
    pass


class MiniCPMV4_5ForConditionalGeneration(MiniCPMV2_6ForConditionalGeneration):
    def forward(**super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.

        Example:

        ```python
        >>> from transformers import AutoModelForImageTextToText, AutoProcessor

        >>> model = AutoModelForImageTextToText.from_pretrained("openbmb/MiniCPM-V-4_5", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-4_5")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        ...             {"type": "text", "text": "What is in this image?"},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> generated = model.generate(**inputs, max_new_tokens=32)
        >>> processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "MiniCPMV4_5Config",
    "MiniCPMV4_5VisionConfig",
    "MiniCPMV4_5PreTrainedModel",
    "MiniCPMV4_5VisionTransformer",
    "MiniCPMV4_5Model",
    "MiniCPMV4_5ForConditionalGeneration",
]
