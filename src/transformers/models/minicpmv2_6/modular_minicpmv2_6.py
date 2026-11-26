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

import torch
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..idefics2.configuration_idefics2 import Idefics2VisionConfig
from ..idefics2.modeling_idefics2 import Idefics2VisionEmbeddings, Idefics2VisionTransformer
from ..minicpmv2_5.configuration_minicpmv2_5 import MiniCPMV2_5Config
from ..minicpmv2_5.modeling_minicpmv2_5 import (
    MiniCPMV2_5ForConditionalGeneration,
    MiniCPMV2_5Model,
    MiniCPMV2_5PreTrainedModel,
)
from ..minicpmv2_5.processing_minicpmv2_5 import MiniCPMV2_5Processor


@auto_docstring(checkpoint="openbmb/MiniCPM-V-2_6")
@strict
class MiniCPMV2_6VisionConfig(Idefics2VisionConfig):
    model_type = "minicpmv2_6_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 980
    patch_size: int | list[int] | tuple[int, int] = 14


@auto_docstring(checkpoint="openbmb/MiniCPM-V-2_6")
@strict
class MiniCPMV2_6Config(MiniCPMV2_5Config):
    r"""
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of learned latent queries of the resampler, i.e. the number of tokens each image slice becomes.
    resampler_n_heads (`int`, *optional*, defaults to 28):
        Number of attention heads of the resampler's cross-attention.
    image_token_id (`int`, *optional*, defaults to 128244):
        Token id of the image placeholder (`<unk>`), repeated `resampler_n_latents` times per image slice.

    Example:

    ```python
    >>> from transformers import MiniCPMV2_6Config, MiniCPMV2_6ForConditionalGeneration

    >>> configuration = MiniCPMV2_6Config()
    >>> model = MiniCPMV2_6ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmv2_6"
    sub_configs_defaults = {
        "vision_config": MiniCPMV2_6VisionConfig,
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen2",
            init_kwargs={
                "vocab_size": 151666,
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "bos_token_id": 151643,
                "eos_token_id": 151645,
            },
        ),
    }

    resampler_n_latents: int = 64
    resampler_n_heads: int = 28
    image_token_id: int = 128244

    def __post_init__(self, **kwargs):
        # Hub configs keep the original flat `rope_scaling` for the remote code; `text_config` has the same values
        kwargs.pop("rope_scaling", None)
        PreTrainedConfig.__post_init__(self, **kwargs)


class MiniCPMV2_6VisionEmbeddings(Idefics2VisionEmbeddings):
    def forward(self, pixel_values: torch.Tensor, patch_position_ids: torch.LongTensor) -> torch.Tensor:
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
        )
        # Bucket each patch by its (row, col) in its slice's 2D patch grid; padding stays at (0, 0)
        step = 1.0 / (patch_position_ids.amax(dim=1, keepdim=True) + 1)
        bucket_coords = torch.bucketize(patch_position_ids * step, boundaries, right=True)
        position_ids = bucket_coords[..., 0] * self.num_patches_per_side + bucket_coords[..., 1]
        return embeddings + self.position_embedding(position_ids)


class MiniCPMV2_6PreTrainedModel(MiniCPMV2_5PreTrainedModel):
    _no_split_modules = ["MiniCPMV2_6EncoderLayer", "MiniCPMV2_6Resampler"]


class MiniCPMV2_6VisionTransformer(Idefics2VisionTransformer):
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor,
        patch_position_ids: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every image slice as a single row of patches, padded to the longest slice.
        patch_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid.
        """
        hidden_states = self.embeddings(pixel_values, patch_position_ids)
        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=hidden_states, attention_mask=patch_attention_mask
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, **kwargs)
        return BaseModelOutputWithPooling(last_hidden_state=self.post_layernorm(encoder_outputs.last_hidden_state))


class MiniCPMV2_6Model(MiniCPMV2_5Model):
    @can_return_tuple
    @auto_docstring(custom_intro="Encodes image slices and resamples each one to `config.resampler_n_latents` tokens.")
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every image slice of the batch as a single row of patches (as the checkpoint was trained), padded to the
            longest slice, as returned by [`MiniCPMV2_5ImageProcessor`].
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid, used by the vision embeddings and the resampler.
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values.to(self.vision_tower.dtype),
            patch_attention_mask=pixel_attention_mask,
            patch_position_ids=patch_position_ids,
            return_dict=True,
            **kwargs,
        )
        vision_outputs.pooler_output = self.resampler(
            vision_outputs.last_hidden_state, pixel_attention_mask, patch_position_ids
        )
        return vision_outputs


class MiniCPMV2_6ForConditionalGeneration(MiniCPMV2_5ForConditionalGeneration):
    def forward(**super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.

        Example:

        ```python
        >>> from transformers import AutoModelForImageTextToText, AutoProcessor

        >>> model = AutoModelForImageTextToText.from_pretrained("openbmb/MiniCPM-V-4", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-4")

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
        ...     messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> generated = model.generate(**inputs, max_new_tokens=32)
        >>> processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0]
        ```"""
        return super().forward(**super_kwargs)


@auto_docstring
class MiniCPMV2_6Processor(MiniCPMV2_5Processor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<unk>",
        num_image_tokens=64,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<unk>"`):
            Marks an image in the text, and is repeated `num_image_tokens` times per image slice.
        num_image_tokens (`int`, *optional*, defaults to 64):
            Number of image tokens per slice (the resampler's number of latents).
        """
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            image_token=image_token,
            num_image_tokens=num_image_tokens,
            **kwargs,
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # Overview image, then each slice in its own <slice>...</slice>, one line per row of slices
        image_tokens = self.image_token * self.num_image_tokens
        num_rows, num_cols = image_inputs["grids"][image_idx]
        rows = [(self.slice_start_token + image_tokens + self.slice_end_token) * num_cols for _ in range(num_rows)]
        return self.image_start_token + image_tokens + self.image_end_token + "\n".join(rows)


__all__ = [
    "MiniCPMV2_6Config",
    "MiniCPMV2_6VisionConfig",
    "MiniCPMV2_6Processor",
    "MiniCPMV2_6PreTrainedModel",
    "MiniCPMV2_6VisionTransformer",
    "MiniCPMV2_6Model",
    "MiniCPMV2_6ForConditionalGeneration",
]
