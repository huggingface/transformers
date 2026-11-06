# Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import torch

from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPooling
from ...utils import auto_docstring


@auto_docstring
@dataclass
class XiaomiLens0GenerateOutput(GenerateDecoderOnlyOutput):
    r"""
    scores (`tuple[torch.FloatTensor]`, *optional*):
        Processed prediction scores, one tensor per generated token. Returned when `output_scores=True`.
    logits (`tuple[torch.FloatTensor]`, *optional*):
        Unprocessed language-model logits, one tensor per generated token. Returned when `output_logits=True`.
    image_hidden_states (`torch.FloatTensor` of shape `(num_patches, vision_hidden_size)`, *optional*):
        Vision patch features before spatial merging, in original input-image order.
    """

    image_hidden_states: torch.FloatTensor | None = None


class XiaomiLens0GenerationMixin(GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        generation_config: GenerationConfig | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs,
    ):
        # Custom generate to return image_hidden_states for mask decoding.
        if (
            pixel_values is not None or kwargs.get("pixel_values_videos") is not None
        ) and mm_encoder_outputs is not None:
            raise ValueError("You cannot pass both: raw pixels and pre-computed embeddings")

        mm_encoder_outputs = {} if mm_encoder_outputs is None else dict(mm_encoder_outputs)
        if pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, kwargs.get("image_grid_thw"), return_dict=True
            )

        image_outputs = mm_encoder_outputs.get("image")
        image_hidden_states = None if image_outputs is None else image_outputs.last_hidden_state

        result = super().generate(
            inputs, generation_config=generation_config, mm_encoder_outputs=mm_encoder_outputs, **kwargs
        )

        if isinstance(result, GenerateDecoderOnlyOutput):
            return XiaomiLens0GenerateOutput(**result, image_hidden_states=image_hidden_states)
        return result


__all__ = ["XiaomiLens0GenerateOutput"]
