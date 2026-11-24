# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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

from typing import Any

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_multimodal_utils import merge_mm_encoder_outputs
from ...modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithSplitSizes
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import accepts_precomputed_kwargs, can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import select_hidden_states
from ..aria.modeling_aria import AriaCausalLMOutputWithPast, AriaModelOutputWithPast
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaPreTrainedModel,
    TransformersKwargs,
)
from ..mistral.modeling_mistral import MistralRMSNorm
from .configuration_mistral3 import Mistral3Config


logger = logging.get_logger(__name__)


class Mistral3RMSNorm(MistralRMSNorm):
    pass


def get_merge_index(
    image_sizes: torch.Tensor | list, patch_size: int, merge_size: int, kwargs: dict | None = None
) -> torch.Tensor:
    """For each merged token, the indices of its `merge_size ** 2` patches in the packed sequence, block by block in
    row-major order (an image's last row or column of patches is dropped when it doesn't fill a block), or pop
    `"merge_index"` from `kwargs` if precomputed."""
    if kwargs is not None and (merge_index := kwargs.pop("merge_index", None)) is not None:
        return merge_index
    device = image_sizes.device if isinstance(image_sizes, torch.Tensor) else None
    block = torch.arange(merge_size, device=device)
    merge_index, offset = [], 0
    for height, width in (torch.as_tensor(image_sizes) // patch_size).tolist():
        rows = torch.arange(height // merge_size, device=device)[:, None] * merge_size
        columns = torch.arange(width // merge_size, device=device)[None, :] * merge_size
        within = (block[:, None] * width + block[None, :]).flatten()
        merge_index.append((rows * width + columns).reshape(-1, 1) + within + offset)
        offset += height * width
    return torch.cat(merge_index)


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor, **kwargs) -> torch.Tensor:
        merge_index = get_merge_index(image_sizes, self.patch_size, self.spatial_merge_size, kwargs=kwargs)
        # Each merged token concatenates its patches channel-major, as `unfold` lays them out
        merged_size = image_features.shape[-1] * merge_index.shape[-1]
        image_features = image_features[merge_index].transpose(1, 2).reshape(-1, merged_size)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.norm = Mistral3RMSNorm(config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        self.num_feature_layers = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * self.num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor, **kwargs):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes, **kwargs)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Mistral3CausalLMOutputWithPast(AriaCausalLMOutputWithPast):
    pass


class Mistral3ModelOutputWithPast(AriaModelOutputWithPast):
    pass


class Mistral3PreTrainedModel(LlavaPreTrainedModel):
    pass


class Mistral3Model(LlavaModel):
    def get_data_dependent_inputs(
        self, pixel_values: torch.Tensor | None = None, image_sizes: torch.Tensor | None = None, **kwargs
    ) -> dict[str, Any]:
        """What the forward derives from the processor's outputs with data-dependent ops: the vision
        tower's (see [`PixtralVisionModel.get_data_dependent_inputs`]) and the patch merger's `merge_index`,
        prefixed `image_`, as `get_image_embeddings` takes them."""
        if pixel_values is None or image_sizes is None:
            return {}
        inputs = self.vision_tower.get_data_dependent_inputs(pixel_values, image_sizes)
        inputs["merge_index"] = get_merge_index(
            image_sizes, self.config.vision_config.patch_size, self.config.spatial_merge_size
        )
        return {f"image_{name}": value for name, value in inputs.items()}

    @accepts_precomputed_kwargs(modality="image")
    @merge_with_config_defaults
    @can_return_tuple
    @select_hidden_states("vision_feature_layer")
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_embeddings(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | list[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        image_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            return_dict=True,
            **kwargs,
        )
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_outputs.pooler_output = self.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes, **kwargs
        )
        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        image_outputs = BaseModelOutputWithSplitSizes(
            **image_outputs,
            split_sizes=(
                torch.as_tensor(image_sizes, device=image_outputs.pooler_output.device) // downsample_ratio
            ).prod(dim=-1),
        )
        return image_outputs

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    @merge_mm_encoder_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_feature_layer: int | list[int] | list[int] | None = None,
        use_cache: bool | None = None,
        image_sizes: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Mistral3ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_embeddings(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes,
                return_dict=True,
                **kwargs,
            )

        if mm_encoder_outputs.get("image") is not None:
            image_features = mm_encoder_outputs["image"].pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Mistral3ModelOutputWithPast(**outputs, image_outputs=mm_encoder_outputs.get("image"))


class Mistral3ForConditionalGeneration(LlavaForConditionalGeneration):
    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Mistral3CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        >>> model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        >>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

        >>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is the image?The image depicts two cats lying on a pink blanket."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_sizes=image_sizes,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Mistral3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
        )


__all__ = [
    "Mistral3Model",
    "Mistral3PreTrainedModel",
    "Mistral3ForConditionalGeneration",
]
