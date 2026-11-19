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

import math

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple, is_flash_attention_requested
from ..auto import AutoConfig, AutoModel
from ..idefics2.configuration_idefics2 import Idefics2VisionConfig
from ..idefics2.modeling_idefics2 import Idefics2VisionTransformer, eager_attention_forward
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from ..minicpmv4_6.image_processing_minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMV4_6ImageProcessorKwargs,
    ensure_divide,
)
from ..minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil


@auto_docstring(checkpoint="openbmb/MiniCPM-Llama3-V-2_5")
@strict
class MiniCPMV2_5VisionConfig(Idefics2VisionConfig):
    model_type = "minicpmv2_5_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 980
    patch_size: int | list[int] | tuple[int, int] = 14


@auto_docstring(checkpoint="openbmb/MiniCPM-Llama3-V-2_5")
@strict
class MiniCPMV2_5Config(PreTrainedConfig):
    r"""
    query_num (`int`, *optional*, defaults to 96):
        Number of learned queries of the resampler, i.e. the number of tokens each image slice becomes.
    image_token_id (`int`, *optional*, defaults to 128002):
        Token id of the image placeholder (`<unk>`), repeated `query_num` times per image slice.

    Example:

    ```python
    >>> from transformers import MiniCPMV2_5Config, MiniCPMV2_5ForConditionalGeneration

    >>> configuration = MiniCPMV2_5Config()
    >>> model = MiniCPMV2_5ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmv2_5"
    sub_configs_defaults = {
        "vision_config": MiniCPMV2_5VisionConfig,
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="llama",
            init_kwargs={
                "vocab_size": 128256,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
                "bos_token_id": 128000,
                "eos_token_id": 128001,
            },
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    query_num: int = 96
    image_token_id: int = 128002
    tie_word_embeddings: bool = False


def find_best_resize(
    image_size: tuple[float, float], scale_resolution: int, patch_size: int, allow_upscale: bool = False
) -> tuple[int, int]:
    """
    Resize `(height, width)` to roughly `scale_resolution**2` pixels (only when larger, unless `allow_upscale`), keeping
    the aspect ratio, then round each side to a multiple of `patch_size`.
    """
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    return ensure_divide(height, patch_size), ensure_divide(width, patch_size)


def pad_slices(pixel_values: torch.Tensor, target_sizes: torch.Tensor, patch_size: int) -> dict[str, torch.Tensor]:
    """
    Split MiniCPM-V 4.6's packed `(1, num_channels, patch_size, total_patches * patch_size)` strip into one row per
    slice, padded to the longest slice, plus the mask of real patches and each patch's `(row, col)` in its slice grid
    (0 for padding).
    """
    num_patches = (target_sizes[:, 0] * target_sizes[:, 1]).tolist()
    max_patches = max(num_patches)
    slices = pixel_values[0].split([n * patch_size for n in num_patches], dim=-1)
    padded = torch.stack([F.pad(s, (0, (max_patches - n) * patch_size)) for s, n in zip(slices, num_patches)])
    patch_index = torch.arange(max_patches)[None, :]
    patch_attention_mask = patch_index < torch.tensor(num_patches)[:, None]
    widths = target_sizes[:, 1:2].long()
    patch_position_ids = torch.stack([patch_index // widths, patch_index % widths], dim=-1)
    patch_position_ids = patch_position_ids * patch_attention_mask[..., None]
    return {
        "pixel_values": padded,
        "patch_attention_mask": patch_attention_mask,
        "patch_position_ids": patch_position_ids,
    }


class MiniCPMV2_5ImageProcessorKwargs(MiniCPMV4_6ImageProcessorKwargs, total=False):
    pass


class MiniCPMV2_5ImageProcessor(MiniCPMV4_6ImageProcessor):
    # Slicing is MiniCPM-V 4.6's. Differences: no visual-token merger, so sizes are rounded to the patch size only
    # (`downsample_mode` is unused), and slices are returned padded to the same length instead of packed.
    model_input_names = ["pixel_values", "patch_attention_mask", "patch_position_ids"]

    def find_best_resize(self, image_size, scale_resolution, patch_size, downsample_mode=None, allow_upscale=False):
        return find_best_resize(image_size, scale_resolution, patch_size, allow_upscale)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MiniCPMV2_5ImageProcessorKwargs]) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", None)
        outputs = super().preprocess(images, **kwargs)
        target_sizes = outputs["target_sizes"]
        data = pad_slices(outputs["pixel_values"], target_sizes, kwargs.get("patch_size", self.patch_size))
        data.update(
            target_sizes=target_sizes, grids=outputs["grids"], num_patches_per_image=outputs["num_patches_per_image"]
        )
        return BatchFeature(
            data=data, tensor_type=return_tensors, skip_tensor_conversion=["grids", "num_patches_per_image"]
        )


class MiniCPMV2_5ImageProcessorPil(MiniCPMV4_6ImageProcessorPil):
    valid_kwargs = MiniCPMV2_5ImageProcessorKwargs
    model_input_names = ["pixel_values", "patch_attention_mask", "patch_position_ids"]

    def __init__(self, **kwargs: Unpack[MiniCPMV2_5ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def find_best_resize(self, image_size, scale_resolution, patch_size, downsample_mode=None, allow_upscale=False):
        return find_best_resize(image_size, scale_resolution, patch_size, allow_upscale)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MiniCPMV2_5ImageProcessorKwargs]) -> BatchFeature:
        # Same as `pad_slices`, in NumPy so this backend works without PyTorch.
        return_tensors = kwargs.pop("return_tensors", None)
        outputs = super().preprocess(images, **kwargs)
        target_sizes, patch_size = outputs["target_sizes"], kwargs.get("patch_size", self.patch_size)
        num_patches = (target_sizes[:, 0] * target_sizes[:, 1]).tolist()
        max_patches = max(num_patches)
        slices = np.split(outputs["pixel_values"][0], np.cumsum(num_patches)[:-1] * patch_size, axis=-1)
        pad_widths = [((0, 0), (0, 0), (0, (max_patches - n) * patch_size)) for n in num_patches]
        patch_index = np.arange(max_patches)[None, :]
        patch_attention_mask = patch_index < np.array(num_patches)[:, None]
        widths = target_sizes[:, 1:2].astype(np.int64)
        patch_position_ids = np.stack([patch_index // widths, patch_index % widths], axis=-1)
        data = {
            "pixel_values": np.stack([np.pad(s, pad) for s, pad in zip(slices, pad_widths)]),
            "patch_attention_mask": patch_attention_mask,
            "patch_position_ids": patch_position_ids * patch_attention_mask[..., None],
            "target_sizes": target_sizes,
            "grids": outputs["grids"],
            "num_patches_per_image": outputs["num_patches_per_image"],
        }
        return BatchFeature(
            data=data, tensor_type=return_tensors, skip_tensor_conversion=["grids", "num_patches_per_image"]
        )


@auto_docstring
class MiniCPMV2_5Processor(ProcessorMixin):
    image_token = "<unk>"
    image_start_token = "<image>"
    image_end_token = "</image>"
    slice_start_token = "<slice>"
    slice_end_token = "</slice>"
    num_image_tokens = 96

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # Overview image, then (if the image was sliced) all slices row by row inside a single <slice>...</slice>.
        one_slice = self.image_start_token + self.image_token * self.num_image_tokens + self.image_end_token
        placeholder = one_slice
        num_rows, num_cols = image_inputs["grids"][image_idx]
        if num_rows > 0 and num_cols > 0:
            rows = [one_slice * num_cols for _ in range(num_rows)]
            placeholder += self.slice_start_token + "\n".join(rows) + self.slice_end_token
        return placeholder

    @property
    def unused_input_names(self) -> list[str]:
        return ["grids", "num_patches_per_image", "target_sizes"]


class MiniCPMV2_5ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class MiniCPMV2_5CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class MiniCPMV2_5PreTrainedModel(LlavaPreTrainedModel):
    _no_split_modules = ["MiniCPMV2_5EncoderLayer", "MiniCPMV2_5Resampler"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MiniCPMV2_5Resampler):
            init.trunc_normal_(module.query, std=0.02)
            init.normal_(module.proj, std=module.embed_dim**-0.5)


class MiniCPMV2_5VisionTransformer(Idefics2VisionTransformer):
    pass


class MiniCPMV2_5Resampler(nn.Module):
    """
    Perceiver-style resampler: `query_num` learned queries cross-attend to the patch features of one image slice, so
    every slice becomes `query_num` tokens at the language model's width. Keys get a 2D sin-cos position embedding of
    the slice's patch grid; values do not.
    """

    def __init__(self, config: MiniCPMV2_5Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.text_config.hidden_size
        self.num_heads = self.embed_dim // 128
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.query = nn.Parameter(torch.zeros(config.query_num, self.embed_dim))
        self.kv_proj = nn.Linear(config.vision_config.hidden_size, self.embed_dim, bias=False)
        self.ln_q = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = nn.Parameter(torch.zeros(self.embed_dim, self.embed_dim))

    def get_2d_pos_embed(self, patch_position_ids: torch.Tensor) -> torch.Tensor:
        """
        2D sin-cos embedding of `(row, col)` patch positions, shape `(num_slices, max_patches, embed_dim)`. Channels
        are laid out as [sin_w | cos_w | sin_h | cos_h] like the original.
        """
        quarter = self.embed_dim // 4
        omega = torch.arange(quarter, device=patch_position_ids.device, dtype=torch.float32) / quarter
        omega = 1.0 / 10000**omega
        rows = patch_position_ids[..., 0, None].float() * omega
        cols = patch_position_ids[..., 1, None].float() * omega
        return torch.cat([cols.sin(), cols.cos(), rows.sin(), rows.cos()], dim=-1)

    def forward(
        self,
        vision_features: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_slices = vision_features.shape[0]
        pos_embed = self.get_2d_pos_embed(patch_position_ids) * patch_attention_mask[..., None]
        kv = self.ln_kv(self.kv_proj(vision_features))
        queries = self.ln_q(self.query).unsqueeze(0).expand(num_slices, -1, -1)

        query_states = self.q_proj(queries).view(num_slices, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(kv + pos_embed.to(kv.dtype))
        key_states = key_states.view(num_slices, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(kv).view(num_slices, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if is_flash_attention_requested(self.config):
            # Flash attention's padding handling assumes the queries are the last positions of the keys (decoder
            # layout), which doesn't hold for this cross-attention over right-padded slices: use SDPA for this call
            # only, so the vision tower and language model still use flash attention.
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
            attention_mask = patch_attention_mask[:, None, None, :]
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=queries,
                attention_mask=patch_attention_mask,
                encoder_hidden_states=kv,
            )
        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, dropout=0.0
        )
        attn_output = attn_output.reshape(num_slices, -1, self.embed_dim)
        return self.ln_post(self.o_proj(attn_output)) @ self.proj


class MiniCPMV2_5Model(LlavaModel):
    def __init__(self, config: MiniCPMV2_5Config):
        MiniCPMV2_5PreTrainedModel.__init__(self, config)
        self.vision_tower = MiniCPMV2_5VisionTransformer._from_config(config.vision_config)
        self.resampler = MiniCPMV2_5Resampler(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring(custom_intro="Encodes image slices and resamples each one to `config.query_num` tokens.")
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every image slice of the batch as a single row of patches (as the checkpoint was trained), padded to the
            longest slice, as returned by [`MiniCPMV2_5ImageProcessor`].
        patch_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid, used by the resampler's 2D position embedding.
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values.to(self.vision_tower.dtype),
            patch_attention_mask=patch_attention_mask[:, None, :],
            return_dict=True,
            **kwargs,
        )
        vision_outputs.pooler_output = self.resampler(
            vision_outputs.last_hidden_state, patch_attention_mask, patch_position_ids
        )
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        patch_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniCPMV2_5ModelOutputWithPast:
        r"""
        patch_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
                patch_position_ids=patch_position_ids,
                return_dict=True,
            )

        image_features = None
        if mm_encoder_outputs.get("image") is not None:
            image_features = mm_encoder_outputs["image"].pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return MiniCPMV2_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class MiniCPMV2_5ForConditionalGeneration(LlavaForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every image slice of the batch as a single row of patches, padded to the longest slice.
        patch_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            patch_position_ids=patch_position_ids,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        patch_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniCPMV2_5CausalLMOutputWithPast:
        r"""
        patch_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.

        Example:

        ```python
        >>> from transformers import AutoProcessor, MiniCPMV2_5ForConditionalGeneration

        >>> model = MiniCPMV2_5ForConditionalGeneration.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")

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
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            patch_position_ids=patch_position_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs[0], logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        return MiniCPMV2_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "MiniCPMV2_5Config",
    "MiniCPMV2_5VisionConfig",
    "MiniCPMV2_5ImageProcessor",
    "MiniCPMV2_5ImageProcessorPil",
    "MiniCPMV2_5Processor",
    "MiniCPMV2_5PreTrainedModel",
    "MiniCPMV2_5Model",
    "MiniCPMV2_5ForConditionalGeneration",
]
