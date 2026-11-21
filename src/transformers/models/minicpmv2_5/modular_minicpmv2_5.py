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
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig, AutoModel
from ..detr.modeling_detr import DetrCrossAttention
from ..idefics2.configuration_idefics2 import Idefics2VisionConfig
from ..idefics2.modeling_idefics2 import Idefics2VisionTransformer
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel, LlavaPreTrainedModel
from ..minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor, ensure_divide
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
    resampler_n_latents (`int`, *optional*, defaults to 96):
        Number of learned latent queries of the resampler, i.e. the number of tokens each image slice becomes.
    resampler_n_heads (`int`, *optional*, defaults to 32):
        Number of attention heads of the resampler's cross-attention.
    image_token_id (`int`, *optional*, defaults to 128002):
        Token id of the image placeholder (`<unk>`), repeated `resampler_n_latents` times per image slice.

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
    resampler_n_latents: int = 96
    resampler_n_heads: int = 32
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


class MiniCPMV2_5ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    max_slice_nums (`int`, *optional*, defaults to 9):
        Maximum number of slices when splitting a high-resolution image.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split images into multiple slices for higher resolution.
    """

    max_slice_nums: int
    scale_resolution: int
    patch_size: int
    slice_mode: bool


class MiniCPMV2_5ImageProcessor(MiniCPMV4_6ImageProcessor):
    # Slicing is MiniCPM-V 4.6's, but with no visual-token merger sizes are rounded to the patch size only, and each
    # slice is returned as one row of patches, padded to the longest slice.
    downsample_mode = AttributeError()
    use_image_id = AttributeError()
    model_input_names = ["pixel_values", "pixel_attention_mask", "patch_position_ids"]

    def find_best_resize(self, image_size, scale_resolution, patch_size, allow_upscale=False):
        return find_best_resize(image_size, scale_resolution, patch_size, allow_upscale)

    def get_refine_size(
        self,
        image_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        grid_y, grid_x = grid
        best_height, best_width = self.find_best_resize(
            image_size=(ensure_divide(height, grid_y) / grid_y, ensure_divide(width, grid_x) / grid_x),
            scale_resolution=scale_resolution,
            patch_size=patch_size,
            allow_upscale=allow_upscale,
        )
        return best_height * grid_y, best_width * grid_x

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        return_tensors: str | TensorType | None = None,
        disable_grouping: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        slices, slice_sizes, grids = [], [], []
        for image in images:
            image_size = image.shape[-2:]
            best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None

            # The overview image, then (if sliced) the slices of the refined image, row by row
            image_patches = [image]
            source_height, source_width = image_size
            slice_height = slice_width = 0
            if do_resize:
                source_height, source_width = self.find_best_resize(
                    image_size, scale_resolution, patch_size, allow_upscale=best_grid is None
                )
                image_patches = [
                    self.resize(image, size=SizeDict(height=source_height, width=source_width), resample=resample)
                ]
                if best_grid is not None:
                    refine_height, refine_width = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_image = self.resize(
                        image, size=SizeDict(height=refine_height, width=refine_width), resample=resample
                    )
                    slice_height, slice_width = refine_height // best_grid[0], refine_width // best_grid[1]
                    image_patches.extend(divide_to_patches(refine_image, (slice_height, slice_width)))

            grouped_patches, grouped_index = group_images_by_shape(image_patches, disable_grouping=disable_grouping)
            processed_grouped = {
                shape: self.rescale_and_normalize(
                    stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                for shape, stacked in grouped_patches.items()
            }
            processed_patches = reorder_images(processed_grouped, grouped_index)

            slices.extend(self.reshape_by_patch(patch, patch_size) for patch in processed_patches)
            slice_sizes.append((source_height // patch_size, source_width // patch_size))
            slice_sizes.extend(
                [(slice_height // patch_size, slice_width // patch_size)] * (len(processed_patches) - 1)
            )
            grids.append(best_grid if best_grid is not None else [0, 0])

        # Pad every slice to the longest one; the mask marks real patches, the position ids give each one's (row, col)
        num_patches = [height * width for height, width in slice_sizes]
        max_patches = max(num_patches)
        device = slices[0].device
        pixel_values = torch.stack(
            [F.pad(s, (0, (max_patches - n) * patch_size)) for s, n in zip(slices, num_patches)]
        )
        patch_index = torch.arange(max_patches, device=device)[None, :]
        pixel_attention_mask = patch_index < torch.tensor(num_patches, device=device)[:, None]
        widths = torch.tensor([width for _, width in slice_sizes], device=device)[:, None]
        patch_position_ids = torch.stack([patch_index // widths, patch_index % widths], dim=-1)
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "patch_position_ids": patch_position_ids * pixel_attention_mask[..., None],
                "grids": grids,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids"],
        )


class MiniCPMV2_5ImageProcessorPil(MiniCPMV4_6ImageProcessorPil):
    downsample_mode = AttributeError()
    use_image_id = AttributeError()
    valid_kwargs = MiniCPMV2_5ImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_attention_mask", "patch_position_ids"]

    def __init__(self, **kwargs: Unpack[MiniCPMV2_5ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MiniCPMV2_5ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def find_best_resize(self, image_size, scale_resolution, patch_size, allow_upscale=False):
        return find_best_resize(image_size, scale_resolution, patch_size, allow_upscale)

    def get_refine_size(
        self,
        image_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        grid_y, grid_x = grid
        best_height, best_width = self.find_best_resize(
            image_size=(ensure_divide(height, grid_y) / grid_y, ensure_divide(width, grid_x) / grid_x),
            scale_resolution=scale_resolution,
            patch_size=patch_size,
            allow_upscale=allow_upscale,
        )
        return best_height * grid_y, best_width * grid_x

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        slices, slice_sizes, grids = [], [], []
        for image in images:
            image_size = image.shape[-2:]
            best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None

            # The overview image, then (if sliced) the slices of the refined image, row by row
            image_patches = [image]
            source_height, source_width = image_size
            slice_height = slice_width = 0
            if do_resize:
                source_height, source_width = self.find_best_resize(
                    image_size, scale_resolution, patch_size, allow_upscale=best_grid is None
                )
                image_patches = [
                    self.resize(image, size=SizeDict(height=source_height, width=source_width), resample=resample)
                ]
                if best_grid is not None:
                    refine_height, refine_width = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_image = self.resize(
                        image, size=SizeDict(height=refine_height, width=refine_width), resample=resample
                    )
                    slice_height, slice_width = refine_height // best_grid[0], refine_width // best_grid[1]
                    image_patches.extend(divide_to_patches(refine_image, (slice_height, slice_width)))

            for patch in image_patches:
                if do_rescale:
                    patch = self.rescale(patch, rescale_factor)
                if do_normalize:
                    patch = self.normalize(patch, image_mean, image_std)
                slices.append(self.reshape_by_patch(patch, patch_size))
            slice_sizes.append((source_height // patch_size, source_width // patch_size))
            slice_sizes.extend([(slice_height // patch_size, slice_width // patch_size)] * (len(image_patches) - 1))
            grids.append(best_grid if best_grid is not None else [0, 0])

        # Pad every slice to the longest one; the mask marks real patches, the position ids give each one's (row, col)
        num_patches = [height * width for height, width in slice_sizes]
        max_patches = max(num_patches)
        pixel_values = np.stack(
            [np.pad(s, ((0, 0), (0, 0), (0, (max_patches - n) * patch_size))) for s, n in zip(slices, num_patches)]
        )
        patch_index = np.arange(max_patches)[None, :]
        pixel_attention_mask = patch_index < np.array(num_patches)[:, None]
        widths = np.array([width for _, width in slice_sizes])[:, None]
        patch_position_ids = np.stack([patch_index // widths, patch_index % widths], axis=-1)
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "patch_position_ids": patch_position_ids * pixel_attention_mask[..., None],
                "grids": grids,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids"],
        )


@auto_docstring
class MiniCPMV2_5Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token: str = "<unk>",
        image_start_token: str = "<image>",
        image_end_token: str = "</image>",
        slice_start_token: str = "<slice>",
        slice_end_token: str = "</slice>",
        num_image_tokens: int = 96,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<unk>"`):
            Marks an image in the text, and is repeated `num_image_tokens` times per image slice.
        image_start_token (`str`, *optional*, defaults to `"<image>"`):
            Opens the image tokens of one slice.
        image_end_token (`str`, *optional*, defaults to `"</image>"`):
            Closes the image tokens of one slice.
        slice_start_token (`str`, *optional*, defaults to `"<slice>"`):
            Opens the slices of a sliced image.
        slice_end_token (`str`, *optional*, defaults to `"</slice>"`):
            Closes the slices of a sliced image.
        num_image_tokens (`int`, *optional*, defaults to 96):
            Number of image tokens per slice (the resampler's number of latents).
        """
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.slice_start_token = slice_start_token
        self.slice_end_token = slice_end_token
        self.num_image_tokens = num_image_tokens
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
        # Each image's slice grid, which the processor needs to build its placeholder text
        return ["grids"]


class MiniCPMV2_5PreTrainedModel(LlavaPreTrainedModel):
    _no_split_modules = ["MiniCPMV2_5EncoderLayer", "MiniCPMV2_5Resampler"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MiniCPMV2_5Resampler):
            init.trunc_normal_(module.query, std=0.02)
            init.normal_(module.proj, std=module.embed_dim**-0.5)


class MiniCPMV2_5VisionTransformer(Idefics2VisionTransformer):
    pass


class MiniCPMV2_5CrossAttention(DetrCrossAttention):
    """
    Multi-headed cross-attention, as in DETR: keys can get position embeddings, values never do. The resampler adds the
    slice's 2D position embedding to the keys only.
    """

    pass


class MiniCPMV2_5Resampler(nn.Module):
    """
    Perceiver-style resampler: `resampler_n_latents` learned queries cross-attend to the patch features of one image
    slice, so every slice becomes `resampler_n_latents` tokens at the language model's width. Keys get a 2D sin-cos
    position embedding of the slice's patch grid; values do not.
    """

    def __init__(self, config: MiniCPMV2_5Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.text_config.hidden_size
        self.query = nn.Parameter(torch.zeros(config.resampler_n_latents, self.embed_dim))
        self.kv_proj = nn.Linear(config.vision_config.hidden_size, self.embed_dim, bias=False)
        self.ln_q = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MiniCPMV2_5CrossAttention(config, self.embed_dim, config.resampler_n_heads)
        self.ln_post = nn.LayerNorm(self.embed_dim, eps=1e-6)
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
        pixel_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        kv = self.ln_kv(self.kv_proj(vision_features))
        pos_embed = (self.get_2d_pos_embed(patch_position_ids) * pixel_attention_mask[..., None]).to(kv.dtype)
        queries = self.ln_q(self.query).unsqueeze(0).expand(kv.shape[0], -1, -1)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=queries,
            attention_mask=pixel_attention_mask,
            encoder_hidden_states=kv,
        )
        attn_output, _ = self.attn(
            queries, kv, attention_mask=attention_mask, encoder_position_embeddings=pos_embed, **kwargs
        )
        return self.ln_post(attn_output) @ self.proj


class MiniCPMV2_5Model(LlavaModel):
    def __init__(self, config: MiniCPMV2_5Config):
        MiniCPMV2_5PreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.resampler = MiniCPMV2_5Resampler(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

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
            `(row, col)` of each patch in its slice's patch grid, used by the resampler's 2D position embedding.
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values.to(self.vision_tower.dtype),
            patch_attention_mask=pixel_attention_mask[:, None, :],
            return_dict=True,
            **kwargs,
        )
        vision_outputs.pooler_output = self.resampler(
            vision_outputs.last_hidden_state, pixel_attention_mask, patch_position_ids
        )
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
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
                pixel_attention_mask=pixel_attention_mask,
                patch_position_ids=patch_position_ids,
                return_dict=True,
                **kwargs,
            )

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
        return MultimodalBaseModelOutputWithPast(**outputs, image_outputs=mm_encoder_outputs.get("image"))


class MiniCPMV2_5ForConditionalGeneration(LlavaForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every image slice of the batch as a single row of patches, padded to the longest slice.
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
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
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
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
            pixel_attention_mask=pixel_attention_mask,
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
        return MultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
        )


__all__ = [
    "MiniCPMV2_5Config",
    "MiniCPMV2_5VisionConfig",
    "MiniCPMV2_5ImageProcessor",
    "MiniCPMV2_5ImageProcessorPil",
    "MiniCPMV2_5Processor",
    "MiniCPMV2_5PreTrainedModel",
    "MiniCPMV2_5VisionTransformer",
    "MiniCPMV2_5Model",
    "MiniCPMV2_5ForConditionalGeneration",
]
