# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..aimv2.modeling_aimv2 import Aimv2Attention, Aimv2EncoderLayer
from ..auto import AutoModel
from ..got_ocr2.image_processing_got_ocr2 import (
    GotOcr2ImageProcessor,
    GotOcr2ImageProcessorKwargs,
    get_optimal_tiled_canvas,
)
from ..got_ocr2.image_processing_pil_got_ocr2 import GotOcr2ImageProcessorPil
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel
from ..llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, LlavaNextModelOutputWithPast
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionEmbeddings
from .configuration_ovis2 import Ovis2Config, Ovis2VisionConfig


def hard_softmax(logits: torch.Tensor, dim: int):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


@auto_docstring
@dataclass
class BaseModelOutputWithVisualIndicatorFeatures(BaseModelOutputWithPooling):
    r"""
    visual_indicator_features (`torch.FloatTensor` of shape `(batch_size, visual_indicator_size)`):
        Visual indicator features extracted from the model, which can be used for auxiliary tasks or further processing.
    """

    visual_indicator_features: torch.FloatTensor | None = None


class Ovis2ModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class Ovis2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class Ovis2RMSNorm(LlamaRMSNorm):
    pass


class Ovis2VisionMLP(LlamaMLP):
    pass


class Ovis2VisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def interpolate_pos_encoding(self):
        raise NotImplementedError("Not needed for Ovis2")

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = self.rms_norm(embeddings)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class Ovis2VisionAttention(Aimv2Attention):
    pass


class Ovis2VisionEncoderLayer(Aimv2EncoderLayer):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.attention = Ovis2VisionAttention(config)


class Ovis2VisionEncoder(SiglipEncoder):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Ovis2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Ovis2VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Ovis2VisionEmbeddings(config)
        self.encoder = Ovis2VisionEncoder(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        pixel_values,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)


class Ovis2PreTrainedModel(PreTrainedModel):
    config: Ovis2Config
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ovis2VisionAttention"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Ovis2VisionEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


class Ovis2VisionModel(Ovis2PreTrainedModel):
    config: Ovis2VisionConfig
    _can_record_outputs = {
        "hidden_states": Ovis2VisionEncoderLayer,
        "attentions": Ovis2VisionAttention,
    }

    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.config = config
        self.transformer = Ovis2VisionTransformer(config)
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        self.vocab_size = config.vocab_size
        self.head_linear = nn.Linear(
            config.hidden_size * config.hidden_stride * config.hidden_stride,
            self.vocab_size - self.num_visual_indicator_tokens,
            bias=False,
        )
        self.head_norm = nn.LayerNorm(self.vocab_size - self.num_visual_indicator_tokens)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithVisualIndicatorFeatures:
        outputs = self.transformer(pixel_values, **kwargs)
        last_hidden_state = outputs[0]
        if self.config.hidden_stride > 1:
            num_images, seq_len, hidden_dim = last_hidden_state.shape
            hidden_stride = self.config.hidden_stride

            sqrt_l = int(math.sqrt(seq_len))
            if sqrt_l * sqrt_l != seq_len:
                raise ValueError("Token sequence length must be a perfect square")

            pad_size = (hidden_stride - (sqrt_l % hidden_stride)) % hidden_stride
            last_hidden_state = nn.functional.pad(last_hidden_state, (0, 0, 0, pad_size, 0, pad_size), "constant", 0)
            sqrt_l += pad_size

            last_hidden_state = last_hidden_state.reshape(
                num_images, sqrt_l // hidden_stride, hidden_stride, sqrt_l // hidden_stride, hidden_stride, hidden_dim
            )
            last_hidden_state = last_hidden_state.permute(0, 1, 3, 2, 4, 5)
            last_hidden_state = last_hidden_state.reshape(
                num_images, -1, hidden_stride * hidden_stride * hidden_dim
            )  # (n, (sqrt_l//hs)^2, hs^2*d)

        logits = self.head_linear(last_hidden_state)
        logits = self.head_norm(logits)

        if self.config.tokenize_function == "gumbel_argmax":
            prob_token = nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
        elif self.config.tokenize_function == "st_argmax":
            prob_token = hard_softmax(logits, dim=-1)
        elif self.config.tokenize_function == "softmax":
            prob_token = nn.functional.softmax(logits, dim=-1)

        return BaseModelOutputWithVisualIndicatorFeatures(
            last_hidden_state=last_hidden_state,
            pooler_output=prob_token,
        )


class Ovis2Model(LlavaModel):
    def __init__(self, config: Ovis2Config):
        super().__init__(config)
        self.vision_tower = Ovis2VisionModel(config.vision_config)
        self.visual_embeddings_table = Ovis2VisualEmbeddingTable(config.vision_config.vocab_size, config.hidden_size)

        self.visual_vocab_size = config.vision_config.vocab_size
        self.vocab_size = config.vocab_size
        self.visual_indicator_token_ids = config.visual_indicator_token_ids
        self.language_model = AutoModel.from_config(config.text_config)
        del self.multi_modal_projector

    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithVisualIndicatorFeatures:
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        image_features = image_outputs.pooler_output
        batch_size, img_seq_len, _ = image_features.shape
        padding_tensor = torch.zeros(
            (batch_size, img_seq_len, self.vision_tower.num_visual_indicator_tokens),
            dtype=image_features.dtype,
            device=image_features.device,
            requires_grad=False,
            layout=image_features.layout,
        )
        image_features = torch.cat([image_features, padding_tensor], dim=2)
        image_features = self.visual_embeddings_table(image_features)

        visual_indicator = torch.arange(
            self.visual_vocab_size - self.vision_tower.num_visual_indicator_tokens,
            self.visual_vocab_size,
            dtype=torch.long,
        ).to(image_features.device)
        image_outputs.pooler_output = image_features
        image_outputs.visual_indicator_features = self.visual_embeddings_table(visual_indicator)

        return image_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs,
    ) -> tuple | Ovis2ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values=pixel_values, return_dict=True)

        if mm_encoder_outputs.get("image") is not None:
            image_features = mm_encoder_outputs["image"].pooler_output
            visual_indicator_features = mm_encoder_outputs["image"].visual_indicator_features

            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            for i, visual_indicator_id in enumerate(self.visual_indicator_token_ids):
                if input_ids is None:
                    mask = inputs_embeds == self.get_input_embeddings()(
                        torch.tensor(visual_indicator_id, dtype=torch.long, device=inputs_embeds.device)
                    )
                    mask = mask.all(-1)
                else:
                    mask = (input_ids == visual_indicator_id).to(inputs_embeds.device)

                if mask.any():
                    inputs_embeds[mask] = visual_indicator_features[i].to(inputs_embeds.device, inputs_embeds.dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Ovis2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if mm_encoder_outputs.get("image") is not None else None,
        )


@auto_docstring
class Ovis2ForConditionalGeneration(LlavaForConditionalGeneration, GenerationMixin):
    def __init__(self, config: Ovis2Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithVisualIndicatorFeatures:
        return self.model.get_image_features(pixel_values=pixel_values, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs,
    ) -> tuple | Ovis2CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, Ovis2ForConditionalGeneration

        >>> model = Ovis2ForConditionalGeneration.from_pretrained("thisisiron/Ovis2-2B-hf")
        >>> processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

        >>> prompt = "<|im_start|>user\n<image>\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n"
        >>> url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        "user\n\nDescribe the image.\nassistant\nThe image features a brown dog standing on a wooden floor, looking up with"
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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

        return Ovis2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


def compute_patch_covering_area(left: int, upper: int, right: int, lower: int, side: int) -> float:
    w = right - left
    h = lower - upper
    w, h = max(w, h), min(w, h)
    if w > side:
        h = h / w * side
        w = side
    return w * h


def split_image_into_grid(h: int, w: int, grid: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    row_height = h // grid[0]
    col_width = w // grid[1]
    return [
        (
            col * col_width,
            row * row_height,
            w if col == grid[1] - 1 else (col + 1) * col_width,
            h if row == grid[0] - 1 else (row + 1) * row_height,
        )
        for row in range(grid[0])
        for col in range(grid[1])
    ]


def get_min_tile_covering_grid(
    image_size: tuple[int, int],
    target_patch_size: int,
    supported_aspect_ratios: list[list[int]],
    covering_threshold: float = 0.9,
) -> tuple[int, int]:
    image_height, image_width = image_size
    image_area = image_width * image_height
    evaluated_grids = []
    sufficient_covering_grids = []

    for tile_grid in supported_aspect_ratios:
        tile_regions = split_image_into_grid(image_height, image_width, tile_grid)
        tile_covering_ratio = (
            sum(compute_patch_covering_area(*region, target_patch_size) for region in tile_regions) / image_area
        )
        evaluated_grids.append((tile_grid, tile_covering_ratio))
        if tile_covering_ratio > covering_threshold:
            sufficient_covering_grids.append((tile_grid, tile_covering_ratio))

    if sufficient_covering_grids:
        return min(sufficient_covering_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0]
    return min(evaluated_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0]


class Ovis2ImageProcessorKwargs(GotOcr2ImageProcessorKwargs, total=False):
    """
    crop_to_patches (`bool`, *optional*, defaults to `False`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to 1):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Only used to build `supported_aspect_ratios`, so it can't be changed in `preprocess`.
    max_patches (`int`, *optional*, defaults to 12):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Only used to build `supported_aspect_ratios`, so it can't be changed in `preprocess`.
    supported_aspect_ratios (`list[list[int]]`, *optional*):
        The (columns, rows) tile grids an image can be cropped to, sorted by number of tiles. Built from
        `min_patches` and `max_patches` at init if not provided.
    use_covering_area_grid (`bool`, *optional*, defaults to `True`):
        Whether to use the covering area grid to determine the number of patches. Only has an effect if
        `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
        `preprocess` method.
    """

    use_covering_area_grid: bool


@auto_docstring
class Ovis2ImageProcessor(GotOcr2ImageProcessor):
    default_to_square = True
    use_covering_area_grid = True
    valid_kwargs = Ovis2ImageProcessorKwargs

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        supported_aspect_ratios: list[list[int]],
        use_covering_area_grid: bool = True,
        covering_threshold: float = 0.9,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the images to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the supported tile grids.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            images (`torch.Tensor`):
                The images to be cropped.
            supported_aspect_ratios (`list[list[int]]`):
                The (columns, rows) tile grids the image can be cropped to.
            use_covering_area_grid (`bool`, *optional*, defaults to `True`):
                Whether to use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90%
                of the image area. If `False`, the closest aspect ratio to the target is used.
            covering_threshold (`float`, *optional*, defaults to `0.9`):
                The threshold for the covering area. Only has an effect if `use_covering_area_grid` is set to `True`.
            patch_size (`SizeDict`, *optional*):
                The size of the output patches.
            resample (`PILImageResampling | int | None`, *optional*):
                Resampling filter to use if resizing the image.

        Returns:
            Tuple[`torch.Tensor`, `list`]: A tuple containing the processed images tensor and the grid information.
        """
        num_image = images.shape[0]
        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = images.shape[-2:]

        if use_covering_area_grid:
            # Use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90% of the image area
            num_columns, num_rows = get_min_tile_covering_grid(
                (original_height, original_width),
                target_patch_size=patch_size_height,  # square patch size
                supported_aspect_ratios=supported_aspect_ratios,
                covering_threshold=covering_threshold,
            )
        else:
            # find the closest aspect ratio to the target
            num_columns, num_rows = get_optimal_tiled_canvas(
                (original_height, original_width), (patch_size_height, patch_size_width), supported_aspect_ratios
            )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(images, SizeDict(height=target_height, width=target_width), resample=resample)
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(images, patch_size, resample=resample)
            processed_images.insert(0, thumbnail_img)

        processed_images = torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()
        grid = [[num_rows, num_columns] for _ in range(num_image)]

        return processed_images, grid

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = False,
        supported_aspect_ratios: list[list[int]] | None = None,
        use_covering_area_grid: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if crop_to_patches and self.max_patches > 1:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            processed_images_grouped = {}
            grids = {}
            for shape, stacked_images in grouped_images.items():
                stacked_images, grid = self.crop_image_to_patches(
                    stacked_images,
                    supported_aspect_ratios,
                    patch_size=size,
                    use_covering_area_grid=use_covering_area_grid,
                    resample=resample,
                )
                processed_images_grouped[shape] = stacked_images
                grids[shape] = grid
            images = reorder_images(processed_images_grouped, grouped_images_index)
            images = [image for images_list in images for image in images_list]
            grids = reorder_images(grids, grouped_images_index)
        else:
            grids = [[1, 1] for _ in range(len(images))]

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images, "grids": grids}, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None):
        raise AttributeError("Not needed for Ovis2")


@auto_docstring
class Ovis2ImageProcessorPil(GotOcr2ImageProcessorPil):
    default_to_square = True
    use_covering_area_grid = True
    valid_kwargs = Ovis2ImageProcessorKwargs

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        supported_aspect_ratios: list[list[int]],
        use_covering_area_grid: bool = True,
        covering_threshold: float = 0.9,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        Mirrors TorchvisionBackend.crop_image_to_patches.
        """
        # Normalize to CHW when called directly (e.g. from tests); _preprocess already receives CHW
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = image.shape[-2:]

        if use_covering_area_grid:
            # Use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90% of the image area
            num_columns, num_rows = get_min_tile_covering_grid(
                (original_height, original_width),
                target_patch_size=patch_size_height,  # square patch size
                supported_aspect_ratios=supported_aspect_ratios,
                covering_threshold=covering_threshold,
            )
        else:
            # find the closest aspect ratio to the target
            num_columns, num_rows = get_optimal_tiled_canvas(
                (original_height, original_width), (patch_size_height, patch_size_width), supported_aspect_ratios
            )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            patch_image = resized_image[:, box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            processed_images.insert(0, thumbnail_img)

        return processed_images, [num_rows, num_columns]

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = False,
        supported_aspect_ratios: list[list[int]] | None = None,
        use_covering_area_grid: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if crop_to_patches and self.max_patches > 1:
            # Crop to patches first
            processed_images = []
            grids = []
            for image in images:
                patches, grid = self.crop_image_to_patches(
                    image,
                    supported_aspect_ratios,
                    patch_size=size,
                    use_covering_area_grid=use_covering_area_grid,
                    resample=resample,
                )
                processed_images.extend(patches)
                grids.append(grid)
            images = processed_images
        else:
            grids = [[1, 1] for _ in range(len(images))]

        # Process all images (including patches if any) through the standard pipeline
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images, "grids": grids}, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None):
        raise AttributeError("Not needed for Ovis2")


__all__ = [
    "Ovis2PreTrainedModel",
    "Ovis2Model",
    "Ovis2ForConditionalGeneration",
    "Ovis2ImageProcessor",
    "Ovis2ImageProcessorPil",
]
