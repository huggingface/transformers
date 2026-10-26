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
"""PyTorch Xiaomi-Lens-0 model."""

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..sam.modeling_sam import SamLayerNorm
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0TextConfig(Qwen3_5TextConfig):
    model_type = "xiaomi_lens0_text"

    vocab_size: int = 248078
    hidden_size: int = 2560
    intermediate_size: int = 9216
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0VisionConfig(Qwen3_5VisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 2560):
        The output hidden size of the vision model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        The maximum sequence length that this model might ever be used with
    """

    model_type = "xiaomi_lens0_vision"

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    out_hidden_size: int = 2560


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0MaskDecoderConfig(PreTrainedConfig):
    r"""
    vision_hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the vision encoder patch features, taken before the spatial merger.
    upscale_channels (`int`, *optional*, defaults to 256):
        Number of channels the patch features are reduced to before upscaling.
    mask_embed_dim (`int`, *optional*, defaults to 32):
        Dimensionality of the per-pixel embeddings and of the `<|mask|>` token embeddings they are matched against.
    """

    model_type = "xiaomi_lens0_mask_decoder"
    base_config_key = "mask_decoder_config"

    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_attention_heads: int = 16
    vision_hidden_size: int = 1024
    upscale_channels: int = 256
    mask_embed_dim: int = 32
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0Config(Qwen3_5Config):
    r"""
    mask_decoder_config (`Union[XiaomiLens0MaskDecoderConfig, dict]`, *optional*):
        The config object or dictionary of the mask decoder.
    mask_token_id (`int`, *optional*, defaults to 248077):
        The id of the `<|mask|>` token. Every `<|mask|>` token is decoded into a segmentation mask.

    Example:

    ```python
    >>> from transformers import XiaomiLens0ForConditionalGeneration, XiaomiLens0Config

    >>> # Initializing a Xiaomi-Lens-0 style configuration
    >>> configuration = XiaomiLens0Config()

    >>> # Initializing a model from the Xiaomi-Lens-0 style configuration
    >>> model = XiaomiLens0ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    sub_configs_defaults = {
        "vision_config": XiaomiLens0VisionConfig,
        "text_config": XiaomiLens0TextConfig,
        "mask_decoder_config": XiaomiLens0MaskDecoderConfig,
    }

    mask_decoder_config: dict | PreTrainedConfig | None = None
    mask_token_id: int = 248077
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        raise AttributeError("Not needed for Xiaomi-Lens-0")


@auto_docstring
class XiaomiLens0PreTrainedModel(Qwen3_5PreTrainedModel):
    _no_split_modules = ["XiaomiLens0DecoderLayer", "XiaomiLens0VisionBlock", "XiaomiLens0MaskDecoder"]


@auto_docstring
class XiaomiLens0VisionModel(Qwen3_5VisionModel):
    pass


@auto_docstring
class XiaomiLens0TextModel(Qwen3_5TextModel):
    pass


class XiaomiLens0Model(Qwen3_5Model):
    pass


@auto_docstring(
    custom_intro="""
    Base class for Xiaomi-Lens-0 causal language model (or autoregressive) outputs with segmentation masks.
    """
)
@dataclass
class XiaomiLens0CausalLMOutputWithPast(Qwen3_5CausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
        The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`, *optional*):
        Mask logits for every `<|mask|>` token in `input_ids`, in order of appearance. Each mask is predicted at the
        resolution of the (resized) image fed to the vision encoder and placed in the top-left corner of the
        `(height, width)` canvas, which covers the largest image of the batch. Returned when `input_ids` contain
        `<|mask|>` tokens and images are passed.
    """

    pred_masks: torch.FloatTensor | None = None


class XiaomiLens0MaskDecoderAttention(SiglipAttention):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.num_key_value_groups = 1  # needed for eager attention


class XiaomiLens0MaskDecoderMLP(SiglipMLP):
    pass


class XiaomiLens0MaskDecoderLayer(SiglipEncoderLayer):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.self_attn = XiaomiLens0MaskDecoderAttention(config)
        self.mlp = XiaomiLens0MaskDecoderMLP(config)


class XiaomiLens0MaskEmbedding(SiglipMLP):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.mask_embed_dim)


class XiaomiLens0LayerNorm(SamLayerNorm):
    pass


class XiaomiLens0MaskUpscaleBlock(nn.Module):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(config.upscale_channels, config.upscale_channels // 2, kernel_size=2, stride=2)
        self.layer_norm = XiaomiLens0LayerNorm(config.upscale_channels // 2, data_format="channels_first")
        self.activation = ACT2FN[config.hidden_act]
        self.conv2 = nn.ConvTranspose2d(config.upscale_channels // 2, config.mask_embed_dim, kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(self.layer_norm(hidden_states))
        return self.conv2(hidden_states)


class XiaomiLens0PixelDecoder(nn.Module):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__()
        self.patch_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.patch_layer = XiaomiLens0MaskDecoderLayer(config)
        self.channel_reduce = nn.Conv2d(config.hidden_size, config.upscale_channels, kernel_size=1)
        self.upscale_block = XiaomiLens0MaskUpscaleBlock(config)

    def forward(self, image_features: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        batch_size, grid_height, grid_width, _ = image_features.shape
        patch_embeddings = self.patch_proj(image_features).flatten(1, 2)
        patch_embeddings = self.patch_layer(patch_embeddings, attention_mask=None, **kwargs)
        patch_embeddings = patch_embeddings.transpose(1, 2).reshape(batch_size, -1, grid_height, grid_width)

        pixel_embeddings = self.channel_reduce(patch_embeddings)
        return self.upscale_block(pixel_embeddings)


@auto_docstring
class XiaomiLens0MaskDecoder(XiaomiLens0PreTrainedModel):
    """Decodes `<|mask|>` token hidden states into mask logits."""

    config: XiaomiLens0MaskDecoderConfig

    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.mask_embed = XiaomiLens0MaskEmbedding(config)
        self.pixel_decoder = XiaomiLens0PixelDecoder(config)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        mask_hidden_states: torch.Tensor,
        image_features: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        mask_hidden_states (`torch.Tensor` of shape `(batch_size, num_masks, hidden_size)`):
            Last hidden states of the `<|mask|>` tokens.
        image_features (`torch.Tensor` of shape `(batch_size, grid_height, grid_width, vision_hidden_size)`):
            Vision encoder patch features before the spatial merger, in row-major spatial layout.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_masks, 16 * grid_height, 16 * grid_width)`: the mask logits.
        """
        batch_size = image_features.shape[0]
        mask_embeddings = self.mask_embed(mask_hidden_states)

        pixel_embeddings = self.pixel_decoder(image_features, **kwargs)

        height, width = pixel_embeddings.shape[-2:]
        mask_logits = mask_embeddings @ pixel_embeddings.flatten(2) / self.config.mask_embed_dim**0.5
        mask_logits = mask_logits.view(batch_size, -1, height, width)
        # The fixed 4x upsampling brings the masks to the resolution of the image fed to the vision encoder
        return F.interpolate(mask_logits, scale_factor=4, mode="bilinear", align_corners=False)


@auto_docstring
class XiaomiLens0ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mask_decoder = XiaomiLens0MaskDecoder(config.mask_decoder_config)

        self.post_init()

    def get_pred_masks(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        image_hidden_states: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        r"""
        Decodes every `<|mask|>` token of `input_ids` against the closest image preceding it in the same sequence.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden states of the language model.
            image_hidden_states (`torch.FloatTensor` of shape `(num_patches, vision_hidden_size)`):
                Vision encoder patch features before the spatial merger, as returned in the `last_hidden_state` of
                [`~XiaomiLens0Model.get_image_features`].
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`: the mask logits, see
            [`XiaomiLens0CausalLMOutputWithPast`].
        """
        is_image_token = input_ids == self.config.image_token_id
        is_image_start = is_image_token & ~F.pad(is_image_token[:, :-1], (1, 0))
        images_per_sample = is_image_start.sum(-1)
        image_index_in_sample = is_image_start.cumsum(-1) - 1
        image_index = image_index_in_sample + (images_per_sample.cumsum(0) - images_per_sample)[:, None]

        is_mask_token = input_ids == self.config.mask_token_id
        mask_index = is_mask_token.cumsum(-1) - 1
        # `<|mask|>` tokens without a preceding image in their sequence are not decoded
        batch_indices, token_indices = (is_mask_token & (image_index_in_sample >= 0)).nonzero(as_tuple=True)
        mask_image_index = image_index[batch_indices, token_indices]

        merge_size = self.config.vision_config.spatial_merge_size
        patches_per_image = image_grid_thw.prod(-1).tolist()
        image_hidden_states = torch.split(image_hidden_states, patches_per_image)
        decoded_masks = []
        for image_idx in mask_image_index.unique().tolist():
            _, grid_height, grid_width = image_grid_thw[image_idx].tolist()
            # The vision encoder orders patches by spatial merge window, restore the row-major layout
            image_features = (
                image_hidden_states[image_idx]
                .view(grid_height // merge_size, grid_width // merge_size, merge_size, merge_size, -1)
                .permute(0, 2, 1, 3, 4)
                .reshape(1, grid_height, grid_width, -1)
            )
            is_image_mask = mask_image_index == image_idx
            mask_hidden_states = hidden_states[batch_indices[is_image_mask], token_indices[is_image_mask]]
            masks = self.mask_decoder(mask_hidden_states[None], image_features.to(hidden_states.dtype), **kwargs)
            decoded_masks.append((is_image_mask, masks[0]))

        canvas_height, canvas_width = (
            image_grid_thw[:, 1:].max(0).values * self.config.vision_config.patch_size
        ).tolist()
        pred_masks = hidden_states.new_zeros(
            input_ids.shape[0], is_mask_token.sum(-1).max(), canvas_height, canvas_width
        )
        for is_image_mask, masks in decoded_masks:
            mask_batch_indices = batch_indices[is_image_mask]
            mask_slot_indices = mask_index[mask_batch_indices, token_indices[is_image_mask]]
            pred_masks[mask_batch_indices, mask_slot_indices, : masks.shape[-2], : masks.shape[-1]] = masks
        return pred_masks

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | XiaomiLens0CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, XiaomiLens0ForConditionalGeneration

        >>> model = XiaomiLens0ForConditionalGeneration.from_pretrained("SeerRay-Lab/Xiaomi-Lens-0", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("SeerRay-Lab/Xiaomi-Lens-0")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        ...             {
        ...                 "type": "text",
        ...                 "text": 'Locate all instances in the image that match the description "cat".\nReport bbox coordinates and masks.',
        ...             },
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)

        >>> # Generate the answer, then decode one mask per generated `<|mask|>` token
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> completion = processor.decode(generated_ids[0, inputs.input_ids.shape[1] :]).removesuffix("<|im_end|>")
        >>> messages.append({"role": "assistant", "content": completion})
        >>> mask_inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> outputs = model(**mask_inputs)
        >>> results = processor.post_process_grounding(
        ...     mask_inputs.input_ids, outputs.pred_masks, mask_inputs.image_grid_thw
        ... )
        ```
        """
        image_features = mm_encoder_outputs.get("image") if mm_encoder_outputs is not None else None
        decode_masks = (
            input_ids is not None
            and image_grid_thw is not None
            and (pixel_values is not None or image_features is not None)
            and (input_ids == self.config.mask_token_id).any()
        )
        if decode_masks and image_features is None:
            # The mask decoder needs the patch features before the spatial merger, compute them once here
            image_features = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
            mm_encoder_outputs = {**(mm_encoder_outputs or {}), "image": image_features}
            pixel_values = None

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(slice_logits_to_keep(hidden_states, logits_to_keep))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        pred_masks = None
        if decode_masks:
            pred_masks = self.get_pred_masks(
                input_ids, hidden_states, image_features.last_hidden_state, image_grid_thw
            )

        return XiaomiLens0CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            pred_masks=pred_masks,
        )


class XiaomiLens0Processor(Qwen3VLProcessor):
    def post_process_grounding(
        self,
        input_ids: torch.LongTensor,
        pred_masks: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        target_sizes: list[tuple[int, int]] | None = None,
        mask_threshold: float = 0.5,
    ) -> list[dict[str, torch.Tensor]]:
        r"""
        Extracts the boxes and masks of the final answer of [`XiaomiLens0ForConditionalGeneration`]. Each grounded
        instance is written as `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`, with box coordinates normalized to
        `[0, 1000]`. Masks decoded from `<|mask|>` tokens inside the thinking block are discarded. Expects one image per
        sample.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The full sequences (prompt and generated tokens) the masks were decoded from.
            pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
                The `pred_masks` returned by [`XiaomiLens0ForConditionalGeneration`].
            image_grid_thw (`torch.LongTensor` of shape `(batch_size, 3)`):
                The image grids returned by the processor.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                The `(height, width)` of each image to resize the boxes and masks to. Defaults to the size of the
                resized image fed to the vision encoder.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Probability threshold to binarize the masks.

        Returns:
            `list[dict]`: One dictionary per sample with:
                - **boxes** (`torch.FloatTensor` of shape `(num_boxes, 4)`): Boxes in absolute
                  `(top_left_x, top_left_y, bottom_right_x, bottom_right_y)` format.
                - **masks** (`torch.BoolTensor` of shape `(num_masks, height, width)`): Binary masks.
        """
        mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        box_pattern = re.compile(r"<bbox>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</bbox>")
        patch_size = self.image_processor.patch_size

        results = []
        for sample_idx, sample_ids in enumerate(input_ids.tolist()):
            answer_start = 0
            if think_end_token_id in sample_ids:
                answer_start = len(sample_ids) - sample_ids[::-1].index(think_end_token_id)
            num_thinking_masks = sample_ids[:answer_start].count(mask_token_id)
            num_masks = sample_ids.count(mask_token_id)

            _, grid_height, grid_width = image_grid_thw[sample_idx].tolist()
            canvas_height, canvas_width = grid_height * patch_size, grid_width * patch_size
            height, width = target_sizes[sample_idx] if target_sizes is not None else (canvas_height, canvas_width)

            mask_logits = pred_masks[sample_idx, num_thinking_masks:num_masks, :canvas_height, :canvas_width]
            if (height, width) != (canvas_height, canvas_width):
                mask_logits = F.interpolate(
                    mask_logits[None].float(), size=(height, width), mode="bilinear", align_corners=False
                )[0]
            masks = mask_logits.sigmoid() >= mask_threshold

            answer = self.tokenizer.decode(sample_ids[answer_start:], skip_special_tokens=False)
            boxes = torch.tensor(
                [[int(coordinate) for coordinate in match] for match in box_pattern.findall(answer)],
                dtype=torch.float32,
            ).reshape(-1, 4)
            boxes = boxes / 1000 * torch.tensor([width, height, width, height])
            results.append({"boxes": boxes, "masks": masks})
        return results


__all__ = [
    "XiaomiLens0Config",
    "XiaomiLens0TextConfig",
    "XiaomiLens0VisionConfig",
    "XiaomiLens0MaskDecoderConfig",
    "XiaomiLens0VisionModel",
    "XiaomiLens0TextModel",
    "XiaomiLens0Model",
    "XiaomiLens0ForConditionalGeneration",
    "XiaomiLens0PreTrainedModel",
    "XiaomiLens0Processor",
]
