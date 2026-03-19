# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, SizeDict
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import TensorType
from ...utils.import_utils import requires
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2VisionAttention,
    GotOcr2VisionEncoder,
)


@auto_docstring(checkpoint="PaddlePaddle/SLANeXt_wired_safetensors")
class SLANeXtVisionConfig(GotOcr2VisionConfig):
    image_size: int = 512


class SLANeXtVisionAttention(GotOcr2VisionAttention):
    pass


@auto_docstring(
    checkpoint="PaddlePaddle/SLANeXt_wired_safetensors",
    custom_args=r"""
    post_conv_in_channels (`int`, *optional*, defaults to 256):
        Number of input channels for the post-encoder convolution layer.
    post_conv_out_channels (`int`, *optional*, defaults to 512):
        Number of output channels for the post-encoder convolution layer.
    out_channels (`int`, *optional*, defaults to 50):
        Number of output token classes for the structure prediction head (i.e., vocabulary size for table structure
        tokens).
    max_text_length (`int`, *optional*, defaults to 500):
        Maximum number of decoding steps (tokens) for the autoregressive structure and location decoder.
    loc_reg_num (`int`, *optional*, defaults to 8):
        Number of regression values predicted per token for bounding box location (e.g., 8 for four corner
        coordinates).
    """,
)
@strict(accept_kwargs=True)
class SLANeXtConfig(PreTrainedConfig):
    model_type = "slanext"
    sub_configs = {"vision_config": SLANeXtVisionConfig}

    vision_config: dict | SLANeXtVisionConfig | None = None
    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 512
    out_channels: int = 50
    hidden_size: int = 512
    max_text_length: int = 500
    loc_reg_num: int = 8

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = SLANeXtVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = SLANeXtVisionConfig(**self.vision_config)
        super().__post_init__(**kwargs)


class SLANeXtPreTrainedModel(PreTrainedModel):
    """
    Base class for all SLANeXt pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: SLANeXtConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)

        # Initialize positional embeddings to zero (SLANeXtVisionEncoder holds pos_embed)
        if isinstance(module, SLANeXtVisionEncoder):
            if module.pos_embed is not None:
                init.constant_(module.pos_embed, 0.0)

        # Initialize relative positional embeddings to zero (SLANeXtVisionAttention holds rel_pos_h/w)
        if isinstance(module, SLANeXtVisionAttention):
            if module.use_rel_pos:
                init.constant_(module.rel_pos_h, 0.0)
                init.constant_(module.rel_pos_w, 0.0)

        # Initialize GRUCell (replicates PyTorch default reset_parameters)
        if isinstance(module, nn.GRUCell):
            stdv = 1.0 / math.sqrt(module.hidden_size) if module.hidden_size > 0 else 0
            init.uniform_(module.weight_ih, -stdv, stdv)
            init.uniform_(module.weight_hh, -stdv, stdv)
            if module.bias_ih is not None:
                init.uniform_(module.bias_ih, -stdv, stdv)
            if module.bias_hh is not None:
                init.uniform_(module.bias_hh, -stdv, stdv)

        # Initialize SLAHead layers
        if isinstance(module, SLANeXtSLAHead):
            stdv = 1.0 / math.sqrt(self.config.hidden_size * 1.0)
            # Initialize structure_generator and loc_generator layers
            for generator in (module.structure_generator, module.loc_generator):
                for layer in generator.children():
                    if isinstance(layer, nn.Linear):
                        init.uniform_(layer.weight, -stdv, stdv)
                        if layer.bias is not None:
                            init.uniform_(layer.bias, -stdv, stdv)


class SLANeXtVisionEncoder(GotOcr2VisionEncoder):
    pass


class SLANeXtBackbone(nn.Module):
    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        self.vision_config = config.vision_config
        self.post_conv_in_channels = config.post_conv_in_channels
        self.post_conv_out_channels = config.post_conv_out_channels

        self.vision_tower = SLANeXtVisionEncoder(self.vision_config)
        self.post_conv = nn.Conv2d(
            self.post_conv_in_channels, self.post_conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, hidden_states):
        if hidden_states.shape[1] == 1:
            hidden_states = torch.repeat_interleave(hidden_states, repeats=3, dim=1)
        hidden_states = self.vision_tower(hidden_states).last_hidden_state
        hidden_states = self.post_conv(hidden_states)
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)
        return hidden_states


class SLANeXtAttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super().__init__()

        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden, batch_hidden, char_onehots):
        batch_hidden_proj = self.input_to_hidden(batch_hidden)
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden).unsqueeze(1)

        attention_scores = batch_hidden_proj + prev_hidden_proj
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.score(attention_scores)

        alpha = F.softmax(attention_scores, dim=1)
        alpha = alpha.transpose(1, 2)
        context = torch.bmm(alpha, batch_hidden).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return (cur_hidden, cur_hidden), alpha


class SLANeXtStructureMLP(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_channels)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class SLANeXtLocationMLP(nn.Module):
    def __init__(self, hidden_size, loc_reg_num):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, loc_reg_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.sigmoid(hidden_states)
        return hidden_states


class SLANeXtSLAHead(nn.Module):
    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config

        self.structure_attention_cell = SLANeXtAttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels
        )
        self.structure_generator = SLANeXtStructureMLP(config.hidden_size, config.out_channels)
        self.loc_generator = SLANeXtLocationMLP(config.hidden_size, config.loc_reg_num)

    def forward(self, features, targets=None):
        batch_size = features.shape[0]

        hidden = torch.zeros((batch_size, self.config.hidden_size), device=features.device)
        structure_preds_list = []
        structure_ids_list = []
        pre_chars = torch.zeros(size=[batch_size], dtype=torch.long, device=features.device)
        for _ in range(self.config.max_text_length + 1):
            hidden, structure_step, loc_step = self._decode(pre_chars, features, hidden)
            pre_chars = structure_step.argmax(dim=1)
            structure_preds_list.append(structure_step)
            structure_ids_list.append(pre_chars)
            if torch.stack(structure_ids_list, dim=1).eq(self.config.out_channels - 1).any(-1).all():
                break
        structure_preds = F.softmax(torch.stack(structure_preds_list, dim=1), dim=-1)

        return structure_preds

    def _decode(self, pre_chars, features, hidden):
        emb_feature = self._char_to_onehot(pre_chars)
        (output, hidden), alpha = self.structure_attention_cell(hidden, features, emb_feature)

        structure_step = self.structure_generator(output)
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        return F.one_hot(input_char, self.config.out_channels).float()


@auto_docstring
class SLANeXtModel(SLANeXtPreTrainedModel):
    """
    Core SLANeXt model, consisting of Backbone and Head networks.
    Generates structure probs for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = SLANeXtBackbone(config=config)
        self.head = SLANeXtSLAHead(config=config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        backbone_states = self.backbone(pixel_values)
        hidden_states = self.head(backbone_states)
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=backbone_states,
        )


@auto_docstring
@requires(backends=("torch",))
class SLANeXtImageProcessorFast(BaseImageProcessorFast):
    resample = 2  # PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    pad_size = {"height": 512, "width": 512}
    rescale_factor = 1 / 255
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    valid_kwargs = ImagesKwargs
    model_input_names = ["pixel_values"]

    def _resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"] = None,
    ) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        image = image.view(batch_size * channels, height, width)

        device = image.device

        scale = max(size.height, size.width) / max(height, width)
        target_height = round(height * scale)
        target_width = round(width * scale)

        target_x = torch.arange(target_width, dtype=torch.float32, device=device)
        src_x = (target_x + 0.5) * (float(width) / float(target_width)) - 0.5
        src_x_floor = src_x.floor().to(torch.int32)
        src_x_frac = src_x - src_x_floor.float()
        # boundary handling
        src_x_frac = torch.where(src_x_floor < 0, torch.zeros_like(src_x_frac), src_x_frac)
        src_x_floor = torch.where(src_x_floor < 0, torch.zeros_like(src_x_floor), src_x_floor)
        src_x_frac = torch.where(src_x_floor >= width - 1, torch.ones_like(src_x_frac), src_x_frac)
        src_x_floor = torch.where(src_x_floor >= width - 1, torch.full_like(src_x_floor, width - 2), src_x_floor)
        # fixed-point weights
        weight_x_r = (src_x_frac * 2048 + 0.5).floor().to(torch.int32)  # round-to-nearest
        weight_x_l = 2048 - weight_x_r  # (target_w,)
        # --- Y coordinate tables ---
        target_y = torch.arange(target_height, dtype=torch.float32, device=device)
        src_y = (target_y + 0.5) * (float(height) / float(target_height)) - 0.5
        src_y_floor = src_y.floor().to(torch.int32)
        src_y_frac = src_y - src_y_floor.float()
        src_y_frac = torch.where(src_y_floor < 0, torch.zeros_like(src_y_frac), src_y_frac)
        src_y_floor = torch.where(src_y_floor < 0, torch.zeros_like(src_y_floor), src_y_floor)
        src_y_frac = torch.where(src_y_floor >= height - 1, torch.ones_like(src_y_frac), src_y_frac)
        src_y_floor = torch.where(src_y_floor >= height - 1, torch.full_like(src_y_floor, height - 2), src_y_floor)
        weight_y_b = (src_y_frac * 2048 + 0.5).floor().to(torch.int32)
        weight_y_t = 2048 - weight_y_b  # (target_h,)

        img_u8 = image.clamp(0, 255).to(torch.uint8)  # (C, H, W)
        img_i32 = img_u8.to(torch.int32)  # (C, H, W)
        x_left = src_x_floor.long()  # (target_w,)
        x_right = (src_x_floor + 1).long()  # (target_w,)  safe: src_x_floor <= width-2
        y_top = src_y_floor.long()  # (target_h,)
        y_bottom = (src_y_floor + 1).long()  # (target_h,)
        # gather 4 neighbours: (C, target_h, target_w)
        p00 = img_i32[:, y_top[:, None], x_left[None, :]]
        p10 = img_i32[:, y_top[:, None], x_right[None, :]]
        p01 = img_i32[:, y_bottom[:, None], x_left[None, :]]
        p11 = img_i32[:, y_bottom[:, None], x_right[None, :]]
        # fixed-point bilinear: weights broadcast over (C, target_h, target_w)
        weight_y_b_ = weight_y_b.view(1, target_height, 1)
        weight_y_t_ = weight_y_t.view(1, target_height, 1)
        weight_x_r_ = weight_x_r.view(1, 1, target_width)
        weight_x_l_ = weight_x_l.view(1, 1, target_width)
        val = weight_y_t_ * (weight_x_l_ * p00 + weight_x_r_ * p10) + weight_y_b_ * (
            weight_x_l_ * p01 + weight_x_r_ * p11
        )
        val = (val + (1 << 21)) >> 22
        result = val.clamp(0, 255).to(torch.uint8)  # (B*C, target_h, target_w)

        return result.view(batch_size, channels, target_height, target_width).to(dtype=image.dtype)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._resize(image=stacked_images, size=size, interpolation=interpolation)
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

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)
        self.init_decoder()

    def init_decoder(self):
        """
        Initialize the decoder vocabulary for table structure recognition.

        Builds a character dictionary mapping HTML table structure tokens (e.g., `<thead>`, `<tr>`, `<td>`, colspan/
        rowspan attributes) to integer indices. The dictionary includes special `"sos"` (start-of-sequence) and
        `"eos"` (end-of-sequence) tokens. Merged `<td></td>` tokens are used in place of standalone `<td>` tokens
        when applicable.
        """
        dict_character = [
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tr>",
            "</tr>",
            "<td>",
            "<td",
            ">",
            "</td>",
        ]
        dict_character += [f' colspan="{i + 2}"' for i in range(19)]
        dict_character += [f' rowspan="{i + 2}"' for i in range(19)]

        if "<td></td>" not in dict_character:
            dict_character.append("<td></td>")
        if "<td>" in dict_character:
            dict_character.remove("<td>")

        dict_character = ["sos"] + dict_character + ["eos"]
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]
        self.bos_id = self.dict["sos"]
        self.eos_id = self.dict["eos"]

    def post_process_table_recognition(self, outputs):
        """
        Post-process the raw model outputs to decode the predicted table structure into an HTML token sequence.

        Converts the model's predicted probability distributions over the structure vocabulary into a sequence of
        HTML tokens representing the table structure. The decoded tokens are wrapped with `<html>`, `<body>`, and
        `<table>` tags to form a complete HTML table structure.

        Args:
            outputs ([`BaseModelOutputWithNoAttention`]):
                Raw outputs from the SLANeXt model. The `last_hidden_state` field contains the predicted probability
                distributions over the structure vocabulary at each decoding step, with shape
                `(batch_size, max_text_length, num_classes)`.

        Returns:
            `dict`: A dictionary containing:
                - **structure** (`list[str]`): The predicted HTML table structure as a list of tokens, wrapped with
                  `<html>`, `<body>`, and `<table>` tags.
                - **structure_score** (`float`): The mean confidence score across all predicted tokens.
        """
        self.pred = outputs.last_hidden_state
        structure_probs = self.pred[0:1]
        ignored_tokens = [int(self.bos_id), int(self.eos_id)]
        end_idx = int(self.eos_id)

        structure_idx = structure_probs.argmax(dim=2)
        structure_probs = structure_probs.max(dim=2).values

        structure_str_list = []
        batch_size = structure_idx.shape[0]
        for batch_index in range(batch_size):
            structure_list = []
            score_list = []
            for position in range(structure_idx.shape[1]):
                char_idx = int(structure_idx[batch_index, position])
                if position > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                structure_list.append(text)
                score_list.append(structure_probs[batch_index, position])
            structure_str_list.append(structure_list)
            structure_score = torch.stack(score_list).mean().item()

        structure = ["<html>", "<body>", "<table>"] + structure_str_list[0] + ["</table>", "</body>", "</html>"]
        return {"structure": structure, "structure_score": structure_score}


__all__ = [
    "SLANeXtImageProcessorFast",
    "SLANeXtConfig",
    "SLANeXtModel",
    "SLANeXtPreTrainedModel",
]
