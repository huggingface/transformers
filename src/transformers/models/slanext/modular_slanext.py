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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2CLS
from ...backbone_utils import filter_output_hidden_states
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, SizeDict
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import TensorType, merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2VisionAttention,
    GotOcr2VisionEncoder,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/SLANeXt_wired_safetensors")
@strict
class SLANeXtVisionConfig(GotOcr2VisionConfig):
    image_size: int = 512


class SLANeXtVisionAttention(GotOcr2VisionAttention):
    pass


@auto_docstring(checkpoint="PaddlePaddle/SLANeXt_wired_safetensors")
@strict
class SLANeXtConfig(PreTrainedConfig):
    r"""
    vision_config (`dict` or [`SLANeXtVisionConfig`], *optional*):
        Configuration for the vision encoder. If `None`, a default [`SLANeXtVisionConfig`] is used.
    post_conv_in_channels (`int`, *optional*, defaults to 256):
        Number of input channels for the post-encoder convolution layer.
    post_conv_out_channels (`int`, *optional*, defaults to 512):
        Number of output channels for the post-encoder convolution layer.
    out_channels (`int`, *optional*, defaults to 50):
        Vocabulary size for the table structure token prediction head, i.e., the number of distinct structure
        tokens the model can predict.
    hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the hidden states in the attention GRU cell and the structure/location prediction heads.
    max_text_length (`int`, *optional*, defaults to 500):
        Maximum number of autoregressive decoding steps (tokens) for the structure and location decoder.
    """

    model_type = "slanext"
    sub_configs = {"vision_config": SLANeXtVisionConfig}

    vision_config: dict | SLANeXtVisionConfig | None = None
    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 512
    out_channels: int = 50
    hidden_size: int = 512
    max_text_length: int = 500

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = SLANeXtVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = SLANeXtVisionConfig(**self.vision_config)
        super().__post_init__(**kwargs)


class SLANeXtAttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()

        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def forward(
        self,
        prev_hidden: torch.FloatTensor,
        batch_hidden: torch.FloatTensor,
        char_onehots: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_hidden_proj = self.input_to_hidden(batch_hidden)
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden).unsqueeze(1)

        attention_scores = batch_hidden_proj + prev_hidden_proj
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.score(attention_scores)

        attn_weights = F.softmax(attention_scores, dim=1, dtype=torch.float32).to(attention_scores.dtype)
        attn_weights = attn_weights.transpose(1, 2)
        context = torch.matmul(attn_weights, batch_hidden).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)
        hidden_states = self.rnn(concat_context, prev_hidden)

        return hidden_states, attn_weights


class SLANeXtMLP(nn.Module):
    def __init__(self, hidden_size, out_channels, activation=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)
        self.act_fn = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class SLANeXtPreTrainedModel(PreTrainedModel):
    config: SLANeXtConfig
    base_model_prefix = "backbone"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules_strict = ["structure_attention_cell", "structure_generator"]

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
            std = 1.0 / math.sqrt(module.hidden_size) if module.hidden_size > 0 else 0
            init.uniform_(module.weight_ih, -std, std)
            init.uniform_(module.weight_hh, -std, std)
            if module.bias_ih is not None:
                init.uniform_(module.bias_ih, -std, std)
            if module.bias_hh is not None:
                init.uniform_(module.bias_hh, -std, std)

        # Initialize SLAHead layers
        if isinstance(module, SLANeXtSLAHead):
            std = 1.0 / math.sqrt(self.config.hidden_size * 1.0)
            # Initialize structure_generator and loc_generator layers
            for generator in (module.structure_generator,):
                for layer in generator.children():
                    if isinstance(layer, nn.Linear):
                        init.uniform_(layer.weight, -std, std)
                        if layer.bias is not None:
                            init.uniform_(layer.bias, -std, std)


class SLANeXtVisionEncoder(GotOcr2VisionEncoder):
    pass


class SLANeXtBackbone(SLANeXtPreTrainedModel):
    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config)
        self.vision_tower = SLANeXtVisionEncoder(config.vision_config)
        self.post_conv = nn.Conv2d(
            config.post_conv_in_channels, config.post_conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.post_init()

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]):
        vision_output = self.vision_tower(hidden_states, **kwargs)
        hidden_states = self.post_conv(vision_output.last_hidden_state)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=vision_output.hidden_states,
            attentions=vision_output.attentions,
        )


class SLANeXtSLAHead(SLANeXtPreTrainedModel):
    _can_record_outputs = {
        "attentions": SLANeXtAttentionGRUCell,
    }

    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config)

        self.structure_attention_cell = SLANeXtAttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels
        )
        self.structure_generator = SLANeXtMLP(config.hidden_size, config.out_channels)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @filter_output_hidden_states
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        targets: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        features = torch.zeros(
            (hidden_states.shape[0], self.config.hidden_size), dtype=torch.float32, device=hidden_states.device
        )
        predicted_chars = torch.zeros(size=[hidden_states.shape[0]], dtype=torch.long, device=hidden_states.device)

        structure_preds_list = []
        structure_ids_list = []
        for _ in range(self.config.max_text_length + 1):
            embedding_feature = F.one_hot(predicted_chars, self.config.out_channels).float()
            features, _ = self.structure_attention_cell(features, hidden_states.float(), embedding_feature)
            structure_step = self.structure_generator(features)
            predicted_chars = structure_step.argmax(dim=1)

            structure_preds_list.append(structure_step)
            structure_ids_list.append(predicted_chars)
            if torch.stack(structure_ids_list, dim=1).eq(self.config.out_channels - 1).any(-1).all():
                break
        structure_preds = F.softmax(torch.stack(structure_preds_list, dim=1), dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )

        return BaseModelOutput(last_hidden_state=structure_preds, hidden_states=structure_preds_list)


@dataclass
@auto_docstring
class SLANeXtForTableRecognitionOutput(BaseModelOutput):
    r"""
    head_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Hidden-states of the SLANeXtSLAHead at each prediction step, varies up to max `self.config.max_text_length` states (depending on early exits).
    head_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Attentions of the SLANeXtSLAHead at each prediction step, varies up to max `self.config.max_text_length` attentions (depending on early exits).
    """

    head_hidden_states: torch.FloatTensor | None = None
    head_attentions: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    SLANeXt Table Recognition model for table recognition tasks. Wraps the core SLANeXtPreTrainedModel
    and returns outputs compatible with the Transformers table recognition API.
    """
)
class SLANeXtForTableRecognition(SLANeXtPreTrainedModel):
    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = SLANeXtBackbone(config=config)
        self.head = SLANeXtSLAHead(config=config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | SLANeXtForTableRecognitionOutput:
        backbone_outputs = self.backbone(pixel_values, **kwargs)
        head_outputs = self.head(backbone_outputs.last_hidden_state, **kwargs)
        return SLANeXtForTableRecognitionOutput(
            last_hidden_state=head_outputs.last_hidden_state,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            head_hidden_states=head_outputs.hidden_states,
            head_attentions=head_outputs.attentions,
        )


@auto_docstring
@requires(backends=("torch",))
class SLANeXtImageProcessor(TorchvisionBackend):
    resample = 2  # PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    pad_size = {"height": 512, "width": 512}
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True

    def _resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        batch_size, channels, height, width = image.shape
        image = image.view(batch_size * channels, height, width)

        device = image.device

        scale = max(size.height, size.width) / max(height, width)
        target_height = round(height * scale)
        target_width = round(width * scale)

        target_col = torch.arange(target_width, dtype=torch.float32, device=device)
        src_col = (target_col + 0.5) * (float(width) / float(target_width)) - 0.5
        src_col_floor = src_col.floor().to(torch.int32)
        src_col_frac = src_col - src_col_floor.float()
        # boundary handling
        src_col_frac = torch.where(src_col_floor < 0, torch.zeros_like(src_col_frac), src_col_frac)
        src_col_floor = torch.where(src_col_floor < 0, torch.zeros_like(src_col_floor), src_col_floor)
        src_col_frac = torch.where(src_col_floor >= width - 1, torch.ones_like(src_col_frac), src_col_frac)
        src_col_floor = torch.where(
            src_col_floor >= width - 1, torch.full_like(src_col_floor, width - 2), src_col_floor
        )
        # fixed-point weights
        weight_right = (src_col_frac * 2048 + 0.5).floor().to(torch.int32)  # round-to-nearest
        weight_left = 2048 - weight_right  # (target_w,)
        # --- row coordinate tables ---
        target_row = torch.arange(target_height, dtype=torch.float32, device=device)
        src_row = (target_row + 0.5) * (float(height) / float(target_height)) - 0.5
        src_row_floor = src_row.floor().to(torch.int32)
        src_row_frac = src_row - src_row_floor.float()
        src_row_frac = torch.where(src_row_floor < 0, torch.zeros_like(src_row_frac), src_row_frac)
        src_row_floor = torch.where(src_row_floor < 0, torch.zeros_like(src_row_floor), src_row_floor)
        src_row_frac = torch.where(src_row_floor >= height - 1, torch.ones_like(src_row_frac), src_row_frac)
        src_row_floor = torch.where(
            src_row_floor >= height - 1, torch.full_like(src_row_floor, height - 2), src_row_floor
        )
        weight_bottom = (src_row_frac * 2048 + 0.5).floor().to(torch.int32)
        weight_top = 2048 - weight_bottom  # (target_h,)

        image_uint8 = image.clamp(0, 255).to(torch.uint8)  # (C, H, W)
        image_int32 = image_uint8.to(torch.int32)  # (C, H, W)
        col_left = src_col_floor.long()  # (target_w,)
        col_right = (src_col_floor + 1).long()  # (target_w,)  safe: src_col_floor <= width-2
        row_top = src_row_floor.long()  # (target_h,)
        row_bottom = (src_row_floor + 1).long()  # (target_h,)
        # gather 4 neighbours: (C, target_h, target_w)
        pixel_top_left = image_int32[:, row_top[:, None], col_left[None, :]]
        pixel_top_right = image_int32[:, row_top[:, None], col_right[None, :]]
        pixel_bottom_left = image_int32[:, row_bottom[:, None], col_left[None, :]]
        pixel_bottom_right = image_int32[:, row_bottom[:, None], col_right[None, :]]
        # fixed-point bilinear: weights broadcast over (C, target_h, target_w)
        weight_bottom_3d = weight_bottom.view(1, target_height, 1)
        weight_top_3d = weight_top.view(1, target_height, 1)
        weight_right_3d = weight_right.view(1, 1, target_width)
        weight_left_3d = weight_left.view(1, 1, target_width)
        interp = weight_top_3d * (
            weight_left_3d * pixel_top_left + weight_right_3d * pixel_top_right
        ) + weight_bottom_3d * (weight_left_3d * pixel_bottom_left + weight_right_3d * pixel_bottom_right)
        interp = (interp + (1 << 21)) >> 22
        result = interp.clamp(0, 255).to(torch.uint8)  # (B*C, target_h, target_w)

        return result.view(batch_size, channels, target_height, target_width).to(dtype=image.dtype)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "tvF.InterpolationMode | int | None",
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
        if resample is not None and not is_torchdynamo_compiling():
            logger.warning_once("Resampling is not supported in SLANeXt")

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._resize(image=stacked_images, size=size)
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
            outputs ([`SLANeXtForTableRecognitionOutput`]):
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
    "SLANeXtImageProcessor",
    "SLANeXtConfig",
    "SLANeXtSLAHead",
    "SLANeXtBackbone",
    "SLANeXtForTableRecognition",
    "SLANeXtPreTrainedModel",
]
