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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import normalize, pad, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import TensorType
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2VisionAttention,
    GotOcr2VisionEncoder,
)


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
            stdv = 1.0 / math.sqrt(module.hidden_size * 1.0)
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
        config: SLANeXtConfig,
        post_conv_in_channels: int = 256,
        post_conv_out_channels: int = 512,
    ):
        super().__init__()

        self.vision_tower = SLANeXtVisionEncoder(config)
        self.post_conv = nn.Conv2d(
            post_conv_in_channels, post_conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False
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

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class SLANeXtLocationMLP(nn.Module):
    def __init__(self, hidden_size, loc_reg_num):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, loc_reg_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class SLANeXtSLAHead(nn.Module):
    def __init__(
        self,
        in_channels=512,
        hidden_size=512,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num
        self.eos = self.num_embeddings - 1

        self.structure_attention_cell = SLANeXtAttentionGRUCell(in_channels, hidden_size, self.num_embeddings)
        self.structure_generator = SLANeXtStructureMLP(hidden_size, out_channels)

        self.loc_generator = SLANeXtLocationMLP(hidden_size, loc_reg_num)

    def forward(self, features, targets=None):
        batch_size = features.shape[0]

        hidden = torch.zeros((batch_size, self.hidden_size), device=features.device)
        structure_preds_list = []
        structure_ids_list = []
        pre_chars = torch.zeros(size=[batch_size], dtype=torch.long, device=features.device)
        for _ in range(self.max_text_length + 1):
            hidden, structure_step, loc_step = self._decode(pre_chars, features, hidden)
            pre_chars = structure_step.argmax(dim=1)
            structure_preds_list.append(structure_step)
            structure_ids_list.append(pre_chars)
            if torch.stack(structure_ids_list, dim=1).eq(self.eos).any(-1).all():
                break
        structure_preds = F.softmax(torch.stack(structure_preds_list, dim=1), dim=-1)

        return structure_preds

    def _decode(self, pre_chars, features, hidden):
        emb_feature = self.emb(pre_chars)
        (output, hidden), alpha = self.structure_attention_cell(hidden, features, emb_feature)

        structure_step = self.structure_generator(output)
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        return F.one_hot(input_char, self.num_embeddings).float()


@auto_docstring
class SLANeXtModel(SLANeXtPreTrainedModel):
    """
    Core SLANeXt model, consisting of Backbone and Head networks.
    Generates structure probs for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = SLANeXtBackbone(
            config=config.vision_config,
            post_conv_in_channels=config.post_conv_in_channels,
            post_conv_out_channels=config.post_conv_out_channels,
        )
        self.head = SLANeXtSLAHead(
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            max_text_length=config.max_text_length,
            loc_reg_num=config.loc_reg_num,
        )
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
class SLANeXtImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SLANeXt image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's long edge to the specified `size`. Can be overridden by the `do_resize`
            parameter in the `preprocess` method.
        size (`dict[str, int]`, *optional*, defaults to `{"height": 512, "width": 512}`):
            Size of the image after resizing. The image is resized such that the long edge matches
            `max(size["height"], size["width"])` while preserving the aspect ratio. Can be overridden by the `size`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
            method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to a square of size `max(size["height"], size["width"])` after resizing. Padding
            is applied to the bottom and right edges with zeros. Can be overridden by the `do_pad` parameter in the
            `preprocess` method.
    """

    model_input_names = ["pixel_values"]
    valid_kwargs = ImagesKwargs

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 512, "width": 512}
        size = get_size_dict(size)

        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad

        self.init_decoder()

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        **kwargs,
    ):
        """
        Resize an image while preserving the aspect ratio, using a custom bilinear interpolation that closely
        matches OpenCV's `cv2.resize` with `INTER_LINEAR`. The image is scaled so that its long edge matches
        `max(size["height"], size["width"])`.

        This implementation uses OpenCV's approach with vectorized operations:
        1. Float32 precision for all floating-point calculations
        2. Fixed-point arithmetic with 11-bit precision (scale = 2048)
        3. Vectorized bilinear interpolation for efficiency
        4. Proper boundary handling

        Args:
            image (`np.ndarray`):
                Image to resize, in HWC format (height, width, channels).
            size (`dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the target size. The image's
                long edge will be resized to `max(size["height"], size["width"])` while maintaining the aspect ratio.

        Returns:
            `np.ndarray`: The resized image in HWC format.
        """
        height, width = image.shape[:2]
        scale = max(size["height"], size["width"]) / max(height, width)
        target_height = round(height * scale)
        target_width = round(width * scale)

        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV's fixed-point arithmetic constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scale factors using float32 (matching OpenCV)
        scale_x = np.float32(width) / np.float32(target_width)
        scale_y = np.float32(height) / np.float32(target_height)

        # Pre-compute X interpolation tables (vectorized)
        dx_arr = np.arange(target_width, dtype=np.float32)
        fx_arr = (dx_arr + np.float32(0.5)) * scale_x - np.float32(0.5)
        sx_arr = np.floor(fx_arr).astype(np.int32)
        fx_frac_arr = fx_arr - sx_arr.astype(np.float32)

        # Handle X boundaries
        mask_left = sx_arr < 0
        mask_right = sx_arr >= width - 1
        sx_arr[mask_left] = 0
        fx_frac_arr[mask_left] = 0.0
        sx_arr[mask_right] = width - 2
        fx_frac_arr[mask_right] = 1.0

        xalpha = np.round(fx_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        xofs = sx_arr

        # Pre-compute Y interpolation tables (vectorized)
        dy_arr = np.arange(target_height, dtype=np.float32)
        fy_arr = (dy_arr + np.float32(0.5)) * scale_y - np.float32(0.5)
        sy_arr = np.floor(fy_arr).astype(np.int32)
        fy_frac_arr = fy_arr - sy_arr.astype(np.float32)

        # Handle Y boundaries
        mask_top = sy_arr < 0
        mask_bottom = sy_arr >= height - 1
        sy_arr[mask_top] = 0
        fy_frac_arr[mask_top] = 0.0
        sy_arr[mask_bottom] = height - 2
        fy_frac_arr[mask_bottom] = 1.0

        yalpha = np.round(fy_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        yofs = sy_arr

        # Create meshgrid for vectorized operations
        sy_grid = yofs[:, np.newaxis]  # (target_h, 1)
        sx_grid = xofs[np.newaxis, :]  # (1, target_w)
        ay_grid = yalpha[:, np.newaxis]  # (target_h, 1)
        ax_grid = xalpha[np.newaxis, :]  # (1, target_w)

        ay_inv = INTER_RESIZE_COEF_SCALE - ay_grid
        ax_inv = INTER_RESIZE_COEF_SCALE - ax_grid

        # Perform vectorized bilinear interpolation for each channel
        output = np.zeros((target_height, target_width, image.shape[2]), dtype=np.uint8)

        for channel_idx in range(image.shape[2]):
            # Get 4 corner pixels using advanced indexing
            p00 = image[sy_grid, sx_grid, channel_idx].astype(np.int32)  # (target_h, target_w)
            p10 = image[sy_grid, sx_grid + 1, channel_idx].astype(np.int32)
            p01 = image[sy_grid + 1, sx_grid, channel_idx].astype(np.int32)
            p11 = image[sy_grid + 1, sx_grid + 1, channel_idx].astype(np.int32)

            # Vectorized bilinear interpolation
            val = ay_inv * (ax_inv * p00 + ax_grid * p10) + ay_grid * (ax_inv * p01 + ax_grid * p11)

            # Divide with rounding
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            output[:, :, channel_idx] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def preprocess(
        self,
        image: ImageInput,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_pad: bool = True,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ):
        """
        Preprocess an image or batch of images for the SLANeXt model.

        Args:
            image (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. The image is resized such that the long edge matches
                `max(size["height"], size["width"])` while preserving the aspect ratio.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to a square after resizing.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `np.ndarray`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                    - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad

        size = size if size is not None else self.size
        size = get_size_dict(size)

        image = make_flat_list_of_images(image)
        if not valid_images(image):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")
        image = to_numpy_array(image[0])

        if do_resize:
            image = self.resize(image=image, size=size)

        if do_normalize:
            image = image.astype(np.float32) / 255.0
            image = normalize(image=image, mean=image_mean, std=image_std)

        if do_pad:
            target_pad_size = max(size["height"], size["width"])
            pad_bottom = max(0, target_pad_size - image.shape[0])
            pad_right = max(0, target_pad_size - image.shape[1])
            image = pad(image=image, padding=((0, pad_bottom), (0, pad_right)))

        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        image = np.expand_dims(image, axis=0)

        return BatchFeature(data={"pixel_values": image}, tensor_type=return_tensors)

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
        self.bos_id = np.array(self.dict["sos"])
        self.eos_id = np.array(self.dict["eos"])

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
        self.pred = outputs.last_hidden_state.detach().cpu()
        structure_probs = self.pred[0:1].numpy()
        ignored_tokens = [self.bos_id, self.eos_id]
        end_idx = self.eos_id

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_str_list = []
        batch_size = len(structure_idx)
        for batch_index in range(batch_size):
            structure_list = []
            score_list = []
            for position in range(len(structure_idx[batch_index])):
                char_idx = int(structure_idx[batch_index][position])
                if position > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                structure_list.append(text)
                score_list.append(structure_probs[batch_index, position])
            structure_str_list.append(structure_list)
            structure_score = np.mean(score_list)

        structure = ["<html>", "<body>", "<table>"] + structure_str_list[0] + ["</table>", "</body>", "</html>"]
        return {"structure": structure, "structure_score": structure_score}


@auto_docstring(
    custom_intro="""
    SLANeXt model for table structure recognition tasks. Wraps the core SLANeXtModel
    and returns outputs compatible with the Transformers table recognition API.
    """
)
class SLANeXtForTableRecognition(SLANeXtPreTrainedModel):
    """
    SLANeXt model for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.model = SLANeXtModel(config)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.model(pixel_values)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "SLANeXtForTableRecognition",
    "SLANeXtImageProcessor",
    "SLANeXtConfig",
    "SLANeXtModel",
    "SLANeXtPreTrainedModel",
]
