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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import normalize, pad
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..focalnet.modeling_focalnet import FocalNetMlp
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2VisionAttention,
    GotOcr2VisionEncoder,
)


class SLANeXtVisionConfig(GotOcr2VisionConfig):
    pass


class SLANeXtMlp(FocalNetMlp):
    pass


class SLANeXtVisionAttention(GotOcr2VisionAttention):
    pass


@auto_docstring(
    checkpoint="PaddlePaddle/SLANeXt_wired",
    custom_intro="Configuration for the SLANeXt model.",
    custom_args=r"""
    vision_config (`SLANeXtVisionConfig` or `dict`, *optional*):
        Configuration for the vision encoder. If not provided, default values will be used.
    encoder_embed_dim (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder embeddings, used as the hidden size of the vision encoder (ViT backbone).
    encoder_output_channels (`int`, *optional*, defaults to 256):
        Number of output channels produced by the vision encoder's neck (projection layer after the transformer
        blocks).
    encoder_num_channels (`int`, *optional*, defaults to 3):
        Number of input image channels for the vision encoder (e.g., 3 for RGB).
    encoder_patch_size (`int`, *optional*, defaults to 16):
        Size of each image patch for the vision encoder's patch embedding.
    encoder_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function used in the vision encoder.
    encoder_layer_norm_eps (`float`, *optional*, defaults to 1e-6):
        The epsilon value used by the layer normalization layers in the vision encoder.
    encoder_attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities in the vision encoder.
    encoder_qkv_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query, key, and value projections in the vision encoder's attention layers.
    encoder_use_abs_pos (`bool`, *optional*, defaults to `True`):
        Whether to use absolute position in the vision encoder's attention layers.
    encoder_use_rel_pos(`bool`, *optional*, defaults to `True`):
        Whether to use relative position in the vision encoder's attention layers.
    encoder_window_size (`int`, *optional*, defaults to 14):
        Window size for windowed (local) attention in the vision encoder layers.
    encoder_depth (`int`, *optional*, defaults to 12):
        Number of transformer encoder layers in the vision backbone.
    encoder_num_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the vision encoder.
    encoder_global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        Indexes of the encoder layers that use global (non-windowed) attention instead of local window attention.
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
class SLANeXtConfig(PreTrainedConfig):

    model_type = "slanext"
    sub_configs = {"vision_config": SLANeXtVisionConfig}

    def __init__(
        self,
        vision_config=None,
        image_size: int = 512,
        encoder_embed_dim: int = 768,
        encoder_output_channels: int=256,
        encoder_num_channels: int=3,
        encoder_patch_size: int=16,
        encoder_hidden_act: str="gelu",
        encoder_layer_norm_eps: float=1e-6,
        encoder_attention_dropout: float=0.0,
        encoder_qkv_bias: bool=True,
        encoder_use_abs_pos: bool=True,
        encoder_use_rel_pos: bool=True,
        encoder_window_size: int=14,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_global_attn_indexes: list[int] = [2, 5, 8, 11],
        post_conv_in_channels: int=256,
        post_conv_out_channels: int=512,
        out_channels: int = 50,
        hidden_size: int = 512,
        max_text_length: int = 500,
        loc_reg_num: int = 8,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = SLANeXtVisionConfig(
                hidden_size=encoder_embed_dim,
                output_channels=encoder_output_channels,
                num_hidden_layers=encoder_depth,
                num_attention_heads=encoder_num_heads,
                num_channels=encoder_num_channels,
                image_size=image_size,
                patch_size=encoder_patch_size,
                hidden_act=encoder_hidden_act,
                layer_norm_eps=encoder_layer_norm_eps,
                attention_dropout=encoder_attention_dropout,
                qkv_bias=encoder_qkv_bias,
                use_abs_pos=encoder_use_abs_pos,
                use_rel_pos=encoder_use_rel_pos,
                window_size=encoder_window_size,
                global_attn_indexes=encoder_global_attn_indexes,
                mlp_dim=int(encoder_embed_dim * 4),
            )
        elif isinstance(vision_config, dict):
            vision_config = SLANeXtVisionConfig(**vision_config)
        self.vision_config = vision_config
        super().__init__(**kwargs)
        self.post_conv_in_channels = post_conv_in_channels
        self.post_conv_out_channels= post_conv_out_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.loc_reg_num = loc_reg_num


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
                for layer in generator:
                    if isinstance(layer, nn.Linear):
                        init.uniform_(layer.weight, -stdv, stdv)
                        if layer.bias is not None:
                            init.uniform_(layer.bias, -stdv, stdv)


class SLANeXtVisionEncoder(GotOcr2VisionEncoder):
    pass


class SLANeXtVary_VIT_B(nn.Module):
    def __init__(
        self,
        vision_config: SLANeXtConfig,
        post_conv_in_channels: int=256,
        post_conv_out_channels: int=512,
    ):
        super().__init__()

        self.vision_tower = SLANeXtVisionEncoder(vision_config)
        self.post_conv = nn.Conv2d(
            post_conv_in_channels,
            post_conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
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


class SLANeXtHWAttention(nn.Module):
    def __init__(
        self,
        head_dim=32,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, hidden_states):
        batch_size, seq_len, total_channels = hidden_states.shape
        channels_per_head = total_channels // 3
        qkv = hidden_states.reshape(batch_size, seq_len, 3, channels_per_head // self.head_dim, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        query, key, value = qkv.unbind(0)
        attn = torch.matmul(query, key.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, -1)
        attn = self.attn_drop(attn)
        hidden_states = torch.matmul(attn, value)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, channels_per_head)
        return hidden_states


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    batch_size, height, width, channels = img.shape
    img_reshape = img.reshape(batch_size, height // H_sp, H_sp, width // W_sp, W_sp, channels)
    img_perm = img_reshape.permute(0, 1, 3, 2, 4, 5).reshape(-1, H_sp * W_sp, channels)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    batch_size = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.reshape(batch_size, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).flatten(1, 4)
    return img


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
        """
        @param in_channels: input shape
        @param hidden_size: hidden_size for RNN and Embedding
        @param out_channels: num_classes to rec
        @param max_text_length: max text pred
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num
        self.eos = self.num_embeddings - 1

        self.structure_attention_cell = SLANeXtAttentionGRUCell(in_channels, hidden_size, self.num_embeddings)
        self.structure_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(hidden_size, out_channels),
        )

        dpr = np.linspace(0, 0.1, 2)

        self.loc_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, loc_reg_num),
            nn.Sigmoid(),
        )

    def forward(self, inputs, targets=None):
        features = inputs
        batch_size = features.shape[0]

        hidden = torch.zeros((batch_size, self.hidden_size), device=features.device)
        structure_preds = torch.zeros(
            (batch_size, self.max_text_length + 1, self.num_embeddings), device=features.device
        )
        loc_preds = torch.zeros((batch_size, self.max_text_length + 1, self.loc_reg_num), device=features.device)
        structure_preds.requires_grad = False
        loc_preds.requires_grad = False
        structure_ids = torch.zeros(
            (batch_size, self.max_text_length + 1), dtype=torch.long, device=features.device
        )
        pre_chars = torch.zeros(size=[batch_size], dtype=torch.long, device=features.device)
        for i in range(self.max_text_length + 1):
            hidden, structure_step, loc_step = self._decode(pre_chars, features, hidden)
            pre_chars = structure_step.argmax(dim=1)
            structure_preds[:, i, :] = structure_step
            loc_preds[:, i, :] = loc_step

            structure_ids[:, i] = pre_chars
            if (structure_ids == self.eos).any(-1).all():
                break
        structure_preds = F.softmax(structure_preds[:, : i + 1], dim=-1)
        loc_preds = loc_preds[:, : i + 1]

        return structure_preds

    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        (output, hidden), alpha = self.structure_attention_cell(hidden, features, emb_feature)

        structure_step = self.structure_generator(output)
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        return F.one_hot(input_char, self.num_embeddings).float()


@auto_docstring(custom_intro="The SLANeXt model.")
class SLANeXtModel(SLANeXtPreTrainedModel):
    """
    Core SLANeXt model, consisting of Backbone and Head networks.
    Generates structure probs for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = SLANeXtVary_VIT_B(
            vision_config=config.vision_config,
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
        self, 
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        backbone_states = self.backbone(pixel_values)
        hidden_states = self.head(backbone_states)
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=backbone_states,
        )


@auto_docstring(custom_intro="ImageProcessor for the SLANeXt model.")
class SLANeXtImageProcessor(BaseImageProcessor):
    def __init__(self):
        self.target_long_edge = 512
        self.target_pad_size = 512
        self.init_decoder()

    def _tablerec_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to match cv2.resize with INTER_LINEAR as closely as possible.

        This implementation uses OpenCV's approach with vectorized operations:
        1. Float32 precision for all floating-point calculations
        2. Fixed-point arithmetic with 11-bit precision (scale = 2048)
        3. Vectorized bilinear interpolation for efficiency
        4. Proper boundary handling

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        height, width = img.shape[:2]
        target_width, target_height = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
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
        output = np.zeros((target_height, target_width, img.shape[2]), dtype=np.uint8)

        for channel_idx in range(img.shape[2]):
            # Get 4 corner pixels using advanced indexing
            p00 = img[sy_grid, sx_grid, channel_idx].astype(np.int32)  # (target_h, target_w)
            p10 = img[sy_grid, sx_grid + 1, channel_idx].astype(np.int32)
            p01 = img[sy_grid + 1, sx_grid, channel_idx].astype(np.int32)
            p11 = img[sy_grid + 1, sx_grid + 1, channel_idx].astype(np.int32)

            # Vectorized bilinear interpolation
            val = ay_inv * (ax_inv * p00 + ax_grid * p10) + ay_grid * (ax_inv * p01 + ax_grid * p11)

            # Divide with rounding
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            output[:, :, channel_idx] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def preprocess(self, img):
        img = np.array(img)
        height, width = img.shape[:2]
        scale = self.target_long_edge / max(height, width)
        height_resize = round(height * scale)
        width_resize = round(width * scale)
        img = self._tablerec_resize(img, [width_resize, height_resize])
        img = img / 255.0
        img = normalize(image=img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        height, width = img.shape[:2]
        pad_right = max(0, self.target_pad_size - width)
        pad_bottom = max(0, self.target_pad_size - height)
        img = pad(image=img, padding=((0, pad_bottom), (0, pad_right)))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img).float()

        return img

    def init_decoder(self, merge_no_span_structure=True):
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

        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = ["sos"] + dict_character + ["eos"]
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict["sos"])
        elif beg_or_end == "end":
            idx = np.array(self.dict["eos"])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def post_process_table_recognition(self, outputs):
        self.pred = outputs.last_hidden_state.detach().cpu()
        structure_probs = np.array([list(self.pred[0])])
        """convert text-label into text-index."""
        ignored_tokens = [self.get_beg_end_flag_idx("beg"), self.get_beg_end_flag_idx("end")]
        end_idx = self.dict["eos"]

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

        structure_str_list = [
            ["<html>", "<body>", "<table>"] + structure + ["</table>", "</body>", "</html>"]
            for structure in structure_str_list
        ]

        return [{"structure": structure, "structure_score": structure_score} for structure in structure_str_list][0]


@auto_docstring(custom_intro="TableRecognition for the SLANeXt model.")
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
