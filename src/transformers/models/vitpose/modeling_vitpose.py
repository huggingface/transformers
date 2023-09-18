# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViTPose model."""


import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union
from timm.models.layers import drop_path
import numpy as np
import cv2

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision.ops import box_convert

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_vitpose import ViTPoseConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTPoseConfig"

# Base docstring ## to be changed
_CHECKPOINT_FOR_DOC = "shauray/ViTPose"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "shauray/vitpose"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shauray/vitpose",
    # See all ViTPose models at https://huggingface.co/models?filter=vitpose
]

## GREEN
class ViTPosePatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTPoseConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim
        self.patch_size = config.patch_size
        self.image_size = config.img_size
        self.image_size = self.image_size if isinstance(self.image_size, collections.abc.Iterable) else (self.image_size, self.image_size)
        self.patch_size = self.patch_size if isinstance(self.patch_size, collections.abc.Iterable) else (self.patch_size, self.patch_size)
        self.num_patches = (self.image_size[1] // self.patch_size[1]) * (self.image_size[0] // self.patch_size[0])

        self.projection = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding = (2,2))

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: Optional[bool]=False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(pixel_values)
        Hp, Wp = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        print("afsdasfd",embeddings.shape)
        return embeddings, (Hp,Wp)

    
class ViTPoseEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTPoseConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None
        self.patch_embeddings = ViTPosePatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout_p)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings, (Hp,Wp) = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        print(embeddings.shape)
        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        #cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        #embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            #embeddings = embeddings + self.position_embeddings
            embeddings = embeddings
        
        
        print(embeddings.shape)
        #embeddings = self.dropout(embeddings)

        return embeddings, (Hp,Wp)

## GREEN
class ViTPoseAttention(nn.Module):
    def __init__(self, config: ViTPoseConfig) -> None:
        super().__init__()
        if config.embed_dim % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.embed_dim,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.embed_dim // config.num_attention_heads
        self.all_head_size = self.attention_head_size * config.num_attention_heads

        self.qkv = nn.Linear(config.embed_dim, self.all_head_size*3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.dropout_p)
        self.proj = nn.Linear(self.all_head_size, config.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout_p)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        print("asd",hidden_states.shape)
        B, N, C = hidden_states.shape
        qkv_layer = self.qkv(hidden_states)
        qkv_layer = qkv_layer.reshape(B, N, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)

        query_layer, key_layer, value_layer = qkv_layer[0], qkv_layer[1], qkv_layer[2]

       # value_layer = self.transpose_for_scores(value_layer)
       # key_layer = self.transpose_for_scores(key_layer)
       # query_layer = self.transpose_for_scores(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_drop(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(1,2).reshape(B, N, -1)

        attention_output = self.proj(context_layer)
        attention_output = self.proj_drop(attention_output)

        return attention_output

class ViTPoseMLP(nn.Module):
    def __init__(self, config: ViTPoseConfig) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim*config.mlp_ratio,bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=config.embed_dim*config.mlp_ratio, out_features=config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

    def forward(self,pixel_values: torch.Tensor) -> torch.Tensor:

        """A pretty generic MLP block"""
        pixel_values = self.fc1(pixel_values)
        pixel_values = self.act(pixel_values)
        pixel_values = self.fc2(pixel_values)
        pixel_values = self.dropout(pixel_values)

        return pixel_values

class ViTPoseBlock(nn.Module):
    def __init__(
        self, 
        config: ViTPoseConfig, 
        layer: Optional[int] = 0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-06, elementwise_affine=True)
        self.attn = ViTPoseAttention(config)
        self.drop_path = DropPath(drop_prob = config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-06, elementwise_affine=True)
        self.mlp = ViTPoseMLP(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        pixel_values = pixel_values + self.drop_path(self.attn(self.norm1(pixel_values)))
        pixel_values = pixel_values + self.drop_path(self.mlp(self.norm2(pixel_values)))

        return pixel_values

class DropPath(nn.Module):
    def __init__(self, drop_prob = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'
        

## LOOKS GREEN
class ViTPoseEncoder(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTPoseConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList([ViTPoseBlock(config, i) for i in range(config.depth)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
        head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> torch.Tensor:
        #batch_size, num_channels, height, width = pixel_values.shape
        all_hidden_states = () if output_hidden_states else None
        all_self_attention = () if output_attentions else None

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attention)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                hidden_states = layer_module(hidden_states)

            #hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attention] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states = all_hidden_states,
            attentions = all_self_attention
        )
        
class ViTPoseTopDownHeatMap(nn.Module):
    """
    Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers and a simple conv2d layer.
    """

    def __init__(self, config: ViTPoseConfig):
        super().__init__()
        self.config = config
        self.deconv_layers = []
        if config.keypoint_num_deconv_layer > 0:
          for i in range(config.keypoint_num_deconv_layer):
              in_channels = config.embed_dim if i == 0 else config.keypoint_num_deconv_filters[i - 1]
              out_channels = config.keypoint_num_deconv_filters[i]
              kernel_size = config.keypoint_num_deconv_kernels[i]
              self.deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False))
              self.deconv_layers.append(nn.BatchNorm2d(out_channels))
              self.deconv_layers.append(nn.ReLU(inplace=True))

        elif config.keypoint_num_deconv_layer == 0:
            self.deconv_layers.append(nn.Identity())
        
        else:
            raise ValueError(
                f"num_deconv_layers ({self.num_deconv_layers}) should >= 0."
            )

        self.deconv_layers = nn.Sequential(*self.deconv_layers)
        self.final_layer = nn.Conv2d(config.keypoint_num_deconv_filters[-1], config.num_output_channels, kernel_size=1, stride=1)

    def transform_preds(self, coords, center, scale, output_size, use_udp=False):
        scale = scale * 200.0
        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = torch.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

        return target_coords


    def post_dark_udp(self, coords, batch_heatmaps, kernel=3):
        B, K, H, W = batch_heatmaps.shape
        N = coords.shape[0]
        assert (B == 1 or B == N)
        for heatmaps in batch_heatmaps:
            for heatmap in heatmaps:
                cv2.GaussianBlur(heatmap.numpy(), (kernel, kernel), 0, heatmap.numpy())
        torch.clamp(batch_heatmaps, min=0.001, max=50., out=batch_heatmaps)
        torch.log(batch_heatmaps, out=batch_heatmaps)

        batch_heatmaps_pad = torch.nn.functional.pad(
            batch_heatmaps, (1,1,1,1),
            mode='replicate').flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * torch.arange(0, B * K).reshape(-1, K)
        index = index.type(torch.int32).reshape(-1, 1)
        i_ = batch_heatmaps_pad[index]
        ix1 = batch_heatmaps_pad[index + 1]
        iy1 = batch_heatmaps_pad[index + W + 2]
        ix1y1 = batch_heatmaps_pad[index + W + 3]
        ix1_y1_ = batch_heatmaps_pad[index - W - 3]
        ix1_ = batch_heatmaps_pad[index - 1]
        iy1_ = batch_heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = torch.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = torch.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = torch.linalg.inv((hessian + torch.finfo(torch.float32).eps) * torch.eye(2))
        coords -= torch.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
        return coords

    def keypoints_from_heatmap(self, heatmaps, center, scale, unbiased=False):
        heatmaps = heatmaps.clone()
        use_udp = self.config.udp
        N, K, H, W = heatmaps.shape
        if use_udp:
            if self.config.target_type.lower() == "GaussianHeatMap".lower():
                # GetMaxPreds
                heatmaps_reshaped = heatmaps.reshape((N,K,-1))
                idx = torch.argmax(heatmaps_reshaped, 2).reshape((N,K,1))
                maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N,K,1))

                preds = torch.tile(idx, (1,1,2)).type(torch.float32)
                preds[:,:,0] = preds[:,:,0] % W
                preds[:,:,1] = preds[:,:,1] // W
                preds = torch.where(torch.tile(maxvals, (1,1,2)) > 0.0, preds, -1)

                preds = self.post_dark_udp(preds, heatmaps, kernel=self.config.kernel)

            else:
                raise ValueError("target_type has to be GaussianHeatMap! no other supported")
        else:
            raise ValueError("not supported udp has to be True")

        for i in range(N):
            preds[i] = self.transform_preds(preds[i], center[i], scale[i], [W,H], use_udp=use_udp)

        return preds, maxvals

    def decode(self, img_metas, img, img_size):
        #img_metas = [img_metas]
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas:
            bbox_ids = []
        else:
            bbox_ids = None

        center = torch.zeros((batch_size, 2), dtype=torch.float32)
        scale = torch.zeros((batch_size, 2), dtype=torch.float32)
        img_paths = []
        score = torch.ones(batch_size)

        for i in range(batch_size):
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            img_paths.append(img_metas[i]['img'])

            if 'bbox_score' in img_metas[i]:
                score[i] = torch.tensor(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = self.keypoints_from_heatmap(
            img, center, scale,
        )

        all_preds = torch.zeros((batch_size, preds.shape[1], 3), dtype=torch.float32)
        all_boxes = torch.zeros((batch_size, 6), dtype=torch.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        all_boxes[:, 4] = torch.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score


        result = {}

        result["preds"] = all_preds
        result['boxes'] = all_boxes
        result["image_paths"] = img_paths
        result["bbox_ids"] = bbox_ids

        return result

    def forward(self, x):
        x = self.deconv_layers(x)
        keypoints = self.final_layer(x)
        return keypoints

    def inference_model(self, x, flip_pairs=None):
        output = self.forward(x)

        if flip_pairs is not None:
            #custom function
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type = self.target_type
            )

        else:
            output_heatmap = output.detach().cpu().numpy()

        return output_heatmap


## to be changed
class ViTPosePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTPoseConfig
    base_model_prefix = "vitpose"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, nn.ConvTranspose2d]) -> None:
        """Initialize the weights"""
        print("modeule",module)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print("ini in in in")
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTPoseEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

   # def _set_gradient_checkpointing(self, module: ViTEncoder, value: bool = False) -> None:
   #     if isinstance(module, ViTEncoder):
   #         module.gradient_checkpointing = value


VITPOSE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTPoseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITPOSE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        bboxes (`torch.FloatTensor`):
            bboxes can be obtained using external object detection pipelines such as DeTR or Yolo.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViTPose Model transformer outputting raw hidden-states without any specific head on top.",
    VITPOSE_START_DOCSTRING,
)

class ViTPoseModel(ViTPosePreTrainedModel):
    def __init__(self, config: ViTPoseConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTPoseEmbeddings(config)
        self.encoder = ViTPoseEncoder(config)
        self.layernorm = nn.LayerNorm(config.embed_dim, eps=1e-06, elementwise_affine=True)
        self.keypoint_head = ViTPoseTopDownHeatMap(config)

        self.post_init()


    def get_input_embeddings(self) -> ViTPosePatchEmbeddings:
        return self.embeddings.patch_embeddings

    #def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
    #    """
    #    Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #    class PreTrainedModel
    #    """
    #    for layer, heads in heads_to_prune.items():
    #        self.encoder.layer[layer].attention.prune_heads(heads)

    #@add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=BaseModelOutputWithPooling,
    #    config_class=_CONFIG_FOR_DOC,
    #    modality="vision",
    #    expected_output=_EXPECTED_OUTPUT_SHAPE,
    #)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_metas: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        batch_size, channels, height, width = pixel_values.shape
        results = {}
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.depth)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        pixel_values = pixel_values.to(expected_dtype) if type(pixel_values) != expected_dtype else pixel_values
        print("pe",pixel_values.shape)
        embedding_output, (Hp,Wp) = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        print(embedding_output.shape)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        print(sequence_output.shape)
        sequence_output = self.layernorm(sequence_output)
        print(sequence_output.shape)
        
        sequence_output = sequence_output.permute(0,2,1).reshape(batch_size, -1, Hp, Wp).contiguous()

        keypoint_outputs = self.keypoint_head(
            sequence_output,
        )

        output_heatmap = keypoint_outputs

        if self.config.flip_test == True:
            imgs_flipped = [pixel_value.flip(2) for pixel_value in pixel_values]

            features_flipped = [self.backbone(torch.cat(imgs_flipped, 0))]
            output_flipped_heatmap = self.keypoint_head.inference_model(features_flipped, pixel_metas[0]['flip_pairs'])

            output_heatmap = (output_heatmap + output_flipped_heatmap) * .5

        keypoint_results = self.keypoint_head.decode(
            pixel_metas, output_heatmap, img_size=list(self.config.img_size)
        )

        results.update(keypoint_results)

        if not return_dict:
            output_heatmap = None

        results["output_heatmap"] = output_heatmap
        return results
        ##??
       # if not return_dict:
       #     head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
       #     return head_outputs + encoder_outputs[1:]



class ViTPoseForPoseEstimation(ViTPosePreTrainedModel):
    def __init__(self, config: ViTPoseConfig) -> None:
        super().__init__(config)
        """Gets the architecture from ViTPoseModel above (just a pretty wrapper around the model)"""
        self.config = config
        self.vitpose = ViTPoseModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def process_det(self, results):
        """results[0] defaults to humans"""
        bboxes = results[0]
        person_results = []
        for bbox in bboxes:
            person = {}
            person["bbox"] = bbox
            person_results.append(person)
        return person_results

    def _box2cs(self, config, bbox):
        x, y, w, h = bbox[:4]
        input_size = config.img_size
        aspect_ratio = input_size[0] / input_size[1]
        center = torch.tensor([x + w * 0.5, y + h * 0.5], dtype=torch.float32, requires_grad=False)

        if w > aspect_ratio * h:
           h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
           w = h * aspect_ratio

        scale = torch.tensor([w / 200.0, h / 200.0], dtype=torch.float32, requires_grad=False)
        scale = scale * 1.25

        return center, scale

    def get_warp_matrix(self, theta, size_dst, size_target, size_input):
        theta = torch.tensor(np.deg2rad(theta))
        matrix = torch.zeros((2, 3), dtype=torch.float32)
        scale_x = size_dst[0] / size_target[0]
        scale_y = size_dst[1] / size_target[1]
        matrix[0, 0] = torch.cos(theta) * scale_x
        matrix[0, 1] = -torch.sin(theta) * scale_x
        matrix[0, 2] = scale_x * (-0.5 * size_input[0] * torch.cos(theta) +
                                  0.5 * size_input[1] * torch.sin(theta) +
                                  0.5 * size_target[0])
        matrix[1, 0] = torch.sin(theta) * scale_y
        matrix[1, 1] = torch.cos(theta) * scale_y
        matrix[1, 2] = scale_y * (-0.5 * size_input[0] * torch.sin(theta) -
                                  0.5 * size_input[1] * torch.cos(theta) +
                                  0.5 * size_target[1])
        return matrix

    def warp_affine_joints(self, joints, mat):
        """Apply affine transformation defined by the transform matrix on the
        joints.

        Args:
            joints (torch.Tensor[..., 2]): Origin coordinate of joints.
            mat (torch.Tensor[3, 2]): The affine matrix.

        Returns:
            torch.Tensor[..., 2]: Result coordinate of joints.
        """
        joints = torch.tensor(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        ones = torch.ones((joints.shape[0], 1), dtype=joints.dtype, device=joints.device)
        joints_with_ones = torch.cat((joints, ones), dim=1)
        transformed_joints = torch.mm(joints_with_ones, mat.t())
        return transformed_joints.reshape(shape)

    def processing(self,results):
        image_size = results['ann_info']['image_size']

        img_tensor = results['img']
        joints_3d_tensor = results['joints_3d']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        trans = self.get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)

        img_tensor = torch.tensor(cv2.warpAffine(
            img_tensor.numpy(),
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR
        ), dtype=torch.float32)

        joints_3d_tensor[:, 0:2] = self.warp_affine_joints(
            joints_3d_tensor[:, 0:2].clone(), trans)


        results['img'] = img_tensor
        results['joints_3d'] = joints_3d_tensor
        results["joints_3d_visible"] = joints_3d_visible

        return results


    def _inference_pose_model(self, model, img, bboxes, return_heatmap=False):
        device = next(model.parameters()).device
        if device.type == 'cpu':
            device = -1

        # add whole body thing

        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],[13, 14], [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
               [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
               [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
               [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
               [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
               [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
               [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
               [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
               [111, 132]]
        flip_pairs = body + foot + face + hand

        #flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        #      [13, 14], [15, 16]]

        batch_data = []
        for bbox in bboxes:
            center, scale = self._box2cs(self.config, bbox)

            data = {
                'center':center,
                'scale':scale,
                'bbox_scaore': bbox[4] if len(bbox) == 5 else 1,
                'bbox_id':0,
                'dataset':"TopDownCocoDataset",
                'joints_3d':torch.zeros((self.config.num_joints, 3), dtype=torch.float),
                'joints_3d_visible':torch.zeros((self.config.num_joints, 3), dtype=torch.float),
                'rotation':0,
                'ann_info': {
                    'image_size': torch.tensor(self.config.img_size),
                    'num_joints': self.config.num_joints,
                    'flip_pairs': flip_pairs
                }
            }

            data['img'] = img

            #data = self.processing(data)
            batch_data.append(data)

        with torch.no_grad():
            results = self.vitpose(pixel_values = img, pixel_metas = batch_data)

        return results['preds'], results['output_heatmap']

    def forward(self, pixel_values, pred_boxes):
        """Detection and bounding box pipeling"""
        person_results = self.process_det([pred_boxes])

        pose_results = []
        if person_results is None:
            height, width = pixel_values[2:]
            person_results = [{'bbox': torch.tensor([0,0, width, height])}]

        if len(person_results) == 0:
            return pose_results

        bboxes = [box['bbox'] for box in person_results]
        bboxes_xywh = []
        for bbox in bboxes:
            bboxes_xywh.append(box_convert(bbox, 'xyxy', 'xywh'))
        #bboxes, bboxes_xyxy = bboxes.detach().numpy(), bboxes_xyxy.detach().numpy()

        poses, heatmap = self._inference_pose_model(self.vitpose,
            pixel_values,
            bboxes_xywh,
            return_heatmap = False)

        pose_results.append(pixel_values)
        for pose, person_result, bbox in zip(poses, person_results, bboxes_xywh):
            pose_result = person_result.copy()
            pose_result['keypoints'] = pose
            pose_result['bbox'] = bbox
            pose_results.append(pose_result)

        return pose_results

#class ViTForImageClassification(ViTPosePreTrainedModel):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__(config)
#
#        self.num_labels = config.num_labels
#        self.vitpose = ViTPoseModel(config, add_pooling_layer=False)
#
#        # Classifier head
#        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#
#        # Initialize weights and apply final processing
#        self.post_init()
#
#    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#    @add_code_sample_docstrings(
#        checkpoint=_IMAGE_CLASS_CHECKPOINT,
#        output_type=ImageClassifierOutput,
#        config_class=_CONFIG_FOR_DOC,
#        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
#    )
#    def forward(
#        self,
#        pixel_values: Optional[torch.Tensor] = None,
#        head_mask: Optional[torch.Tensor] = None,
#        labels: Optional[torch.Tensor] = None,
#        output_attentions: Optional[bool] = None,
#        output_hidden_states: Optional[bool] = None,
#        interpolate_pos_encoding: Optional[bool] = None,
#        return_dict: Optional[bool] = None,
#    ) -> Union[tuple, ImageClassifierOutput]:
#        r"""
#        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#        """
#        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#        outputs = self.vit(
#            pixel_values,
#            head_mask=head_mask,
#            output_attentions=output_attentions,
#            output_hidden_states=output_hidden_states,
#            interpolate_pos_encoding=interpolate_pos_encoding,
#            return_dict=return_dict,
#        )
#
#        sequence_output = outputs[0]
#
#        logits = self.classifier(sequence_output[:, 0, :])
#
#        loss = None
#        if labels is not None:
#            # move labels to correct device to enable model parallelism
#            labels = labels.to(logits.device)
#            if self.config.problem_type is None:
#                if self.num_labels == 1:
#                    self.config.problem_type = "regression"
#                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                    self.config.problem_type = "single_label_classification"
#                else:
#                    self.config.problem_type = "multi_label_classification"
#
#            if self.config.problem_type == "regression":
#                loss_fct = MSELoss()
#                if self.num_labels == 1:
#                    loss = loss_fct(logits.squeeze(), labels.squeeze())
#                else:
#                    loss = loss_fct(logits, labels)
#            elif self.config.problem_type == "single_label_classification":
#                loss_fct = CrossEntropyLoss()
#                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#            elif self.config.problem_type == "multi_label_classification":
#                loss_fct = BCEWithLogitsLoss()
#                loss = loss_fct(logits, labels)
#
#        if not return_dict:
#            output = (logits,) + outputs[1:]
#            return ((loss,) + output) if loss is not None else output
#
#        return ImageClassifierOutput(
#            loss=loss,
#            logits=logits,
#            hidden_states=outputs.hidden_states,
#            attentions=outputs.attentions,
#        )
#]
#
#class ViTLayer(nn.Module):
#    """This corresponds to the Block class in the timm implementation."""
#
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.chunk_size_feed_forward = config.chunk_size_feed_forward
#        self.seq_len_dim = 1
#        self.attention = ViTAttention(config)
#        self.intermediate = ViTIntermediate(config)
#        self.output = ViTOutput(config)
#        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: bool = False,
#    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#        self_attention_outputs = self.attention(
#            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
#            head_mask,
#            output_attentions=output_attentions,
#        )
#        attention_output = self_attention_outputs[0]
#        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#
#        # first residual connection
#        hidden_states = attention_output + hidden_states
#
#        # in ViT, layernorm is also applied after self-attention
#        layer_output = self.layernorm_after(hidden_states)
#        layer_output = self.intermediate(layer_output)
#
#        # second residual connection is done here
#        layer_output = self.output(layer_output, hidden_states)
#
#        outputs = (layer_output,) + outputs
#
#        return outputs
#
#
#class ViTEncoder(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.config = config
#        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
#        self.gradient_checkpointing = False
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: bool = False,
#        output_hidden_states: bool = False,
#        return_dict: bool = True,
#    ) -> Union[tuple, BaseModelOutput]:
#        all_hidden_states = () if output_hidden_states else None
#        all_self_attentions = () if output_attentions else None
#
#        for i, layer_module in enumerate(self.layer):
#            if output_hidden_states:
#                all_hidden_states = all_hidden_states + (hidden_states,)
#
#            layer_head_mask = head_mask[i] if head_mask is not None else None
#
#            if self.gradient_checkpointing and self.training:
#
#                def create_custom_forward(module):
#                    def custom_forward(*inputs):
#                        return module(*inputs, output_attentions)
#
#                    return custom_forward
#
#                layer_outputs = torch.utils.checkpoint.checkpoint(
#                    create_custom_forward(layer_module),
#                    hidden_states,
#                    layer_head_mask,
#                )
#            else:
#                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
#
#            hidden_states = layer_outputs[0]
#
#            if output_attentions:
#                all_self_attentions = all_self_attentions + (layer_outputs[1],)
#
#        if output_hidden_states:
#            all_hidden_states = all_hidden_states + (hidden_states,)
#
#        if not return_dict:
#            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#        return BaseModelOutput(
#            last_hidden_state=hidden_states,
#            hidden_states=all_hidden_states,
#            attentions=all_self_attentions,
#        )
#
#
#class ViTPooler(nn.Module):
#    def __init__(self, config: ViTConfig):
#        super().__init__()
#        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#        self.activation = nn.Tanh()
#
#    def forward(self, hidden_states):
#        # We "pool" the model by simply taking the hidden state corresponding
#        # to the first token.
#        first_token_tensor = hidden_states[:, 0]
#        pooled_output = self.dense(first_token_tensor)
#        pooled_output = self.activation(pooled_output)
#        return pooled_output
#
#
#@add_start_docstrings(
#    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).
#
#    <Tip>
#
#    Note that we provide a script to pre-train this model on custom data in our [examples
#    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).
#
#    </Tip>
#    """,
#    VIT_START_DOCSTRING,
#)

#class ViTForMaskedImageModeling(ViTPreTrainedModel):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__(config)
#
#        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
#
#        self.decoder = nn.Sequential(
#            nn.Conv2d(
#                in_channels=config.hidden_size,
#                out_channels=config.encoder_stride**2 * config.num_channels,
#                kernel_size=1,
#            ),
#            nn.PixelShuffle(config.encoder_stride),
#        )
#
#        # Initialize weights and apply final processing
#        self.post_init()
#
#    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
#    def forward(
#        self,
#        pixel_values: Optional[torch.Tensor] = None,
#        bool_masked_pos: Optional[torch.BoolTensor] = None,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: Optional[bool] = None,
#        output_hidden_states: Optional[bool] = None,
#        interpolate_pos_encoding: Optional[bool] = None,
#        return_dict: Optional[bool] = None,
#    ) -> Union[tuple, MaskedImageModelingOutput]:
#        r"""
#        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
#            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
#
#        Returns:
#
#        Examples:
#        ```python
#        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
#        >>> import torch
#        >>> from PIL import Image
#        >>> import requests
#
#        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#        >>> image = Image.open(requests.get(url, stream=True).raw)
#
#        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")
#
#        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
#        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
#        >>> # create random boolean mask of shape (batch_size, num_patches)
#        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
#
#        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
#        >>> list(reconstructed_pixel_values.shape)
#        [1, 3, 224, 224]
#        ```"""
#        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
#            raise ValueError(
#                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
#                "the reconstructed image has the same dimensions as the input."
#                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
#            )
#
#        outputs = self.vit(
#            pixel_values,
#            bool_masked_pos=bool_masked_pos,
#            head_mask=head_mask,
#            output_attentions=output_attentions,
#            output_hidden_states=output_hidden_states,
#            interpolate_pos_encoding=interpolate_pos_encoding,
#            return_dict=return_dict,
#        )
#
#        sequence_output = outputs[0]
#
#        # Reshape to (batch_size, num_channels, height, width)
#        sequence_output = sequence_output[:, 1:]
#        batch_size, sequence_length, num_channels = sequence_output.shape
#        height = width = math.floor(sequence_length**0.5)
#        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
#
#        # Reconstruct pixel values
#        reconstructed_pixel_values = self.decoder(sequence_output)
#
#        masked_im_loss = None
#        if bool_masked_pos is not None:
#            size = self.config.image_size // self.config.patch_size
#            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
#            mask = (
#                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
#                .repeat_interleave(self.config.patch_size, 2)
#                .unsqueeze(1)
#                .contiguous()
#            )
#            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
#            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
#
#        if not return_dict:
#            output = (reconstructed_pixel_values,) + outputs[1:]
#            return ((masked_im_loss,) + output) if masked_im_loss is not None else output
#
#        return MaskedImageModelingOutput(
#            loss=masked_im_loss,
#            reconstruction=reconstructed_pixel_values,
#            hidden_states=outputs.hidden_states,
#            attentions=outputs.attentions,
#        )
#
#
#@add_start_docstrings(
#    """
#    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
#    the [CLS] token) e.g. for ImageNet.
#
#    <Tip>
#
#        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
#        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
#        position embeddings to the higher resolution.
#
#    </Tip>
#    """,
#    VIT_START_DOCSTRING,
#)




#print(model(torch.Tensor(numpy.zeros([1,3,256,192]))))


