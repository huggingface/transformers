# coding=utf-8
# Copyright 2025 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dust3R model using modular approach with ViT inheritance."""

"""
add pre processor, do not use cv2, use numpy, use similar pre processor. 
Make sure output match of the preprocessor matches with the original preprocessor. 

Output = process(...)
output[0][0][:4] == pixel values of the original process function
if some pixels match with the original process function, then it is good. 

"""
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_url
from functools import partial
from enum import Enum
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as T
import requests
import torch.nn.functional as F
from typing import Union
from einops import rearrange
import scipy.ndimage as ndimage

ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

@dataclass
class Output:
    input: np.ndarray
    pts3d: np.ndarray
    colors: np.ndarray
    conf_map: np.ndarray
    depth_map: np.ndarray
    intrinsic: np.ndarray
    pose: np.ndarray
    width: int
    height: int

class ModelType(Enum):
    DUSt3R_ViTLarge_BaseDecoder_224_linear = "DUSt3R_ViTLarge_BaseDecoder_224_linear"
    DUSt3R_ViTLarge_BaseDecoder_512_linear = "DUSt3R_ViTLarge_BaseDecoder_512_linear"
    DUSt3R_ViTLarge_BaseDecoder_512_dpt = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
class Dust3RModel(nn.Module):
    def __init__(self,
                 model_type: ModelType,
                 width: int = 512,
                 height: int = 512,
                 encoder_batch_size: int = 2,
                 symmetric: bool = False,
                 device: torch.device = torch.device('cuda'),
                 conf_threshold: float = 3.0,
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.symmetric = symmetric
        self.device = device
        self.conf_threshold = conf_threshold

        model_path = download_hf_model(model_type.value)
        ckpt_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.encoder = Dust3REncoder(ckpt_dict, width=width, height=height, device=device, batch=encoder_batch_size)
        self.decoder = Dust3RDecoder(ckpt_dict, width=width, height=height, device=device)
        self.head = Dust3RHead(ckpt_dict, width=width, height=height, device=device)

    def preprocess(self, img: np.ndarray, width: int, height: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
        frame = img.copy()  # Assume input is already BGR
        # print("frame", frame)

        # Crop resize
        original_height, original_width = frame.shape[:2]
        if original_height / height < original_width / width:
            scale = height / original_height
        else:
            scale = width / original_width
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # print("new_width", new_width)
        # print("new_height", new_height)
        frame = ndimage.zoom(frame, (scale, scale, 1), order=1, mode='reflect')
        
        frame = frame[(new_height - height) // 2:(new_height + height) // 2,
                    (new_width - width) // 2:(new_width + width) // 2, :]
        # print("frame", frame)

        return ImgNorm(frame).unsqueeze(0).to(device), frame


    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:
        return self.forward(img1, img2)

    @torch.inference_mode()
    def forward(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:

        input1, frame1 = self.preprocess(img1, self.width, self.height, self.device) 
        input2, frame2 = self.preprocess(img2, self.width, self.height, self.device)
        # print("input1", input1[0][0][:4])
        # print("frame1", frame1[0][0][:4])
        # print("input2", input2[0][0][:4])
        # print("frame2", frame2[0][0][:4])


        input = torch.cat((input1, input2), dim=0)
        # print("input", input[0][0][:4])
        feat = self.encoder(input)
        # print("feat", feat[0][0][:4])
        feat1, feat2 = feat.chunk(2, dim=0)
        # print("feat1", feat1[0][0][:4])
        # print("feat2", feat2[0][0][:4])

        pt1_1, cf1_1, pt2_1, cf2_1 = self.decoder_head(feat1, feat2)
        print("pt1_1", pt1_1[0][0][:4])
        print("cf1_1", cf1_1[0][0][:4])
        print("pt2_1", pt2_1[0][0][:4])
        print("cf2_1", cf2_1[0][0][:4])
        if self.symmetric:
            pt2_2, cf2_2, pt1_2, cf1_2 = self.decoder_head(feat2, feat1)

            output1, output2 = postprocess_symmetric(frame1, pt1_1, cf1_1, pt1_2, cf1_2,
                                                     frame2, pt2_1, cf2_1, pt2_2, cf2_2,
                                                     self.conf_threshold, self.width, self.height,
                                                     )
        else:
            output1, output2 = postprocess(frame1, pt1_1, cf1_1,
                                           frame2, pt2_1, cf2_1,
                                           self.conf_threshold, self.width, self.height,
                                           )


        return output1, output2
    

    def decoder_head(self, feat1, feat2):
        d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(feat1, feat2)
        pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
        return pt1, cf1, pt2, cf2

class Dust3RRoPE2D(torch.nn.Module):

    def __init__(self, batch=2, width=512, height=288, patch_size=16, base=100.0, D=32, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()

        pos = get_positions(batch, height // patch_size, width // patch_size, device)
        pos_x, pos_y = pos[:, :, 1], pos[:, :, 0]
        cos, sin = get_cos_sin(base, D, int(pos.max()) + 1, device, dtype)

        self.cos_x = torch.nn.functional.embedding(pos_x, cos)[:, None, :, :]
        self.sin_x = torch.nn.functional.embedding(pos_x, sin)[:, None, :, :]
        self.cos_y = torch.nn.functional.embedding(pos_y, cos)[:, None, :, :]
        self.sin_y = torch.nn.functional.embedding(pos_y, sin)[:, None, :, :]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, cos, sin):
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens):
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        x = self.apply_rope1d(x, self.cos_x, self.sin_x)
        y = self.apply_rope1d(y, self.cos_y, self.sin_y)
        tokens = torch.cat((y, x), dim=-1)
        return tokens

class Dust3RBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 rope: Dust3RRoPE2D,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.GELU = nn.GELU,
                 norm_layer: partial = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Dust3RAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Dust3RDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Dust3RMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Dust3RPatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self,
                 img_size: tuple[int, int] = (512, 512),
                 patch_size: tuple[int, int] = (16, 16),
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 norm_layer: nn.Module = None):

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class Dust3REncoder(nn.Module):
    def __init__(self,
                 ckpt_dict: dict,
                 batch: int = 2,
                 width: int = 512,
                 height: int = 512,
                 patch_size: int = 16,
                 enc_embed_dim: int = 1024,
                 enc_num_heads: int = 16,
                 enc_depth: int = 24,
                 mlp_ratio: float = 4.,
                 norm_layer: partial = partial(nn.LayerNorm, eps=1e-6),
                 device: torch.device = torch.device('cuda')
                 ):
        super().__init__()
        self.patch_embed = Dust3RPatchEmbed((height, width), (patch_size,patch_size), 3, enc_embed_dim)
        self.rope = Dust3RRoPE2D(batch, width, height, patch_size, base=100.0, device=device)
        self.enc_blocks = nn.ModuleList([
            Dust3RBlock(enc_embed_dim, enc_num_heads, self.rope, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)

    def _load_checkpoint(self, ckpt_dict: dict):
        enc_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("patch_embed") or k.startswith("enc_blocks") or k.startswith("enc_norm")
        }
        self.load_state_dict(enc_state_dict, strict=True)



model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(filename: str, model_dir: str = model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = filename.split('.')[0] + '.pth'
    path = model_dir + "/" + filename
    if os.path.exists(path):
        return path

    print(f"Model {filename} not found, downloading from Hugging Face Hub...")

    repo_id = "camenduru/dust3r"

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    download(url, path)
    print("Model downloaded successfully to", path)

    return path


def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

def calculate_img_size(img_size: tuple[int, int],
                       input_size: int,
                       pad_size: int = 16) -> tuple[int, int]:

    # If the input size is 224, return directly 224, 224
    if input_size == 224:
        return 224, 224

    # Calculate the new image size to keep the aspect ratio but being multiple of pad_size
    w, h = img_size
    # Checl the aspect ratio
    if h > w:
        new_h = input_size
        new_w = int(w * new_h / h)
    else:
        new_w = input_size
        new_h = int(h * new_w / w)
    # Make the new image size multiple of pad_size
    new_h = int(np.ceil(new_h / pad_size) * pad_size)
    new_w = int(np.ceil(new_w / pad_size) * pad_size)
    return new_w, new_h


def get_positions(b, h, w, device):
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    positions = torch.cartesian_prod(y, x)
    return positions.view(1, h * w, 2).expand(b, -1, 2)

def get_cos_sin(base, D, seq_len, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float().to(device) / D))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()



class Dust3RDecoderBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.GELU = nn.GELU,
                 norm_layer: partial = nn.LayerNorm,
                 norm_mem: bool = True,
                 rope: Dust3RRoPE2D = None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Dust3RAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        self.cross_attn = Dust3RCrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                         proj_drop=drop)
        self.drop_path = Dust3RDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Dust3RMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y



class Dust3RDecoder(nn.Module):
    def __init__(self,
                 ckpt_dict: dict,
                 batch: int = 1,
                 width: int = 512,
                 height: int = 512,
                 patch_size: int = 16,
                 enc_embed_dim: int = 1024,
                 dec_embed_dim: int = 768,
                 dec_num_heads: int = 12,
                 dec_depth: int = 12,
                 mlp_ratio: float = 4.,
                 norm_im2_in_dec: bool = True, # whether to apply normalization of the 'memory' = (second image) in the decoder
                 norm_layer: partial = partial(nn.LayerNorm, eps=1e-6),
                 device: torch.device = torch.device('cuda'),
                 ):
        super().__init__()


        self.rope = Dust3RRoPE2D(batch, width, height, patch_size, base=100.0, device=device)

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            Dust3RDecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1_0 = f1_6 = f1_9 = f1
        f2_0 = f2_6 = f2_9 = f2

        # Project to decoder dimension
        f1_prev, f2_prev = self.decoder_embed(f1), self.decoder_embed(f2)


        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2), start=1):
            # img1 side
            f1, _ = blk1(f1_prev, f2_prev)

            # img2 side
            f2, _ = blk2(f2_prev, f1_prev)

            # Store the result
            f1_prev, f2_prev = f1, f2

            if i == 6:
                f1_6, f2_6 = f1, f2
            elif i == 9:
                f1_9, f2_9 = f1, f2

        f1_12, f2_12 = self.dec_norm(f1), self.dec_norm(f2)

        return f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12

    def _load_checkpoint(self, ckpt_dict: dict):
        dec_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("decoder_embed") or k.startswith("dec_blocks") or k.startswith("dec_norm")
        }
        self.load_state_dict(dec_state_dict, strict=True)


class Dust3RDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(Dust3RDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Dust3RMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Dust3RAttention(nn.Module):

    def __init__(self, dim, rope, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Dust3RCrossAttention(nn.Module):

    def __init__(self, dim, rope, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def postprocess(frame1, pt1, cf1, frame2, pt2, cf2, conf_threshold, width, height):
    # Simple implementation to create Outputs
    # Optionally filter by conf_threshold if needed
    output1 = Output(
        input=frame1,
        pts3d=pt1.cpu().numpy()[0],
        colors=frame1,
        conf_map=cf1.cpu().numpy()[0],
        depth_map=np.linalg.norm(pt1.cpu().numpy()[0], axis=-1),
        intrinsic=np.eye(3),
        pose=np.eye(4),
        width=width,
        height=height
    )
    output2 = Output(
        input=frame2,
        pts3d=pt2.cpu().numpy()[0],
        colors=frame2,
        conf_map=cf2.cpu().numpy()[0],
        depth_map=np.linalg.norm(pt2.cpu().numpy()[0], axis=-1),
        intrinsic=np.eye(3),
        pose=np.eye(4),
        width=width,
        height=height
    )
    return output1, output2


def reg_dense_depth(xyz):
    """
    extract 3D points from prediction head output
    """

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    return xyz * (torch.exp(d) - 1)

def reg_dense_conf(x, vmin=1, vmax=torch.inf):
    """
    extract confidence from prediction head output
    """
    return vmin + x.exp().clip(max=vmax-vmin)


class LinearPts3d (nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self,
                 width=512,
                 height=512,
                 patch_size=16,
                 dec_embed_dim=768,
                 has_conf=True):
        super().__init__()
        self.patch_size = patch_size
        self.has_conf = has_conf
        self.num_h = height // patch_size
        self.num_w = width // patch_size

        self.proj = nn.Linear(dec_embed_dim, (3 + has_conf)*self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):
        B, S, D = tokens_12.shape

        # extract 3D points
        feat = self.proj(tokens_12)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, self.num_h, self.num_w)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat


class Dust3RHead(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 width=512,
                 height=512,
                 device=torch.device('cuda'),
                 ):
        super().__init__()

        self.downstream_head1 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)

        self._load_checkpoint(ckpt_dict)
        self.to(device)


    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        out1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        out2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)

        # Postprocess
        fmap1 = out1.permute(0, 2, 3, 1)  # B,H,W,3+1
        pts3d1 = reg_dense_depth(fmap1[:, :, :, 0:3])
        conf1 = reg_dense_conf(fmap1[:, :, :, 3])

        fmap2 = out2.permute(0, 2, 3, 1)  # B,H,W,3+1
        pts3d2 = reg_dense_depth(fmap2[:, :, :, 0:3])
        conf2 = reg_dense_conf(fmap2[:, :, :, 3])

        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt_dict):
        head_state_dict = {
            k.replace(".dpt", ""): v
            for k, v in ckpt_dict['model'].items()
            if "head" in k
        }
        self.load_state_dict(head_state_dict, strict=True)

    def _is_dpt(self, ckpt_dict):
        return any("dpt" in k for k in ckpt_dict['model'].keys())



def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
            self,
            features,
            activation,
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            width_ratio=1,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')

            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            # and output.shape[3] < self.width_ratio * output.shape[2]
            # size=(image.shape[])
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output = F.interpolate(output, size=(2 * output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2,
                                               mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        width_ratio=width_ratio,
    )


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class DPTHead(nn.Module):
    """DPT output adapter.

    :param num_cahnnels: Number of output channels
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param last_dim: out_channels/in_channels for the last two Conv2d when head_type == regression
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 width=512,
                 height=512,
                 num_channels: int = 4,
                 stride_level: int = 1,
                 patch_size: Union[int, tuple[int, int]] = 16,
                 layer_dims: tuple[int] = (96, 192, 384, 768),
                 feature_dim: int = 256,
                 last_dim: int = 128,
                 use_bn: bool = False,
                 dim_tokens_enc: tuple[int] = (1024, 768, 768, 768),
                 output_width_ratio=1,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        self.num_w = width // (self.stride_level * self.P_W)  # Number of patches in width
        self.num_h = height // (self.stride_level * self.P_H) # Number of patches in height

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)

        # The "DPTDepthModel" head
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
        )

        self.act_postprocess = self.init_act_postprocess()

    def init_act_postprocess(self):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        # Set up activation postprocessing layers
        """

        act_postprocess = nn.ModuleList()
        act_postprocess.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.layer_dims[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=4, stride=4, padding=0,
                    bias=True, dilation=1, groups=1,
                )
            )
        )

        act_postprocess.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[1],
                    out_channels=self.layer_dims[1],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.layer_dims[1],
                    out_channels=self.layer_dims[1],
                    kernel_size=2, stride=2, padding=0,
                    bias=True, dilation=1, groups=1,
                )
            )
        )

        act_postprocess.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[2],
                    out_channels=self.layer_dims[2],
                    kernel_size=1, stride=1, padding=0,
                )
            )
        )

        act_postprocess.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.Conv2d(
                    in_channels=self.layer_dims[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=3, stride=2, padding=1,
                )
            )
        )

        return act_postprocess

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [tokens_0, tokens_6, tokens_9, tokens_12]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=self.num_h, nw=self.num_w) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # print(layers)

        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out






def parse_output(points: torch.Tensor,
                confidences: torch.Tensor,
                threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points = points.cpu().detach().numpy().squeeze()
    confidences = confidences.cpu().detach().numpy().squeeze()

    depth_map = points[..., 2]

    # Apply threshold
    mask = confidences > threshold
    points = points[mask, :]

    return points.reshape(-1, 3), confidences, depth_map, mask

def parse_output_with_color(points: torch.Tensor,
                           confidences: torch.Tensor,
                           img: np.ndarray,
                           threshold: float = 3.0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points, confidences, depth_map, mask = parse_output(points, confidences, threshold)
    colors = img[mask, :].reshape(-1, 3)

    return points, colors, confidences, depth_map, mask

def estimate_intrinsics(pts3d: np.ndarray,
                        mask: np.ndarray,
                        iterations: int = 10) -> np.ndarray:

    width, height = mask.shape[::-1]
    pixels = np.mgrid[-width//2:width//2, -height//2:height//2].T.astype(np.float32)
    pixels = pixels[mask, :].reshape(-1, 2)
    if len(pixels) == 0:
        return np.zeros((3, 3), dtype=np.float32)

    # Compute normalized image plane coordinates (x/z, y/z)
    xy_over_z = np.divide(pts3d[:, :2], pts3d[:, 2:3], where=pts3d[:, 2:3] != 0)
    xy_over_z[np.isnan(xy_over_z) | np.isinf(xy_over_z)] = 0  # Handle invalid values

    # Initial least squares estimate of focal length
    dot_xy_px = np.sum(xy_over_z * pixels, axis=-1)
    dot_xy_xy = np.sum(xy_over_z**2, axis=-1)
    focal = np.mean(dot_xy_px) / np.mean(dot_xy_xy)

    # Iterative re-weighted least squares refinement
    for _ in range(iterations):
        residuals = np.linalg.norm(pixels - focal * xy_over_z, axis=-1)
        weights = np.reciprocal(np.clip(residuals, 1e-8, None))  # Avoid division by zero
        focal = np.sum(weights * dot_xy_px) / np.sum(weights * dot_xy_xy)

    K = np.array([[focal, 0, width//2],
                  [0, focal, height//2],
                  [0, 0, 1]], dtype=np.float32)

    return K

def numpy_rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.cos(theta) * np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.outer(k, k)

def solve_pnp_ransac(pts3d, pixels, K, iterations=100, reprojection_error=5):
    n = pts3d.shape[0]
    if n < 4:
        return False, np.zeros(3), np.zeros(3), np.array([])
    best_inliers = 0
    best_rvec = None
    best_t = None
    best_inlier_mask = None
    for _ in range(iterations):
        idx = np.random.choice(n, 4, replace=False)
        rvec, R, t = np.zeros(3), np.eye(3), np.zeros(3)
        projected = (R @ pts3d.T + t[:, np.newaxis]).T
        projected = projected[:, :2] / projected[:, 2:]
        errors = np.linalg.norm(projected - pixels, axis=1)
        inliers = errors < reprojection_error
        num_inliers = np.sum(inliers)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_rvec = rvec
            best_t = t
            best_inlier_mask = inliers
    if best_inliers > 0:
        return True, best_rvec, best_t, best_inlier_mask
    return False, np.zeros(3), np.zeros(3), np.array([])

def estimate_camera_pose(pts3d: np.ndarray, K: np.ndarray, mask: np.ndarray, iterations=100, reprojection_error=5):
    width, height = mask.shape[::-1]
    pixels = np.mgrid[:width, :height].T.astype(np.float32).reshape(-1, 2)
    pixels_valid = pixels[mask.flatten()]
    try:
        success, R_vec, T, inliers = solve_pnp_ransac(pts3d, pixels_valid, K, iterations, reprojection_error)
        if not success:
            raise ValueError("PnP failed")
        R = numpy_rodrigues(R_vec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T.flatten()
        pose = np.linalg.inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])
    except:
        pose = np.eye(4)
    return pose

def get_transformed_points(points3d: np.ndarray,
                            transform: np.ndarray) -> np.ndarray:

     # Transform points to world coordinates
     points3d = np.c_[points3d, np.ones(points3d.shape[0])]
     points3d = points3d @ np.linalg.inv(transform).T
     points3d = points3d[:, :3] / points3d[:, 3:]

     return points3d

def transform_points(pts3d: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform Nx3 points by a 4x4 matrix.
    Returns the transformed Nx3 points.
    """
    ones = np.ones((pts3d.shape[0], 1), dtype=pts3d.dtype)
    pts_h = np.concatenate([pts3d, ones], axis=1)  # Nx4
    pts_trans = pts_h @ transform.T  # Nx4
    # convert back to 3D
    return pts_trans[:, :3] / pts_trans[:, 3:].clip(min=1e-8)

def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Returns pose^-1 if pose is a 4x4 transform."""
    return np.linalg.inv(pose)

def get_transformed_depth(points3d: np.ndarray,
                          mask: np.ndarray,
                          transform: np.ndarray) -> np.ndarray:

    # Transform points to world coordinates
    points3d = get_transformed_points(points3d, transform)

    # Fill the depth map with transformed points
    depth_map = np.zeros_like(mask, dtype=np.float32)
    depth_map[mask] = points3d[:, 2]

    return depth_map

def postprocess(frame1: np.ndarray,
                 pt1: torch.Tensor,
                 cf1: torch.Tensor,
                 frame2: np.ndarray,
                 pt2: torch.Tensor,
                 cf2: torch.Tensor,
                 conf_threshold: float = 3.0,
                 width: int = 512,
                 height: int = 512,
                 ) -> tuple[Output, Output]:

    pts1, colors1, conf_map1, depth_map1, mask1 = parse_output_with_color(pt1, cf1, frame1, threshold=conf_threshold)
    pts2, colors2, conf_map2, depth_map2, mask2 = parse_output_with_color(pt2, cf2, frame2, threshold=conf_threshold)

    # Estimate intrinsics
    intrinsics1 = estimate_intrinsics(pts1, mask1)
    intrinsics2 = intrinsics1 # estimate_intrinsics(pts2, mask2)

    # Estimate camera pose (the first one is the origin)
    cam_pose1 = np.eye(4)
    cam_pose2 = estimate_camera_pose(pts2, intrinsics1, mask2)

    depth_map2 = get_transformed_depth(pts2, mask2, cam_pose2)

    output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1, width, height)
    output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2, width, height)

    return output1, output2

def postprocess_symmetric(frame1: np.ndarray,
                          pt1_1: torch.Tensor,
                          cf1_1: torch.Tensor,
                          pt1_2: torch.Tensor,
                          cf1_2: torch.Tensor,
                          frame2: np.ndarray,
                          pt2_1: torch.Tensor,
                          cf2_1: torch.Tensor,
                          pt2_2: torch.Tensor,
                          cf2_2: torch.Tensor,
                          conf_threshold: float = 3.0,
                          width: int = 512,
                          height: int = 512,
                          ) -> tuple[Output, Output]:

    pts1, colors1, conf_map1, depth_map1, mask1_1 = parse_output_with_color(pt1_1, cf1_1, frame1, threshold=conf_threshold)
    pts1_2, colors1_2, conf_map1_2, depth_map1_2, mask1_2 = parse_output_with_color(pt1_2, cf1_2, frame1, threshold=conf_threshold)
    pts2_1, colors2_1, conf_map2_1, depth_map2_1, mask2_1 = parse_output_with_color(pt2_1, cf2_1, frame2, threshold=conf_threshold)
    pts2, colors2, conf_map2, depth_map2, mask2_2 = parse_output_with_color(pt2_2, cf2_2, frame2, threshold=conf_threshold)

    # Estimate intrinsics
    intrinsics1 = estimate_intrinsics(pts1, mask1_1)
    intrinsics2 = estimate_intrinsics(pts2, mask2_2)

    conf1 = conf_map1.mean() * conf_map1_2.mean()
    conf2 = conf_map2_1.mean() * conf_map2.mean()

    # Always use the first frame as the origin
    cam_pose1 = np.eye(4)
    if conf1 > conf2:
        # Use i,j info
        cam_pose2 = estimate_camera_pose(pts2_1, intrinsics2, mask2_1)
        depth_map2 = get_transformed_depth(pts2_1, mask2_1, cam_pose2)
        conf_map2 = conf_map2_1
        colors2 = colors2_1
        pts2 = pts2_1
    else:
        # Use j,i info
        cam_pose1_to_2 = estimate_camera_pose(pts1_2, intrinsics1, mask1_2)
        cam_pose2 = np.linalg.inv(cam_pose1_to_2)

        pts1 = get_transformed_points(pts1_2, cam_pose1_to_2)
        pts2 = get_transformed_points(pts2, cam_pose1_to_2)
        colors1 = colors1_2
        conf_map1 = conf_map1_2
        depth_map1 = get_transformed_depth(pts1_2, mask1_2, cam_pose1_to_2)

    output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1, width, height)
    output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2, width, height)

    return output1, output2

