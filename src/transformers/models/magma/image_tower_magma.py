# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Image processor class for Magma."""

from typing import List, Optional, Union
import logging

# Configure root logger
logging.basicConfig(level=logging.INFO)

import numpy as np
import torchvision
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
)

from transformers.utils import TensorType, is_vision_available, logging
# logging.set_verbosity_info()
logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image

import torchvision

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from open_clip.transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from open_clip.pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg    
from pathlib import Path
from typing import Optional, Tuple, Type
from functools import partial
import torch.utils.checkpoint as checkpoint
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import asdict
HF_HUB_PREFIX = 'hf-hub:'

def _get_hf_config(model_id, cache_dir=None):
    config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_path_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if device == "auto":
        device = {'': device}
    else:
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logger.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logger.info(f'Loaded {model_name} model config.')
        else:
            logger.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_path_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["timm_drop_path"] = force_path_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        # model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                if device != {'':'auto'}:
                    model.to(device=device, dtype=dtype)
                else:
                    model.to(dtype=dtype)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        # else:
        #     model.to(device=device)
        
        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            # if checkpoint_path:
            #     logger.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            #     open_clip.load_checkpoint(model, checkpoint_path)
            # else:
            #     error_str = (
            #         f'Pretrained weights ({pretrained}) not found for model {model_name}.'
            #         f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
            #     logger.warning(error_str)
            #     raise RuntimeError(error_str)
            # pretrained_loaded = True
        elif has_hf_hub_prefix and require_pretrained:
            logger.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            print(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            open_clip.load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_path_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    return create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_path_dropout=force_path_dropout, 
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

class D2CLIP_HF(nn.Module):
    def __init__(self, config, **kwargs):    
        super().__init__()
        self.model_name = config['vision_backbone']
        
        require_pretrained = kwargs.get('require_pretrained', False)
        if self.model_name == "convnextxxlarge":
            clip_model = create_model_and_transforms('hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg', require_pretrained=require_pretrained)
        elif self.model_name == "convnextlarge":
            clip_model = create_model_and_transforms('hf-hub:laion/CLIP-convnext_large-laion2B-s34B-b82K-augreg', require_pretrained=require_pretrained)

        self.clip_vision_model = clip_model.visual

        model_name = self.model_name.lower()
        assert 'convnext' in model_name, f"Only convnext backbone is supported for Magma model, but got {model_name}"
        self.model_type = 'convnext'
        if 'xxlarge' in model_name:
            self.output_channels = [384, 384, 768, 1536, 3072]
        elif 'large' in model_name:
            self.output_channels = [192, 192, 384, 768, 1536]    
        elif 'base' in model_name:
            self.output_channels = [128, 128, 256, 512, 1024]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
        }

    def extract_features_convnext(self, x, gradient_checkpointing=True):
        out = {}
        x = self.clip_vision_model.trunk.stem(x)
        if gradient_checkpointing:
            x = checkpoint.checkpoint(self.clip_vision_model.trunk.stages, x)
        else:
            x = self.clip_vision_model.trunk.stages(x)
        out['clip_vis_dense'] = x
        return out
            

    def forward(self, x, gradient_checkpointing=True):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        return self.extract_features_convnext(x, gradient_checkpointing=gradient_checkpointing)      

    @property
    def size_divisibility(self):
        return 32

class MagmaImageTower(D2CLIP_HF):
    r"""
    Constructs a Magma image processor. Based on [`CLIPImageProcessor`] with incorporation of additional techniques
    for processing high resolution images as explained in the [InternLM-XComposer2-4KHD](https://arxiv.org/pdf/2404.06512)

    Args:
        config (dict): Configuration dictionary containing the keys for the image processor.
    """

    def __init__(
        self,
        config, 
        **kwargs
    ) -> None:
        super().__init__(config, **kwargs)

    @property
    def hidden_size(self):
        return self.output_channels[-1]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): A tensor of shape (N, C, H, W) representing an image.

        Returns:
            torch.Tensor: A tensor of shape (N, C, H, W) representing the processed image.
        """
        return super().forward(x)