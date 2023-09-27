# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for ViT."""

from typing import Dict, Optional, Union

import numpy as np

import os
import re
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import cv2
import scipy

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class ViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    def __init__(
        self,
        video: str,
        mask: str,
        mode: str,
        scale_h: float = 1.0,
        scale_w: float = 1.2,
        resize_ratio: float = 1.0,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = {"height": -1, "width": -1},
        mask_dilation: int = 4,
        ref_stride: int = 10,
        neighbor_length: int = 10,
        subvideo_length: int = 80,
        raft_iter: int = 20,
        save_fps: int = 24,
        do_normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.video = video,
        self.mask = mask,
        self.mode = mode,
        self.scale_h = scale_h,
        self.scale_w = scale_w,
        self.resize_ratio = resize_ratio
        self.do_resize = do_resize
        self.size = size
        self.mask_dilation = mask_dilation
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.raft_iter = raft_iter
        self.save_fps = save_fps
        self.do_normalize = do_normalize


    def to_tensors(self):
        return transforms.Compose([Stack(), ToTorchFormatTensor()])

    def get_device(gpu_id=None):
        IS_HIGH_VERSION = [int(m) for m in list(re.findall(r"^([0-9]+)\.([0-9]+)\.([0-9]+)([^0-9][a-zA-Z0-9]*)?(\+git.*)?$",\
            torch.__version__)[0][:3])] >= [1, 12, 0]

        if gpu_id is None:
            gpu_str = ''
        elif isinstance(gpu_id, int):
            gpu_str = f':{gpu_id}'
        else:
            raise TypeError('Input should be int value.')

        if IS_HIGH_VERSION:
            if torch.backends.mps.is_available():
                return torch.device('mps'+gpu_str)
        return torch.device('cuda'+gpu_str if torch.cuda.is_available() and torch.backends.cudnn.is_available() else 'cpu')

    def read_frame_from_videos(self, frame_root):
        if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
            video_name = os.path.basename(frame_root)[:-4]
            vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
            frames = list(vframes.numpy())
            frames = [Image.fromarray(f) for f in frames]
            fps = info['video_fps']
        else:
            video_name = os.path.basename(frame_root)
            frames = []
            fr_lst = sorted(os.listdir(frame_root))
            for fr in fr_lst:
                frame = cv2.imread(os.path.join(frame_root, fr))
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame)
            fps = None
        size = frames[0].size

        return frames, fps, size, video_name
    
    def binary_mask(self, mask, th=0.1):
        mask[mask>th] = 1
        mask[mask<=th] = 0
        return mask
    
    def read_mask(self, mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
        masks_img = []
        masks_dilated = []
        flow_masks = []

        if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            masks_img = [Image.open(mpath)]
        else:  
            mnames = sorted(os.listdir(mpath))
            for mp in mnames:
                masks_img.append(Image.open(os.path.join(mpath, mp)))

        for mask_img in masks_img:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))

            # Dilate 8 pixel so that all known pixel is trustworthy
            if flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
            else:
                flow_mask_img = self.binary_mask(mask_img).astype(np.uint8)
            # Close the small holes inside the foreground objects
            flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
            flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))

            if mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = self.binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))

        if len(masks_img) == 1:
            flow_masks = flow_masks * length
            masks_dilated = masks_dilated * length

        return flow_masks, masks_dilated

    def extrapolation(self, video_ori, scale):
        """Prepares the data for video outpainting.
        """
        nFrame = len(video_ori)
        imgW, imgH = video_ori[0].size

        # Defines new FOV.
        imgH_extr = int(scale[0] * imgH)
        imgW_extr = int(scale[1] * imgW)
        imgH_extr = imgH_extr - imgH_extr % 8
        imgW_extr = imgW_extr - imgW_extr % 8
        H_start = int((imgH_extr - imgH) / 2)
        W_start = int((imgW_extr - imgW) / 2)

        # Extrapolates the FOV for video.
        frames = []
        for v in video_ori:
            frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
            frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
            frames.append(Image.fromarray(frame))

        # Generates the mask for missing region.
        masks_dilated = []
        flow_masks = []
        
        dilate_h = 4 if H_start > 10 else 0
        dilate_w = 4 if W_start > 10 else 0
        mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

        mask[H_start+dilate_h: H_start+imgH-dilate_h, 
            W_start+dilate_w: W_start+imgW-dilate_w] = 0
        flow_masks.append(Image.fromarray(mask * 255))

        mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
        masks_dilated.append(Image.fromarray(mask * 255))

        flow_masks = flow_masks * nFrame
        masks_dilated = masks_dilated * nFrame
        
        return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)

    def preprocess(
        self,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        device = self.get_device()

        frames, fps, size, video_name = self.read_frame_from_videos(self.video)
        if not self.size['width'] == -1 and not self.size['height'] == -1:
            size = (self.size['width'], self.size['height'])
        if not self.resize_ratio == 1.0:
            size = (int(self.resize_ratio * size[0]), int(self.resize_ratio * size[1]))

        if self.do_resize:
            frames, size, out_size = self.resize_frames(frames, size)

        if self.mode == 'video_inpainting':
            frames_len = len(frames)
            flow_masks, masks_dilated = self.read_mask(self.mask, frames_len, size,
                                                flow_mask_dilates=self.mask_dilation,
                                                mask_dilates=self.mask_dilation)
        elif self.mode == 'video_outpainting':
            frames, flow_masks, masks_dilated, size = self.extrapolation(frames, (self.scale_h, self.scale_w))
        else:
            raise NotImplementedError

        frames = self.to_tensors()(frames).unsqueeze(0) * 2 - 1 
        flow_masks = self.to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = self.to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

        data = {"frames": frames, "flow_masks": flow_masks, "distil_masks": masks_dilated}
        return BatchFeature(data=data, tensor_type=return_tensors)


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img
