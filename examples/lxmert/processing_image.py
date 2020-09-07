"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ResizeShortestEdge:
    def __init__(self, short_edge_length, max_size=sys.maxsize, interp=Image.BILINEAR):
        """
        Args:
            short_edge_length (list[min, max])
            max_size (int): maximum allowed longest edge length.
        """
        self.interp_method = interp
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, imgs):
        img_augs = []
        for img in imgs:
            h, w = img.shape[:2]
            # later: provide list and randomly choose index for resize
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
            if size == 0:
                return img
            scale = size * 1.0 / min(h, w)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size
            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            if img.dtype == np.uint8:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.resize((neww, newh), self.interp_method)
                ret = np.asarray(pil_image)
            else:
                if any(x < 0 for x in img.strides):
                    img = np.ascontiguousarray(img)
                img = torch.from_numpy(img)
                shape = list(img.shape)
                shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
                img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
                img = F.interpolate(
                    img, (newh, neww), mode=self.interp_method, align_corners=False
                )
                shape[:2] = (newh, neww)
                ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
            img_augs.append(ret)

        ret = np.stack(img_augs)

        return ret


class Preprocess:
    def __init__(self, cfg):
        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.SIZE_DIVISIBILITY
        self.pad_value = cfg.PAD_VALUE
        self.max_image_size = cfg.INPUT.MAX_SIZE_TEST
        self.device = cfg.MODEL.DEVICE
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).to(self.device)
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).to(self.device)
        self.normalizer = lambda x: (
            x - self.pixel_mean.view(1, -1, 1, 1) / self.pixel_std.view(1, -1, 1, 1)
        )

    def pad(self, images):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        image_sizes = [im.shape[-2:] for im in images]
        images = [
            F.pad(
                im,
                [0, max_size[-1] - size[1], 0, max_size[-2] - size[0]],
                value=self.pad_value,
            )
            for size, im in zip(image_sizes, images)
        ]

        return torch.stack(images), torch.tensor(image_sizes)

    def __call__(self, images, input_format="BGR"):
        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]
            if self.input_format == "RGB":
                images = [im[:, :, ::-1] for im in images]
            # resize shape
            raw_sizes = torch.tensor([im.shape[:2] for im in images])
            images = self.aug(images)
            # reshape
            images = torch.as_tensor(images.astype("float32").transpose(0, 3, 1, 2)).to(
                self.device
            )
            # Normalize
            images = self.normalizer(images)
            if self.size_divisibility > 0:
                raise NotImplementedError()
            # pad
            images, sizes = self.pad(images)
            scales_yx = torch.true_divide(raw_sizes, sizes).tolist()
            return images, sizes, scales_yx


def _scale_box(boxes, scale_yx):
    boxes[:, 0::2] *= scale_yx[:, 1]
    boxes[:, 1::2] *= scale_yx[:, 0]
    return boxes


def tensorize(im):
    im = cv2.imread(im)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)
