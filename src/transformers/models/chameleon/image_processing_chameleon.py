# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Chameleon."""

from typing import Dict

import numpy as np
import PIL
import torch

from ...image_processing_utils import BaseImageProcessor, BatchFeature


class ChameleonImageProcessor(BaseImageProcessor):
    r"""
    Constructs an Chameleon image processor.

    Args:
        ddconfig (`dict`):
            TODO
        n_embed (`int`):
            TODO
        embed_dim (`int`):
            TODO
        ckpt_path (`str`, *optional*):
            TODO
        ignore_keys (`list`, *optional*, defaults to `[]`):
            TODO
        image_key (`str`, *optional*, defaults to `"image"`):
            TODO
        colorize_nlabels (`int`, *optional*):
            TODO
        remap (`dict`, *optional*):
            TODO
        sane_index_shape (`bool`, *optional*, defaults to `False`):
            TODO
    """

    def __init__(
        self,
        ddconfig: Dict,
        n_embed: int,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        remap=None,
        sane_index_shape=False,
    ):
        self._vq_model = VQModel(
            ddconfig=ddconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self._vq_model.eval()

        self._vocab = None

        devices = {p.device for p in self._vq_model.parameters()}
        assert len(devices) == 1
        self._device = devices.pop()

        dtypes = {p.dtype for p in self._vq_model.parameters()}
        assert len(dtypes) == 1
        self._dtype = dtypes.pop()

    def preprocess(self, image: PIL.Image) -> BatchFeature:
        assert self._vocab is not None

        image = self._whiten_transparency(image)
        vqgan_input = self._vqgan_input_from(image).to(self._device).to(self._dtype)
        _, _, [_, _, img_toks] = self._vq_model.encode(vqgan_input)
        bpe_toks = (
            [self._vocab.image_start_id] + self._vocab.convert_img2bp2(img_toks).tolist() + [self._vocab.image_end_id]
        )
        return BatchFeature(data={"tokens": bpe_toks})

    def set_vocab(self, vocab_map: dict[str, int]):
        self._vocab = _VocabInfo(vocab_map)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()  # CHW format.

        # Add batch dimension.
        return tensor_img.unsqueeze(0)

    def decode(self, x: torch.Tensor) -> PIL.Image:
        pass
        # The below is left intentionally commented out.
        # x = x.detach().cpu()
        # x = torch.clamp(x, -1.0, 1.0)
        # x = (x + 1.0) / 2.0
        # x = x.permute(1, 2, 0).numpy()
        # x = (255 * x).astype(np.uint8)
        # x = Image.fromarray(x)
        # if not x.mode == "RGB":
        #     x = x.convert("RGB")
        # return x
