# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for FAST."""
from ...utils.import_utils import is_cv2_available, is_torch_available
import math
if is_cv2_available():
    import cv2
if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
from transformers.models.textnet.image_processing_textnet import TextNetImageProcessor

class FastImageProcessor(TextNetImageProcessor):
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"] , with the longest edge
        resized to keep the input aspect ratio. Both the height and width are resized to be divisible by 32.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            size_divisor (`int`, *optional*, defaults to `32`):
                Ensures height and width are rounded to a multiple of this value after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            default_to_square (`bool`, *optional*, defaults to `False`):
                The value to be passed to `get_size_dict` as `default_to_square` when computing the image size. If the
                `size` argument in `get_size_dict` is an `int`, it determines whether to default to a square image or
                not.Note that this attribute is not used in computing `crop_size` via calling `get_size_dict`.
        """
        if "shortest_edge" in size:
            size = size["shortest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        height, width = get_resize_output_image_size(
            image, size=size, input_data_format=input_data_format, default_to_square=False
        )
        if height % self.size_divisor != 0:
            height += self.size_divisor - (height % self.size_divisor)
        if width % self.size_divisor != 0:
            width += self.size_divisor - (width % self.size_divisor)
        #TODO: would be great to find a more efficient way in modular
        # as we're only adding this line of code
        self.img_size = (height, width)
        return resize(
            image,
            size=(height, width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    def _max_pooling(self, input_tensor, scale=1):
        kernel_size = self.pooling_size // 2 + 1 if scale == 2 else self.pooling_size
        padding = (self.pooling_size // 2) // 2 if scale == 2 else (self.pooling_size - 1) // 2
        
        pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        
        pooled_output = pooling(input_tensor)
        return pooled_output

    def post_process_text_detection(self, output, target_sizes, threshold, bbox_type="rect", img_size=None):
        scale = 2
        img_size = img_size if img_size is not None else self.img_size
        out = output["last_hidden_state"]
        batch_size = out.size(0)
        final_results = {}

        texts = F.interpolate(
            out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale), mode="nearest"
        )  # B*1*320*320
        texts = self._max_pooling(texts, scale=scale)  # B*1*320*320
        score_maps = torch.sigmoid_(texts)  # B*1*320*320
        score_maps = F.interpolate(score_maps, size=(img_size[0], img_size[1]), mode="nearest")  # B*1*640*640
        score_maps = score_maps.squeeze(1)  # B*640*640

        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
        labels_ = []
        for kernel in kernels.numpy():
            import cv2

            ret, label_ = cv2.connectedComponents(kernel)
            labels_.append(label_)
        labels_ = np.array(labels_)
        labels_ = torch.from_numpy(labels_)
        labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
        labels = F.interpolate(
            labels, size=(img_size[0] // scale, img_size[1] // scale), mode="nearest"
        )  # B*1*320*320
        labels = self._max_pooling(labels, scale=scale)
        labels = F.interpolate(labels, size=(img_size[0], img_size[1]), mode="nearest")  # B*1*640*640
        labels = labels.squeeze(1).to(torch.int32)  # B*640*640

        keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]

        final_results.update({"kernels": kernels.data.cpu()})

        results = []
        for i in range(batch_size):
            org_img_size = target_sizes[i]
            scales = (float(org_img_size[1]) / float(img_size[1]), float(org_img_size[0]) / float(img_size[0]))

            bboxes, scores = self.generate_bbox(
                keys[i], labels[i], score_maps[i], scales, threshold, bbox_type=bbox_type
            )
            results.append({"bboxes": bboxes, "scores": scores})
        final_results.update({"results": results})

        return results

    def generate_bbox(self, keys, label, score, scales, threshold, bbox_type):
        label_num = len(keys)
        bboxes = []
        scores = []
        for index in range(1, label_num):
            i = keys[index]
            ind = label == i
            ind_np = ind.data.cpu().numpy()
            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < threshold:
                label[ind] = 0
                continue

            if bbox_type == "rect":
                rect = cv2.minAreaRect(points[:, ::-1])
                alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
                rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
                bbox = cv2.boxPoints(rect) * scales

            elif bbox_type == "poly":
                binary = np.zeros(label.shape, dtype="uint8")
                binary[ind_np] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scales
            bbox = bbox.astype("int32")
            bboxes.append(bbox.reshape(-1).tolist())
            scores.append(score_i)

        return bboxes, scores

__all__ = [
    "FastImageProcessor",
]