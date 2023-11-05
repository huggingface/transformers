# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Any, Dict, List, Optional, Union

from ...utils.import_utils import is_cv2_available


if is_cv2_available():
    import cv2
import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    import PIL

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


class FastImageProcessor(BaseImageProcessor):
    r"""
    Constructs a FAST image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in the
            `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
            Can be overridden by the `crop_size` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            The mean to use if normalizing the image. This is a float or list of floats of length of the number of
            channels of the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            The standard deviation to use if normalizing the image. This is a float or list of floats of length of the
            number of channels of the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        min_area (`int`, *optional*, defaults to 200): Threshold for min area for results
        min_score (`float`, *optional*, defaults to 0.88): Threshold for min score for results
        bbox_type (`str`, *optional*, defaults to `"rect"`): Type of bbox, rect or poly
        pooling_size (`int`, *optional*, defaults to 9): Pooling size for text detection
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        min_area: int = 200,
        min_score: float = 0.88,
        bbox_type: str = "rect",
        pooling_size: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 640, "width": 640}
        size = get_size_dict(size)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.min_area = min_area
        self.min_score = min_score
        self.bbox_type = bbox_type
        self.pooling_size = pooling_size

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `reduce_labels` is updated if image processor
        is created using from_dict and kwargs e.g. `FastImageProcessor.from_pretrained(checkpoint, reduce_labels=True)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in kwargs:
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to (size["height"], size["width"]).

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=True, param_name="size")
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` argument must contain `height` and `width` keys. Got {size.keys()}")
        return resize(
            image,
            size=(size["height"], size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # All transformations expect numpy arrays.
        image = to_numpy_array(image)
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        image = self._preprocess(
            image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after center crop. If one edge the image is smaller than `crop_size`, it will be
                padded with zeros and then cropped
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
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
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=True, param_name="size")
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                do_center_crop=do_center_crop,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
                resample=resample,
                size=size,
                rescale_factor=rescale_factor,
                crop_size=crop_size,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]

        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1, padding=(self.pooling_size - 1) // 2)(x)
        elif scale == 2:
            x = nn.MaxPool2d(kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2)(
                x
            )
        return x

    def post_process_text_detection(self, output, target_sizes):
        scale = 2
        img_size = (self.size["height"], self.size["width"])
        out = output["hidden_states"]
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

            bboxes, scores = self.generate_bbox(keys[i], labels[i], score_maps[i], scales)
            results.append({"bboxes": bboxes, "scores": scores})
        final_results.update({"results": results})

        return results

    def generate_bbox(self, keys, label, score, scales):
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
            if score_i < self.min_score:
                label[ind] = 0
                continue

            if self.bbox_type == "rect":
                rect = cv2.minAreaRect(points[:, ::-1])
                alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
                rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
                bbox = cv2.boxPoints(rect) * scales

            elif self.bbox_type == "poly":
                binary = np.zeros(label.shape, dtype="uint8")
                binary[ind_np] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scales
            bbox = bbox.astype("int32")
            bboxes.append(bbox.reshape(-1).tolist())
            scores.append(score_i)
        return bboxes, scores
