# coding=utf-8
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for DETR."""

from typing import List, Optional, Union, Dict

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import logging


logger = logging.get_logger(__name__)


class DetrFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a DETR feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.


    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int`, `optional`, defaults to 800):
            Resize the input to the given size. Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        max_size (:obj:`int`, `optional`, defaults to :obj:`1333`):
            The largest size an image dimension can have (otherwise it's capped). Only has an effect if
            :obj:`do_resize` is set to :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (:obj:`int`, defaults to :obj:`[0.485, 0.456, 0.406]s`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`int`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        padding_value (:obj:`float`, defaults to 0.0):
            The pixel value to use when padding images.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self, do_resize=True, size=800, max_size=1333, do_normalize=True, image_mean=None, image_std=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406] 
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225] 

    # inspired by https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/datasets/coco.py#L33
    # with added support for several TensorTypes
    def convert_coco_poly_to_mask(segmentations, height, width, as_tensor):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            # next, convert to appropriate TensorType and apply operations
            if as_tensor == tf.constant:
                mask = as_tensor(mask, dtype=tf.uint8)
                mask = tf.experimental.numpy.any(mask, axis=2)
                masks.append(mask)
                if masks:
                    masks = tf.stack(masks, axis=0)
                else:
                    masks = tf.zeros((0, height, width), dtype=tf.uint8)
            elif as_tensor == torch.tensor:
                mask = as_tensor(mask, dtype=torch.uint8)
                mask = mask.any(dim=2)
                masks.append(mask)
                if masks:
                    masks = torch.stack(masks, dim=0)
                else:
                    masks = torch.zeros((0, height, width), dtype=torch.uint8)
            elif as_tensor == jnp.array:
                mask = as_tensor(mask, dtype=jnp.uint8)
                mask = jnp.any(mask, axis=2)
                masks.append(mask)
                if masks:
                    masks = jnp.stack(masks, axis=0)
                else:
                    masks = jnp.zeros((0, height, width), dtype=jnp.uint8)
            else:
                mask = np.array(mask, dtype=np.uint8)
                mask = np.any(mask, axis=2)
                masks.append(mask)
                if masks:
                    masks = np.stack(masks, axis=0)
                else:
                    masks = np.zeros((0, height, width), dtype=np.uint8)

        return masks
        

    # inspired by https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/datasets/coco.py#L50
    # with added support for several TensorTypes
    def convertCocoToDetrFormat(self, image, target, return_masks=False, return_tensors: Optional[Union[str, TensorType]] = None):
        
        # First: set as_tensor
        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
        else:
            as_tensor = np.asarray
        
        # Second: convert COCO to DETR format
        
        w, h = image.size

        image_id = target["image_id"]
        image_id = as_tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = as_tensor(classes, dtype=torch.int64)

        if return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w, as_tensor)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = as_tensor([obj["area"] for obj in anno])
        iscrowd = as_tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = as_tensor([int(h), int(w)])
        target["size"] = as_tensor([int(h), int(w)])

        return image, target, as_tensor
    
    def _max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes
    
    def _resize(self, image, target, size, as_tensor, max_size=None):
        # size can be min_size (scalar) or (w, h) tuple

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (w, h) since we use PIL to resize images
                # so we revert the tuple 
                return get_size_with_aspect_ratio(image_size, size, max_size)[::-1]

        size = get_size(image.size, size, max_size)
        rescaled_image = self.resize(image, size=size)

        if target is None:
            return rescaled_image, None

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = size
        target["size"] = as_tensor([h, w])

        if "masks" in target:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

        return rescaled_image, target
    
    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        annotations = Union[List[Dict], List[List[Dict]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        .. warning::

           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.


        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            annotations (:obj:`List[Dict]`, :obj:`List[List[Dict]]`):
                The corresponding annotations in COCO format. Each image can have one or more annotations. Each annotation 
                is a Python dictionary, with the following keys: segmentation, area, iscrowd, image_id, bbox, category_id, id. 
            
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:


                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.s
                * :obj:`'jax'`: Return JAX :obj:`jnp.ndarray` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:


            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
            annotations = [annotations]

        # prepare (annotations -> target)
        if annotations is not None:
            for idx, image, anno in enumerate(zip(images, annotations)):
                # TODO update image_id
                target = {'image_id': anno[0]['image_id'], 'annotations': anno}
                image, target, as_tensor = self.convertCocoToDetrFormat(image, target, return_tensors=return_tensors)
                images[idx] = image
                annotations[idx] = target

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            if annotations is not None:
                for idx, image, target in enumerate(zip(images, annotations)):
                    image, target = self._resize(image=image, target=target, size=self.size, as_tensor=as_tensor, max_size=self.max_size)
                    images[idx] = image
                    annotations[idx] = target
            else:
                images = [self._resize(image=image, target=None, size=self.size, max_size=self.max_size)[0] for image in images]
                
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # create pixel_mask
        max_size = self._max_by_axis([list(image.shape) for image in images])
        batch_shape = [len(images)] + max_size
        b, c, h, w = batch_shape
        tensor = np.zeros(batch_shape)
        mask = np.zeros((b, h, w))
        for img, pad_img, m in zip(images, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = np.copy(img)
            mask[: img.shape[1], : img.shape[2]] = True
        
        # return as BatchFeature
        data = {"pixel_values": images, "pixel_mask": mask}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs