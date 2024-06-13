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
"""Image processor class for MPLUGDocOwl."""

from typing import Dict, List, Optional, Union, Tuple
from einops import rearrange
import numpy as np
#FIXME change the import from transformers to import from ...
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging
from PIL import Image

logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL
    from PIL import Image


GRID_DICT = {
    'grid_1':[
        (1,1)],
    'grid_4':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1)],
    'grid_9':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)],
    'grid_3x3':[
        (3,3)],
    'grid_20':[
        (1, 1), 
        (1, 2), (2, 1), 
        (1, 3), (3, 1), (1, 4), (2, 2), (4, 1), 
        (1, 5), (5, 1), 
        (1, 6), (2, 3), (3, 2), (6, 1), 
        (1, 7), (7, 1), 
        (1, 8), (2, 4), (4, 2), (8, 1), 
        (1, 9), (3, 3), (9, 1), 
        (1, 10), (2, 5), (5, 2), (10, 1), 
        (1, 11), (11, 1), 
        (2, 6), (3, 4), (4, 3), (6, 2), 
        (2, 7), (7, 2), 
        (3, 5), (5, 3), 
        (2, 8), (4, 4), (8, 2), 
        (2, 9), (3, 6), (6, 3), (9, 2), 
        (2, 10), (4, 5), (5, 4), (10, 2)]
}
#FIXME write the documentation for these functions
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)
    print(area2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    print(iou)
    return iou, union

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    input_image_bbox = np.array([[0, 0, input_image_size[1], input_image_size[0]]])

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.copy()
    boxes3[:, 3] = input_image_size[0] / input_image_size[1] * anchors[:, 2]  # for resolution-independent iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = np.diag(shape_iou)  # Get diagonal for self-comparison
    index = np.argmax(shape_iou * 100 + iou)
    print(index)
    return index
#FIXME add this into shape adaptive cropping module

def anchor_resize(image:ImageInput,
                  anchors: str = 'grid_9', 
                  size:Dict[str, int] = None,
                  grid_dict: Dict[str, List[Tuple[int, int]]] = GRID_DICT,
                  resample=PILImageResampling.BICUBIC):
        # Convert anchors to xyxy format
        anchors = [tuple(_) for _ in grid_dict[anchors]] 
        size = size['width']
        anchors = np.array(
            [[0, 0, anchor[1] * size, anchor[0] * size]
             for anchor in anchors]
        )
        anchor_areas = box_area(anchors)
        
        # Resize image based on selected anchor
        selected_anchor = anchor_rank(anchors, anchor_areas, (image.size[1], image.size[0]))
        target_size = anchors[selected_anchor][2:].astype(int)  # target width, height
        resized_img = image.resize((target_size[0], target_size[1]), resample=resample)
        resized_img = np.array(resized_img)
       # image_patches_list = [image_input[i] for i in range(image_input.shape[0])]
        return [resized_img], selected_anchor
def shape_adaptive_cropping(image_patches: ImageInput,
                            size: Dict[str, int] = None, 
                            anchors: str = 'grid_9', 
                            grid_dict: Dict[str, List[Tuple[int, int]]] = GRID_DICT,
                            add_global_img: bool = True, 
                            selected_anchor: int = None,):
    
        anchors = [tuple(_) for _ in grid_dict[anchors]] 
        size = size['width']
        #self.anchors = [tuple(_) for _ in grid_dict[anchors]]
        anchor_max = max(max(_) for _ in anchors)
        #breakpoint()
        #image_patches, selected_anchor = anchor_resize(image, anchors, size, interpolation) #w,h
        #image_patches = image_patches.convert("RGB")

        h, w = image_patches.shape[0],image_patches.shape[1] #w,h
        
        image_patches = image_patches.transpose(2,0,1)

        anchor_size = anchors[selected_anchor]

        # Reshape the image
        num_h, num_w = anchor_size
        
        image_input = image_patches.reshape(3, num_h, size, num_w, size)
        # Step 2: Transpose to get the correct order
        image_input = image_input.transpose(1, 3, 2, 4, 0)
        image_input = image_input.reshape((-1,size,size,3))
        #image_input = image_input.transpose(0,2,3,1)
        image_patches_list = [image_input[i] for i in range(image_input.shape[0])]
        anchor = anchors[selected_anchor]  # w,h
        patch_position = np.concatenate([
            np.repeat(np.arange(anchor[0])[:, np.newaxis], anchor[1], axis=1)[:, :, np.newaxis],
            np.repeat(np.arange(anchor[1])[np.newaxis, :], anchor[0], axis=0)[:, :, np.newaxis]
        ], axis=2)
    
        patch_position = patch_position.reshape(-1, 2)
        if add_global_img:
            patch_position = np.vstack((np.ones((1, 2), dtype=np.int64) * anchor_max, patch_position))
          # num_patch, (ph, pw)
        return image_patches_list, patch_position, patch_position.shape[0], anchor_max

class MPLUGDocOwlImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MPLUGDocOwl image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = False,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_shape_adaptive_cropping: bool = True,
        do_anchor_resize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 448, "width": 448}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 448, "width": 448}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.do_shape_adaptive_cropping = do_shape_adaptive_cropping 
        self.do_anchor_resize = do_anchor_resize
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
        #self.adaptive_cropping_module = ShapeAdaptiveCroppingModule()
    def anchor_resize(self,
                image:ImageInput,
                size:Dict[str, int] = None,
                resample: PILImageResampling = PILImageResampling.BICUBIC):
        return anchor_resize(image=image, size=size, resample=resample)

    def adaptive_crop(
            self,
            image_patches: ImageInput,
            size: Dict[str, int] = None,
            selected_anchor: int = None,
        ):
        return shape_adaptive_cropping(image_patches=image_patches, size=size, selected_anchor=selected_anchor)

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
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size: int = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_shape_adaptive_cropping: bool = True,
        do_anchor_resize: bool = True,
        #shape_adaptive_cropping: bool = True,
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
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
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
        size = get_size_dict(size, param_name="size", default_to_square=True)
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_shape_adaptive_cropping = do_shape_adaptive_cropping if do_shape_adaptive_cropping is not None else self.do_shape_adaptive_cropping
        do_anchor_resize = do_anchor_resize if do_anchor_resize is not None else self.do_anchor_resize
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        # 1. Keep global image to be able to work with it later
         

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        patch_images = images.copy()
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_center_crop:
            images = [
                self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
            ]
        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]
       # breakpoint()
        if do_anchor_resize:
            output = [self.anchor_resize(image, size) for image in patch_images][0] 
            patch_images, selected_anchor = output[0], output[1]
            images.extend(patch_images)
           # breakpoint()
            
        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]
            
       # breakpoint()
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]
        if do_shape_adaptive_cropping:
            output = [self.adaptive_crop(image_patches=image, size=size, selected_anchor = selected_anchor) for image in images[1:]][0]
            patch_images, patch_positions, num_patches, anchor_max = output[0], output[1], output[2], output[3]

            del images[1:]
            images.extend(patch_images)
        
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

            # call the module

        data = {"pixel_values": images, "patch_positions": patch_positions, "num_patches": num_patches, "anchor_max": anchor_max}
        return BatchFeature(data=data, tensor_type=return_tensors)

#image_processor = MPLUGDocOwlImageProcessor()
#image = Image.open("/home/dana_aubakirova/test_image.tif")
#pixel_values = image_processor(image, do_rescale=False, do_convert_rgb=True, do_shape_adaptive_cropping=True, do_resize=True, do_normalize=True, return_tensors=TensorType.PYTORCH,image_mean=(0.48145466, 0.4578275, 0.40821073), image_std=(0.26862954, 0.26130258, 0.27577711),resample=None,size=224)
#breakpoint()
#print(pixel_values)

