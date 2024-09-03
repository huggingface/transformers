# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL

GRID_DICT = {
    "grid_1": [(1, 1)],
    "grid_4": [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1)],
    "grid_9": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (2, 3),
        (3, 2),
        (1, 7),
        (7, 1),
        (4, 2),
        (2, 4),
        (1, 8),
        (8, 1),
        (3, 3),
        (1, 9),
        (9, 1),
    ],
    "grid_3x3": [(3, 3)],
    "grid_20": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (1, 4),
        (2, 2),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (2, 3),
        (3, 2),
        (6, 1),
        (1, 7),
        (7, 1),
        (1, 8),
        (2, 4),
        (4, 2),
        (8, 1),
        (1, 9),
        (3, 3),
        (9, 1),
        (1, 10),
        (2, 5),
        (5, 2),
        (10, 1),
        (1, 11),
        (11, 1),
        (2, 6),
        (3, 4),
        (4, 3),
        (6, 2),
        (2, 7),
        (7, 2),
        (3, 5),
        (5, 3),
        (2, 8),
        (4, 4),
        (8, 2),
        (2, 9),
        (3, 6),
        (6, 3),
        (9, 2),
        (2, 10),
        (4, 5),
        (5, 4),
        (10, 2),
    ],
}


def box_area(boxes):
    r"""
    Compute the area of each bounding box in a given set of bounding boxes.

    Args:
        boxes (np.ndarray): An array of shape (N, 4) containing N bounding boxes,
        boxes (`np.ndarray`): An array of shape (N, 4) containing N bounding boxes,
                            each represented by the coordinates [x_min, y_min, x_max, y_max].

    Returns:
        `np.ndarray`: An array of shape (N,) containing the area of each bounding box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, area1, boxes2, eps=1e-5):
    r"""
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (np.ndarray): An array of shape (N, 4) containing N bounding boxes.
        area1 (np.ndarray): An array of shape (N,) containing the area of each bounding box in boxes1.
        boxes2 (np.ndarray): An array of shape (M, 4) containing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
        boxes1 (`np.ndarray`): An array of shape (N, 4) containing N bounding boxes.
        area1 (`np.ndarray`): An array of shape (N,) containing the area of each bounding box in boxes1.
        boxes2 (`np.ndarray`): An array of shape (M, 4) containing M bounding boxes.
        eps (`float`, *optional*): A small value to avoid division by zero. Defaults to 1e-5.

    Returns:
        `tuple`: A tuple containing:
            - `np.ndarray`: An array of shape (N, M) containing the IoU between each pair of boxes from boxes1 and boxes2.
            - `np.ndarray`: An array of shape (N, M) containing the union areas of each pair of boxes.
    """
    area2 = box_area(boxes2)

    top_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = np.clip(bottom_right - top_left, a_min=0, a_max=None)  # [N,M,2]
    intersection = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - intersection

    iou = intersection / (union + eps)

    return iou, union


def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    r"""
    Rank anchors based on their IoU and shape-adaptive IoU with respect to an input image size.

    Args:
        anchors (np.ndarray): An array of shape (N, 4) containing N anchors.
        anchors_areas (np.ndarray): An array of shape (N,) containing the area of each anchor.
        input_image_size (tuple): A tuple (height, width) representing the size of the input image.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
        anchors (`np.ndarray`): An array of shape (N, 4) containing N anchors.
        anchors_areas (`np.ndarray`): An array of shape (N,) containing the area of each anchor.
        input_image_size (`tuple`): A tuple (height, width) representing the size of the input image.
        eps (`float`, *optional*, defaults to 1e-05): A small value to avoid division by zero. Defaults to 1e-5.

    Returns:
        `int`: The index of the selected anchor with the highest rank.

    """
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

    return index


def anchor_resize(
    image: ImageInput,
    anchors: str = "grid_9",
    size: Dict[str, int] = None,
    grid_dict: Dict[str, List[Tuple[int, int]]] = GRID_DICT,
    resample=PILImageResampling.BICUBIC,
):
    r"""
    Resize an image based on selected anchor and its associated size.

    Args:
        image (`ImageInput`): The input image to be resized.
        anchors (`str`, *optional*, defaults to "grid_9"): The key for selecting anchor sizes from the grid_dict. Defaults to "grid_9".
        size (`Dict[str, int]`, *optional*): A dictionary containing the target size for resizing. Defaults to None.
        grid_dict (`Dict[str, List[Tuple[int, int]]]`, *optional*): A dictionary containing the anchor grid configurations. Defaults to GRID_DICT.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`): The resampling method to use. Defaults to PILImageResampling.BICUBIC.
    Returns:
        tuple: A tuple containing:
            - List[np.ndarray]: A list containing the resized image.
            - int: The index of the selected anchor.
            - `List[np.ndarray]`: A list containing the resized image.
            - `int`: The index of the selected anchor.
    """
    # Convert anchors to xyxy format
    anchors = [tuple(_) for _ in grid_dict[anchors]]
    size = size["width"]
    anchors = np.array([[0, 0, anchor[1] * size, anchor[0] * size] for anchor in anchors])
    anchor_areas = box_area(anchors)

    # Resize image based on selected anchor
    selected_anchor = anchor_rank(anchors, anchor_areas, (image.size[1], image.size[0]))
    target_size = anchors[selected_anchor][2:].astype(int)  # target width, height
    resized_img = image.resize((target_size[0], target_size[1]), resample=resample)
    resized_img = np.array(resized_img)
    return (resized_img, selected_anchor)


def shape_adaptive_cropping(
    image_patches: ImageInput,
    size: Dict[str, int] = None,
    anchors: str = "grid_9",
    grid_dict: Dict[str, List[Tuple[int, int]]] = GRID_DICT,
    selected_anchor: int = None,
):
    r"""
    Performs shape-adaptive cropping on image patches based on selected anchor size.

    This function is designed to handle images with various aspect ratios and resolutions by cropping
    the image into multiple sub-images using a shape-adaptive grid. The goal is to preserve the resolution
    and aspect ratio as much as possible to prevent text blur and distortion, which is critical for tasks
    requiring visually-situated language understanding.

    Args:
        image_patches (ImageInput): The input image patches to be cropped.
        size (Dict[str, int], optional): A dictionary containing the target size for cropping. The size
                                         is expected to have a key "width". Defaults to None.
        anchors (str, optional): The key for selecting anchor sizes from the grid_dict. Defaults to "grid_9".
        grid_dict (Dict[str, List[Tuple[int, int]]], optional): A dictionary containing the anchor grid
                                                                configurations. Defaults to GRID_DICT.
        add_global_img (bool, optional): Whether to add the global image to the list of cropped patches.
                                         Defaults to True.
        selected_anchor (int, optional): The index of the selected anchor for cropping. If None, the
                                         function will select an anchor based on the shape-adaptive
                                         criteria. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - List[np.ndarray]: A list of cropped image patches.
            - np.ndarray: An array containing the positions of the patches.
            - int: The number of patches.
            - int: The maximum anchor size.

    Notes:
        The function first converts the input anchors to a format suitable for cropping. It then reshapes
        the image patches according to the selected anchor size. The resulting sub-images maintain the
        resolution and aspect ratio of the original image as much as possible.
        Find more details in the paper https://arxiv.org/pdf/2310.05126.

    Example:
        Consider:
        nh (int): Number of rows in the grid.
        nw (int): Number of columns in the grid.
        Hv (int): Height of the visual encoder input.
        Wv (int): Width of the visual encoder input.
        Nc (int): Maximum number of cells (sub-images) in the grid.

        The grid configurations and their selection are based on two main criteria:
        1. Resolution coherence (Srr): This measures the IoU between the input image resolution and the grid resolution.
           Srr(I, g) = IoU((H, W), (nh * Hv, nw * Wv))
        2. Shape similarity (Sra): This measures the IoU between the input image aspect ratio and the grid aspect ratio.
           Sra(I, g) = IoU((H, W), (nh, nw))

        The matched grid is selected by maximizing the matching score:
           g* = argmax (Sra(I, g) + Srr(I, g))

        After selecting the appropriate grid, the input image is resized to (nh * Hv, nw * Wv) and cropped into nh * nw local images.
        Additionally, to maintain the global structure information of the image, the input image is resized to (Hv, Wv) as a global image.

    """
    anchors = [tuple(_) for _ in grid_dict[anchors]]
    size = size["width"]

    anchor_max = max(max(_) for _ in anchors)

    image_patches = image_patches.transpose(2, 0, 1)

    anchor_size = anchors[selected_anchor]

    num_h, num_w = anchor_size

    image_input = image_patches.reshape(3, num_h, size, num_w, size)

    image_input = image_input.transpose(1, 3, 2, 4, 0)
    image_input = image_input.reshape((-1, size, size, 3))
    image_patches_list = [image_input[i] for i in range(image_input.shape[0])]
    anchor = anchors[selected_anchor]  # w,h
    patch_position = np.concatenate(
        [
            np.repeat(np.arange(anchor[0])[:, np.newaxis], anchor[1], axis=1)[:, :, np.newaxis],
            np.repeat(np.arange(anchor[1])[np.newaxis, :], anchor[0], axis=0)[:, :, np.newaxis],
        ],
        axis=2,
    )

    patch_position = patch_position.reshape(-1, 2)
    patch_position = np.vstack((np.ones((1, 2), dtype=np.int64) * anchor_max, patch_position))
    return image_patches_list, patch_position, patch_position.shape[0], anchor_max


def add_global_image(images, patch_images):
    """
    This function takes a list of global images and a list of lists containing patch images,
    and combines them such that each image is followed by its corresponding patch images.

    :param images: List of global images
    :param patch_images: List of lists of patch images corresponding to each image
    :return: A new list with images followed by their corresponding patch images
    """
    # Create a new list to store the combined elements
    combined_images = []

    # Combine elements
    for image, patches in zip(images, patch_images):
        combined_images.append(image)
        combined_images.extend(patches)

    return combined_images


class MPLUGDocOwlImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MPLUGDocOwlImageProcessor.

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
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to `False`):
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
        do_shape_adaptive_cropping (`bool`, *optional*, defaults to `True`): Whether to do a shape adaptive cropping of the input image. Should be only called if the do_anchor_resize is called.
        do_anchor_resize (`bool`, *optional*, defaults to `True`): Whether to resize the image based on the specified anchor. Should be called before do_shape_adaptive_cropping.
        do_add_global_image (`bool`, *optional*, defaults to `True`): Whether to add the global image to the image input.
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
        do_add_global_image: bool = True,
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
        self.do_add_global_image = do_add_global_image
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
            "do_shape_adaptive_cropping",
            "do_anchor_resize",
            "do_add_global_image",
        ]

    def anchor_resize(
        self, image: ImageInput, size: Dict[str, int] = None, resample: PILImageResampling = PILImageResampling.BICUBIC
    ):
        r"""
        Resizes an image using the specified anchor point and resampling method.

        Args:
            image (ImageInput): The image to be resized.
            size (Dict[str, int], optional): A dictionary specifying the desired width and height. Default is None.
            resample (PILImageResampling, optional): The resampling method to use. Default is PILImageResampling.BICUBIC.

        Returns:
            Image: The resized image.
        """
        return anchor_resize(image=image, size=size, resample=resample)

    def adaptive_crop(
        self,
        image_patches: ImageInput,
        size: Dict[str, int] = None,
        selected_anchor: int = None,
    ):
        r"""
        Performs adaptive cropping on image patches based on a selected anchor point.

        Args:
            image_patches (ImageInput): The image patches to be cropped.
            size (Dict[str, int], optional): A dictionary specifying the desired width and height. Default is None.
            selected_anchor (int, optional): The index of the selected anchor point. Default is None.

        Returns:
            Image: The cropped image patches.
        """
        return shape_adaptive_cropping(image_patches=image_patches, size=size, selected_anchor=selected_anchor)

    def add_global_image(
        self,
        images: List,
        patch_images: List,
    ):
        r"""
        Adds global image data to a list of patch images.

        Args:
            images (List): The list of images to which global image data will be added.
            patch_images (List): The list of patch images to be combined with the global image data.

        Returns:
            List: The combined list of images with global image data.
        """
        return add_global_image(images=images, patch_images=patch_images)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
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
        do_add_global_image: bool = True,
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
            sizeexi (`Dict[str, int]`, *optional*, defaults to `self.size`):
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
        do_shape_adaptive_cropping = (
            do_shape_adaptive_cropping if do_shape_adaptive_cropping is not None else self.do_shape_adaptive_cropping
        )
        do_anchor_resize = do_anchor_resize if do_anchor_resize is not None else self.do_anchor_resize
        do_add_global_image = do_add_global_image if do_add_global_image is not None else self.do_add_global_image
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

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if do_center_crop:
            images = [
                self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
            ]

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_anchor_resize:
            output = [self.anchor_resize(image, size) for image in patch_images]

        if do_shape_adaptive_cropping:
            output = [
                self.adaptive_crop(image_patches=image, size=size, selected_anchor=selected_anchor)
                for (image, selected_anchor) in output
            ]
            patch_images, patch_positions, num_patches, anchor_max = zip(*output)

        if do_add_global_image:
            images = self.add_global_image(images, patch_images)
        else:
            images = [patch for sublist in patch_images for patch in sublist]
            patch_positions = [pos[1:] for pos in patch_positions]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {
            "pixel_values": images,
            "patch_positions": patch_positions,
            "num_patches": num_patches,
            "anchor_max": anchor_max,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)
