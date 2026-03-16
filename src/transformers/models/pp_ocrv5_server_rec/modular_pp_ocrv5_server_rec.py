# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    ChannelDimension,
    ImageInput,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import TensorType
from ..auto import AutoConfig
from ..blip_2.modeling_blip_2 import Blip2Attention
from ..focalnet.modeling_focalnet import FocalNetMlp
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import PPOCRV5ServerDetPreTrainedModel
from ..resnet.modeling_resnet import ResNetConvLayer


logger = logging.get_logger(__name__)


@auto_docstring(
    checkpoint="PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    custom_args=r"""
    head_out_channels (`int`, *optional*, defaults to 18385):
        The number of output channels from the PPOCRV5ServerRecHead, responsible for final classification.
    """,
)
class PPOCRV5ServerRecConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_server_rec"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        hidden_act: str = "silu",
        hidden_size: int = 120,
        mlp_ratio: float = 2.0,
        depth: int = 2,
        head_out_channels: int = 18385,
        conv_kernel_size: list = [1, 3],
        qkv_bias: bool = True,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
                "stage_downsample": [True, True, True, True],
                "stage_downsample_strides": [[2, 1], [1, 2], [2, 1], [2, 1]],
            },
            **kwargs,
        )
        self.backbone_config = backbone_config

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.head_out_channels = head_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.qkv_bias = qkv_bias
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


@auto_docstring(custom_intro="ImageProcessor for the PP-OCRv5_server_rec model.")
class PPOCRV5ServerRecImageProcessor(BaseImageProcessor):
    r"""
    Constructs a PPOCRV5ServerRec image processor.

    Args:
        rec_image_shape (`List[int]`, *optional*, defaults to `[3, 48, 320]`):
            The target image shape for recognition in format [channels, height, width].
        max_img_width (`int`, *optional*, defaults to `3200`):
            The maximum width allowed for the resized image.
        character_list (`List[str]` or `str`, *optional*, defaults to `None`):
            The list of characters for text recognition decoding. If `None`, defaults to
            "0123456789abcdefghijklmnopqrstuvwxyz".
        use_space_char (`bool`, *optional*, defaults to `True`):
            Whether to include space character in the character list.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image pixel values to [0, 1] by dividing by 255.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image with mean=0.5 and std=0.5.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        rec_image_shape: list[int] = [3, 48, 320],
        max_img_width: int = 3200,
        character_list: list[str] | str | None = None,
        use_space_char: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rec_image_shape = rec_image_shape if rec_image_shape is not None else [3, 48, 320]
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

        # Initialize character list for decoding
        self._init_character_list(character_list, use_space_char)

    def _init_character_list(
        self,
        character_list: list[str] | str | None,
        use_space_char: bool,
    ) -> None:
        """
        Initialize the character list and character-to-index mapping for CTC decoding.

        Args:
            character_list (`List[str]` or `str`, *optional*):
                The list of characters or a string of characters. If `None`, defaults to
                "0123456789abcdefghijklmnopqrstuvwxyz".
            use_space_char (`bool`):
                Whether to include space character in the character list.
        """
        if character_list is None:
            characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
        elif isinstance(character_list, str):
            characters = list(character_list)
        else:
            characters = list(character_list)

        if use_space_char:
            characters.append(" ")

        # Add CTC blank token at the beginning
        characters = ["blank"] + characters

        self.character = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}

    def _pil_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to match cv2.resize with INTER_LINEAR as closely as possible.

        This implementation uses OpenCV's approach with vectorized operations:
        1. Float32 precision for all floating-point calculations
        2. Fixed-point arithmetic with 11-bit precision (scale = 2048)
        3. Vectorized bilinear interpolation for efficiency
        4. Proper boundary handling

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV's fixed-point arithmetic constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scale factors using float32 (matching OpenCV)
        scale_x = np.float32(w) / np.float32(target_w)
        scale_y = np.float32(h) / np.float32(target_h)

        # Pre-compute X interpolation tables (vectorized)
        dx_arr = np.arange(target_w, dtype=np.float32)
        fx_arr = (dx_arr + np.float32(0.5)) * scale_x - np.float32(0.5)
        sx_arr = np.floor(fx_arr).astype(np.int32)
        fx_frac_arr = fx_arr - sx_arr.astype(np.float32)

        # Handle X boundaries
        mask_left = sx_arr < 0
        mask_right = sx_arr >= w - 1
        sx_arr[mask_left] = 0
        fx_frac_arr[mask_left] = 0.0
        sx_arr[mask_right] = w - 2
        fx_frac_arr[mask_right] = 1.0

        xalpha = np.round(fx_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        xofs = sx_arr

        # Pre-compute Y interpolation tables (vectorized)
        dy_arr = np.arange(target_h, dtype=np.float32)
        fy_arr = (dy_arr + np.float32(0.5)) * scale_y - np.float32(0.5)
        sy_arr = np.floor(fy_arr).astype(np.int32)
        fy_frac_arr = fy_arr - sy_arr.astype(np.float32)

        # Handle Y boundaries
        mask_top = sy_arr < 0
        mask_bottom = sy_arr >= h - 1
        sy_arr[mask_top] = 0
        fy_frac_arr[mask_top] = 0.0
        sy_arr[mask_bottom] = h - 2
        fy_frac_arr[mask_bottom] = 1.0

        yalpha = np.round(fy_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        yofs = sy_arr

        # Create meshgrid for vectorized operations
        sy_grid = yofs[:, np.newaxis]  # (target_h, 1)
        sx_grid = xofs[np.newaxis, :]  # (1, target_w)
        ay_grid = yalpha[:, np.newaxis]  # (target_h, 1)
        ax_grid = xalpha[np.newaxis, :]  # (1, target_w)

        ay_inv = INTER_RESIZE_COEF_SCALE - ay_grid
        ax_inv = INTER_RESIZE_COEF_SCALE - ax_grid

        # Perform vectorized bilinear interpolation for each channel
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)

        for c in range(img.shape[2]):
            # Get 4 corner pixels using advanced indexing
            p00 = img[sy_grid, sx_grid, c].astype(np.int32)  # (target_h, target_w)
            p10 = img[sy_grid, sx_grid + 1, c].astype(np.int32)
            p01 = img[sy_grid + 1, sx_grid, c].astype(np.int32)
            p11 = img[sy_grid + 1, sx_grid + 1, c].astype(np.int32)

            # Vectorized bilinear interpolation
            val = ay_inv * (ax_inv * p00 + ax_grid * p10) + ay_grid * (ax_inv * p01 + ax_grid * p11)

            # Divide with rounding
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            output[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def _resize_norm_img(
        self,
        img: np.ndarray,
        max_wh_ratio: float,
        data_format: ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Resize and normalize a single image while maintaining aspect ratio.

        Args:
            img (`np.ndarray`):
                The input image in HWC format.
            max_wh_ratio (`float`):
                The maximum width-to-height ratio for resizing.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image.

        Returns:
            `np.ndarray`: The processed image in CHW format with padding.
        """
        img_c, img_h, img_w = self.rec_image_shape

        # Calculate target width based on max_wh_ratio
        target_w = int(img_h * max_wh_ratio)

        if target_w > self.max_img_width:
            # If target width exceeds max, resize to max width
            resized_image = self._pil_resize(img, (self.max_img_width, img_h))
            resized_w = self.max_img_width
            target_w = self.max_img_width
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(img_h * ratio) > target_w:
                resized_w = target_w
            else:
                resized_w = int(math.ceil(img_h * ratio))
            resized_image = self._pil_resize(img, (resized_w, img_h))

        # Convert to float32
        resized_image = resized_image.astype(np.float32)

        # Transpose to CHW format
        resized_image = resized_image.transpose((2, 0, 1))

        # Rescale to [0, 1]
        if self.do_rescale:
            resized_image = resized_image / 255.0

        # Normalize with mean=0.5, std=0.5
        if self.do_normalize:
            resized_image = (resized_image - 0.5) / 0.5

        # Create padded image
        padding_im = np.zeros((img_c, img_h, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def preprocess(
        self,
        img: ImageInput,
        rec_image_shape: list[int] | None = None,
        max_img_width: int | None = None,
        do_rescale: bool | None = None,
        do_normalize: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image for PPOCRV5ServerRec text recognition.

        Args:
            img (`ImageInput`):
                The input image to preprocess. Can be a PIL Image, numpy array, or torch tensor.
            rec_image_shape (`List[int]`, *optional*):
                The target image shape [channels, height, width]. Defaults to `self.rec_image_shape`.
            max_img_width (`int`, *optional*):
                The maximum width for the resized image. Defaults to `self.max_img_width`.
            do_rescale (`bool`, *optional*):
                Whether to rescale pixel values to [0, 1]. Defaults to `self.do_rescale`.
            do_normalize (`bool`, *optional*):
                Whether to normalize with mean=0.5 and std=0.5. Defaults to `self.do_normalize`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be "pt", "tf", "np", or None.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format of the output image.

        Returns:
            `BatchFeature`: A BatchFeature containing the processed `pixel_values`.
        """
        # Use instance defaults if not specified
        rec_image_shape = rec_image_shape if rec_image_shape is not None else self.rec_image_shape
        max_img_width = max_img_width if max_img_width is not None else self.max_img_width
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize

        # Store original values and temporarily update for processing
        original_rec_image_shape = self.rec_image_shape
        original_max_img_width = self.max_img_width
        original_do_rescale = self.do_rescale
        original_do_normalize = self.do_normalize

        self.rec_image_shape = rec_image_shape
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

        try:
            # Convert to numpy array
            img = np.array(img)

            # Get image dimensions
            img_c, img_h, img_w = self.rec_image_shape
            h, w = img.shape[:2]

            # Calculate max_wh_ratio dynamically
            base_wh_ratio = img_w / img_h
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(base_wh_ratio, wh_ratio)

            # Process the image
            processed_img = self._resize_norm_img(img, max_wh_ratio)

            # Add batch dimension
            processed_img = np.expand_dims(processed_img, axis=0)

            data = {"pixel_values": processed_img}
            return BatchFeature(data=data, tensor_type=return_tensors)

        finally:
            # Restore original values
            self.rec_image_shape = original_rec_image_shape
            self.max_img_width = original_max_img_width
            self.do_rescale = original_do_rescale
            self.do_normalize = original_do_normalize

    def _ctc_decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray,
        is_remove_duplicate: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Decode CTC output indices to text.

        Args:
            text_index (`np.ndarray`):
                The predicted character indices with shape (batch_size, sequence_length).
            text_prob (`np.ndarray`):
                The predicted character probabilities with shape (batch_size, sequence_length).
            is_remove_duplicate (`bool`, *optional*, defaults to `True`):
                Whether to remove duplicate consecutive characters.

        Returns:
            `List[Tuple[str, float]]`: A list of tuples containing (decoded_text, confidence_score).
        """
        result_list = []
        ignored_tokens = [0]  # CTC blank token
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def post_process_text_recognition(
        self,
        pred: np.ndarray,
    ) -> tuple[list[str], list[float]]:
        """
        Post-process the model output to decode text recognition results.

        Args:
            pred (`np.ndarray`):
                The model output predictions. Expected shape is (batch_size, sequence_length, num_classes)
                or a list/tuple containing such an array.

        Returns:
            `Tuple[List[str], List[float]]`: A tuple containing:
                - texts: List of decoded text strings.
                - scores: List of confidence scores for each decoded text.
        """
        preds = np.array(pred[0].detach().cpu())
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)

        text = self._ctc_decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
        )

        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])

        return texts, scores


@auto_docstring(custom_intro="FastImageProcessor for the PP-OCRv5_mobile_rec model.")
class PPOCRV5ServerRecImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast PPOCRV5ServerRec image processor that supports batch processing.

    This processor is designed to handle multiple images efficiently while maintaining
    strict compatibility with [`PPOCRV5ServerRecImageProcessor`]. The preprocessing
    results are guaranteed to be identical to the non-fast version.

    Args:
        rec_image_shape (`List[int]`, *optional*, defaults to `[3, 48, 320]`):
            The target image shape for recognition in format [channels, height, width].
        max_img_width (`int`, *optional*, defaults to `3200`):
            The maximum width allowed for the resized image.
        character_list (`List[str]` or `str`, *optional*, defaults to `None`):
            The list of characters for text recognition decoding. If `None`, defaults to
            "0123456789abcdefghijklmnopqrstuvwxyz".
        use_space_char (`bool`, *optional*, defaults to `True`):
            Whether to include space character in the character list.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image pixel values to [0, 1] by dividing by 255.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image with mean=0.5 and std=0.5.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The mean values for image normalization. Used for validation but actual
            normalization uses fixed value 0.5 in `_resize_norm_img`.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The standard deviation values for image normalization. Used for validation
            but actual normalization uses fixed value 0.5 in `_resize_norm_img`.

    Examples:

    ```python
    >>> from PIL import Image
    >>> from transformers import PPOCRV5ServerRecImageProcessorFast

    >>> processor = PPOCRV5ServerRecImageProcessorFast()

    >>> # Process a single image
    >>> image = Image.open("text_image.png")
    >>> inputs = processor(image, return_tensors="pt")

    >>> # Process multiple images in batch
    >>> images = [Image.open(f"text_image_{i}.png") for i in range(4)]
    >>> batch_inputs = processor(images, return_tensors="pt")
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        rec_image_shape: list[int] | None = None,
        max_img_width: int = 3200,
        character_list: list[str] | str | None = None,
        use_space_char: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rec_image_shape = rec_image_shape if rec_image_shape is not None else [3, 48, 320]
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        # Set default image_mean and image_std for normalization (mean=0.5, std=0.5)
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        # Initialize character list for decoding
        self._init_character_list(character_list, use_space_char)

    def _init_character_list(
        self,
        character_list: list[str] | str | None,
        use_space_char: bool,
    ) -> None:
        """
        Initialize the character list and character-to-index mapping for CTC decoding.

        Args:
            character_list (`List[str]` or `str`, *optional*):
                The list of characters or a string of characters. If `None`, defaults to
                "0123456789abcdefghijklmnopqrstuvwxyz".
            use_space_char (`bool`):
                Whether to include space character in the character list.
        """
        if character_list is None:
            characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
        elif isinstance(character_list, str):
            characters = list(character_list)
        else:
            characters = list(character_list)

        if use_space_char:
            characters.append(" ")

        # Add CTC blank token at the beginning
        characters = ["blank"] + characters

        self.character = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}

    def _pil_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to exactly match cv2.resize behavior using fixed-point arithmetic.

        This implementation uses fixed-point arithmetic (matching OpenCV's internal approach)
        with float32 precision for coordinate calculations to achieve maximum compatibility.

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV uses fixed-point arithmetic with these constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scaling factors using float32 (like cv2)
        scale_x = np.float32(w) / np.float32(target_w)
        scale_y = np.float32(h) / np.float32(target_h)

        # Create coordinate grids for output image using float32
        out_y, out_x = np.meshgrid(
            np.arange(target_h, dtype=np.float32), np.arange(target_w, dtype=np.float32), indexing="ij"
        )

        # Map output coordinates to input coordinates (pixel-center alignment) using float32
        src_x = (out_x + np.float32(0.5)) * scale_x - np.float32(0.5)
        src_y = (out_y + np.float32(0.5)) * scale_y - np.float32(0.5)

        # Clip to valid range
        src_x = np.clip(src_x, np.float32(0), np.float32(w - 1))
        src_y = np.clip(src_y, np.float32(0), np.float32(h - 1))

        # Get integer parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        # Calculate fractional parts (keep in float32)
        fx = src_x - x0.astype(np.float32)
        fy = src_y - y0.astype(np.float32)

        # Convert to fixed-point (with rounding, matching cv2's behavior)
        wx = np.round(fx * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        wy = np.round(fy * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)

        # Clamp to valid range
        wx = np.minimum(wx, INTER_RESIZE_COEF_SCALE)
        wy = np.minimum(wy, INTER_RESIZE_COEF_SCALE)

        # Calculate the four interpolation weights in fixed-point
        w0 = (INTER_RESIZE_COEF_SCALE - wx) * (INTER_RESIZE_COEF_SCALE - wy)
        w1 = wx * (INTER_RESIZE_COEF_SCALE - wy)
        w2 = (INTER_RESIZE_COEF_SCALE - wx) * wy
        w3 = wx * wy

        # Perform bilinear interpolation for each channel using fixed-point arithmetic
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)

        for c in range(img.shape[2]):
            # Get the four corner pixel values (as int32 for fixed-point math)
            Ia = img[y0, x0, c].astype(np.int32)
            Ib = img[y0, x1, c].astype(np.int32)
            Ic = img[y1, x0, c].astype(np.int32)
            Id = img[y1, x1, c].astype(np.int32)

            # Fixed-point interpolation
            val = w0 * Ia + w1 * Ib + w2 * Ic + w3 * Id

            # Divide by INTER_RESIZE_COEF_SCALE^2 with rounding
            # This is equivalent to: (val + (1 << 21)) >> 22
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            # Clamp to [0, 255]
            output[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def _resize_norm_img(
        self,
        img: np.ndarray,
        max_wh_ratio: float,
        data_format: ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Resize and normalize a single image while maintaining aspect ratio.

        This method uses PIL-based resizing instead of cv2 while maintaining
        consistent preprocessing results with [`PPOCRV5ServerRecImageProcessor`].

        Args:
            img (`np.ndarray`):
                The input image in HWC format.
            max_wh_ratio (`float`):
                The maximum width-to-height ratio for resizing.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image.

        Returns:
            `np.ndarray`: The processed image in CHW format with padding.
        """
        img_c, img_h, img_w = self.rec_image_shape

        # Calculate target width based on max_wh_ratio
        target_w = int(img_h * max_wh_ratio)

        if target_w > self.max_img_width:
            # If target width exceeds max, resize to max width
            resized_image = self._pil_resize(img, (self.max_img_width, img_h))
            resized_w = self.max_img_width
            target_w = self.max_img_width
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(img_h * ratio) > target_w:
                resized_w = target_w
            else:
                resized_w = int(math.ceil(img_h * ratio))
            resized_image = self._pil_resize(img, (resized_w, img_h))

        # Convert to float32
        resized_image = resized_image.astype(np.float32)

        # Transpose to CHW format
        resized_image = resized_image.transpose((2, 0, 1))

        # Rescale to [0, 1]
        if self.do_rescale:
            resized_image = resized_image / 255.0

        # Normalize with mean=0.5, std=0.5
        if self.do_normalize:
            resized_image = (resized_image - 0.5) / 0.5

        # Create padded image
        padding_im = np.zeros((img_c, img_h, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess a batch of images for text recognition.

        Args:
            images (`List[torch.Tensor]`):
                List of images to preprocess.
            **kwargs:
                Additional keyword arguments.

        Returns:
            `BatchFeature`: A dictionary containing the processed pixel values.
        """
        # Convert torch tensors to numpy arrays in HWC format
        np_images = []
        for img in images:
            # img is a torch.Tensor in CHW format, convert to HWC numpy array
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)
            np_images.append(img_np)

        # Calculate max width-to-height ratio across all images
        for img in np_images:
            imgC, imgH, imgW = self.rec_image_shape
            max_wh_ratio = imgW / imgH
            h, w = img.shape[:2]
            wh_ratio = w / float(h)
            max_wh_ratio = max(max_wh_ratio, wh_ratio)

        # Process each image
        processed_images = []
        for img in np_images:
            processed_img = self._resize_norm_img(
                img,
                max_wh_ratio=max_wh_ratio,
            )
            processed_images.append(processed_img)

        # Stack into batch tensor
        pixel_values = np.stack(processed_images, axis=0)
        pixel_values = torch.from_numpy(pixel_values)

        return BatchFeature(data={"pixel_values": pixel_values})

    def _ctc_decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray,
        is_remove_duplicate: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Decode CTC output indices to text.

        This method is identical to the one in [`PPOCRV5ServerRecImageProcessor`] to ensure
        consistent decoding results.

        Args:
            text_index (`np.ndarray`):
                The predicted character indices with shape (batch_size, sequence_length).
            text_prob (`np.ndarray`):
                The predicted character probabilities with shape (batch_size, sequence_length).
            is_remove_duplicate (`bool`, *optional*, defaults to `True`):
                Whether to remove duplicate consecutive characters.

        Returns:
            `List[Tuple[str, float]]`: A list of tuples containing (decoded_text, confidence_score).
        """
        result_list = []
        ignored_tokens = [0]  # CTC blank token
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def post_process_text_recognition(
        self,
        pred: np.ndarray,
    ) -> tuple[list[str], list[float]]:
        """
        Post-process the model output to decode text recognition results.

        This method is identical to the one in [`PPOCRV5ServerRecImageProcessor`] to ensure
        consistent post-processing behavior.

        Args:
            pred (`np.ndarray`):
                The model output predictions. Expected shape is (batch_size, sequence_length, num_classes)
                or a list/tuple containing such an array.

        Returns:
            `Tuple[List[str], List[float]]`: A tuple containing:
                - texts: List of decoded text strings.
                - scores: List of confidence scores for each decoded text.
        """
        preds = np.array(pred[0].detach().cpu())
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)

        text = self._ctc_decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
        )

        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])

        return texts, scores


class PPOCRV5ServerRecBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        mlp_ratio = config.mlp_ratio

        self.mixer = PPOCRV5ServerRecAttention(config)
        self.mlp = PPOCRV5ServerRecMlp(
            config=config,
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mixer(self.norm1(hidden_states))[0]
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class PPOCRV5ServerRecAttention(Blip2Attention):
    pass


class PPOCRV5ServerRecConvLayer(ResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
        )


class PPOCRV5ServerRecHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.backbone_config.stage_out_channels[-1]
        self.ctc_encoder = PPOCRV5ServerRecEncoderWithSVTR(in_channels, config)
        self.ctc_head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.ctc_encoder(hidden_states)
        hidden_states = hidden_states.squeeze(2).permute(0, 2, 1)
        hidden_states = self.ctc_head(hidden_states)
        hidden_states = F.softmax(hidden_states, dim=2)

        return hidden_states


class PPOCRV5ServerRecMlp(FocalNetMlp):
    pass


class PPOCRV5ServerRecEncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        config,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.conv1 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=config.conv_kernel_size,
        )
        self.conv2 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels // 8, out_channels=hidden_size, kernel_size=(1, 1)
        )

        self.svtr_block = nn.ModuleList()
        for _ in range(config.depth):
            self.svtr_block.append(PPOCRV5ServerRecBlock(config=config))

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.conv3 = PPOCRV5ServerRecConvLayer(in_channels=hidden_size, out_channels=in_channels, kernel_size=(1, 1))
        self.conv4 = PPOCRV5ServerRecConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels // 8,
            kernel_size=config.conv_kernel_size,
        )

        self.conv5 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels // 8, out_channels=hidden_size, kernel_size=(1, 1)
        )

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            hidden_states = blk(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(torch.cat((residual, hidden_states), dim=1))
        hidden_states = self.conv5(hidden_states)

        return hidden_states


class PPOCRV5ServerRecPreTrainedModel(PPOCRV5ServerDetPreTrainedModel):
    pass


@auto_docstring(custom_intro="PPOCRV5ServerRec model, consisting of Backbone and Head networks.")
class PPOCRV5ServerRecModel(PPOCRV5ServerRecPreTrainedModel):
    def __init__(self, config: PPOCRV5ServerRecConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        # PP-OCRv5_server_rec needs to modify the stride for the HGNetV2 layers
        self.backbone.embedder.stem3.convolution.stride = (1, 1)
        for idx, stride in enumerate(self.backbone.config.stage_downsample_strides):
            self.backbone.encoder.stages[idx].downsample.convolution.stride = stride

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.backbone(pixel_values, **kwargs)
        hidden_state = outputs.feature_maps[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(custom_intro="PPOCRV5ServerRec model for text recognition tasks.")
class PPOCRV5ServerRecForTextRecognition(PPOCRV5ServerRecPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5ServerRecConfig):
        super().__init__(config)
        self.model = PPOCRV5ServerRecModel(config)
        self.head = PPOCRV5ServerRecHead(config)

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.model(pixel_values, **kwargs)
        logits = self.head(outputs.last_hidden_state)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "PPOCRV5ServerRecForTextRecognition",
    "PPOCRV5ServerRecImageProcessor",
    "PPOCRV5ServerRecImageProcessorFast",
    "PPOCRV5ServerRecConfig",
    "PPOCRV5ServerRecModel",
    "PPOCRV5ServerRecPreTrainedModel",
]
