import math
from typing import List, Union, Dict

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import (
    normalize,
    pad,
    resize,
)
from ...image_utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    to_numpy_array,
    make_list_of_images
)
from ...utils import is_torch_available, is_vision_available, logging, requires_backends


if is_vision_available():
    import PIL

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always img_h ........................................... 1080 img_w
        ........................................... 1920 Then, it patches up these images using the patchify_image
        function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.

    """

    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(
        self,
        patch_size: Dict[str, int] = None,
        target_height: int = 1080,
        target_width: int = 1920,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_width = target_width
        self.target_height = target_height
        self.patch_size = patch_size if patch_size is not None else {"height": 30, "width": 30}
        self.padding_value = padding_value
        self.padding_mode = padding_mode

    def _scale_to_target_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width = get_image_size(image)
        if image_width <= self.target_width and image_height <= self.target_height:
            return image

        height_scale_factor = self.target_height / image_height
        width_scale_factor = self.target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = resize(image=image, size=(new_height, new_width))
        return np.array(scaled_image)

    def _pad_to_target_size(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image.shape

        padding_top = 0
        padding_left = 0
        padding_bottom = self.target_height - image_height
        padding_right = self.target_width - image_width
        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=self.padding_mode,
            constant_values=self.padding_value,
        )
        return padded_image

    def scale_pad_normalize(self, image: Union[np.ndarray, PIL.Image.Image]) -> np.ndarray:
        if isinstance(image, PIL.Image.Image):
            image = to_numpy_array(image)
        scaled_image = self._scale_to_target_aspect_ratio(image)
        padded_image = self._pad_to_target_size(scaled_image)
        normalized_padded_image = normalize(padded_image, 0.5, 0.5)
        return normalized_padded_image

    def get_num_patches(self, image_height: int, image_width: int) -> int:
        """Calculate number of patches required to encode an image."""
        if image_height % self.patch_size["height"] != 0:
            raise ValueError(f'{image_height=} must be divisible by {self.patch_size["height"]}')
        if image_width % self.patch_size["width"] != 0:
            raise ValueError(f'{image_width=} must be divisible by {self.patch_size["width"]}')

        num_patches_per_dim_h = image_height // self.patch_size["height"]
        num_patches_per_dim_w = image_width // self.patch_size["width"]
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w

        return num_patches

    def patchify_image(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Convert an image into a tensor of patches.

        Args:
            image: Image to convert. Shape: [batch, channels, height, width]
        """
        requires_backends(self, ["torch"])

        # TODO refer to https://github.com/ArthurZucker/transformers/blob/0f0a3fe5ca5697ee58faeb5b53f049af720b5e98/src/transformers/models/vit_mae/modeling_vit_mae.py#L871
        # torch implementation is faster but does not handle non-squares

        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, self.patch_size['height'], self.patch_size['height'])
        patches = unfolded_along_height.unfold(3, self.patch_size["width"], self.patch_size["width"])

        patches_reshaped = patches.contiguous().view(
            batch_size, channels, -1, self.patch_size['height'], self.patch_size["width"])

        patches_final = patches_reshaped.permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, channels * self.patch_size['height'] * self.patch_size["width"]
        )

        return patches_final

    def preprocess(self, images):
        """Utility function to preprocess the images and extract necessary information about original formats."""
        batch_images = []
        image_unpadded_heights = []
        image_unpadded_widths = []
        images = make_list_of_images(images)
        for image in images:
            image = to_numpy_array(image)
            if not is_scaled_image(image):
                image = image / 255.0
            channel_dimension = infer_channel_dimension_format(image, 3)
            if channel_dimension == ChannelDimension.FIRST:
                width_index = 2
                height_index = 1
            elif channel_dimension == ChannelDimension.LAST:
                width_index = 1
                height_index = 0

            image_unpadded_widths.append([image.shape[width_index]])
            image_unpadded_heights.append([image.shape[height_index]])

            # Reproduct adept padding sampler
            padded_image = self.scale_pad_normalize(image)
            tensor_img = torch.Tensor(padded_image).permute(2, 0, 1)
            batch_images.append([tensor_img])

        return batch_images, torch.Tensor(image_unpadded_heights), torch.Tensor(image_unpadded_widths)

    def postprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
    ) -> dict:
        """Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input: [batch_size, subsequence_size, c, h, w] tensor of images padded to model input size.
            image_present: [batch_size, subsequence_size] tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h: [batch_size, subsequence_size] tensor of unpadded image heights.
            image_unpadded_w: [batch_size, subsequence_size] tensor of unpadded image widths.
            image_placeholder_id: The id of the image placeholder token. Comes from an associated tokenizer.
            image_newline_id: The id of the image newline token. Comes from an associated tokenizer.
            variable_sized: Whether to process images as variable-sized.
        """
        requires_backends(self, ["torch"])
        # Only images that are present.
        images: List[List[torch.Tensor]] = []
        image_patches: List[List[torch.Tensor]] = []
        # Image input ids for every subsequence, including ones with no image present.
        image_input_ids: List[List[torch.Tensor]] = []
        for batch_index in range(image_input.shape[0]):
            images.append([])
            image_input_ids.append([])
            image_patches.append([])
            for subseq_index in range(image_input.shape[1]):
                if image_present[batch_index, subseq_index]:
                    image = image_input[batch_index, subseq_index]
                    if variable_sized:
                        # The min() is required here due to floating point issues:
                        # math.ceil(torch.tensor(300).cuda() / 30) == 11
                        new_h = min(
                            image.shape[1], math.ceil(image_unpadded_h[batch_index, subseq_index] /
                                                      self.patch_size["height"]) * self.patch_size["height"]
                        )
                        new_w = min(
                            image.shape[2], math.ceil(image_unpadded_w[batch_index, subseq_index] /
                                                      self.patch_size["width"]) * self.patch_size["width"]
                        )
                        image = image[:, :new_h, :new_w]
                    images[batch_index].append(image)
                    num_patches = self.get_num_patches(
                        image_height=image.shape[1],
                        image_width=image.shape[2],
                    )
                    tensor_of_image_ids = torch.full([num_patches], image_placeholder_id,
                                                     dtype=torch.int32, device=image_input.device)
                    patches = self.patchify_image(
                        image=image.unsqueeze(0)
                    ).squeeze(0)
                    if variable_sized:
                        # Now terminate each line with |NEWLINE|.
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1, new_w // self.patch_size["width"])
                        tensor_of_image_ids = torch.cat(
                            [
                                tensor_of_image_ids,
                                torch.full(
                                    [tensor_of_image_ids.shape[0], 1], image_newline_id, dtype=torch.int32, device=image_input.device
                                ),
                            ],
                            dim=1,
                        )
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1)
                    image_input_ids[batch_index].append(tensor_of_image_ids)
                    image_patches[batch_index].append(patches)
                else:
                    image_input_ids[batch_index].append(torch.tensor([], dtype=torch.int32, device=image_input.device))

        # Create image_patch_input_indices, where non-negative values correspond to image patches to be inserted in
        # the stream.
        image_patch_indices_per_batch: List[List[torch.Tensor]] = []
        image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []
        for batch_index in range(len(image_input_ids)):
            image_patch_indices_per_batch.append([])
            image_patch_indices_per_subsequence.append([])
            index_offset = 0
            for subseq_index in range(len(image_input_ids[batch_index])):
                # Indices of image patches.
                num_patches = torch.count_nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id)
                indices = torch.arange(
                    num_patches,
                    dtype=image_input_ids[batch_index][subseq_index].dtype,
                    device=image_input_ids[batch_index][subseq_index].device,
                )

                # Place those indices in the image input ids token stream, with -1 representing non-index tokens.
                indices_in_stream_per_batch = torch.full_like(image_input_ids[batch_index][subseq_index], -1)
                indices_in_stream_per_subsequence = torch.full_like(image_input_ids[batch_index][subseq_index], -1)
                indices_in_stream_per_batch[
                    torch.nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id, as_tuple=True)[0]
                ] = (indices + index_offset)
                indices_in_stream_per_subsequence[
                    torch.nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id, as_tuple=True)[0]
                ] = indices

                image_patch_indices_per_batch[batch_index].append(indices_in_stream_per_batch)
                image_patch_indices_per_subsequence[batch_index].append(indices_in_stream_per_subsequence)
                index_offset += num_patches

        return {
            "images": images,
            "image_input_ids": image_input_ids,
            "image_patches": image_patches,
            "image_patch_indices_per_batch": image_patch_indices_per_batch,
            "image_patch_indices_per_subsequence": image_patch_indices_per_subsequence,
        }
