from .fuyu_processing_utils import tokenize_prompts_with_images, construct_full_unpacked_stream, full_unpacked_stream_to_tensor

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
    normalize,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

from .image_processing_fuyu import FuyuImageProcessor

import numpy as np

import PIL.Image
import torch
import einops

from typing import List
import math

from torchvision.transforms import ConvertImageDtype, Normalize, Compose

# Copied from transformers.models.detr.image_processing_detr.max_across_indices


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]

# Copied from transformers.models.detr.image_processing_detr.get_max_height_width


def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)

# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask


def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


class FuyuProcessor():  # ProcessorMixin):
    r"""
    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    # attributes = ["image_processor", "tokenizer"]
    # image_processor_class = "FuyuImageProcessor"
    # tokenizer_class = "LlamaTokenizerFast"
    # FIXME How are these requirements propagated? currently getting AttributeError: module transformers has no attribute FuyuImageProcessor

    def __init__(self, image_processor, tokenizer):
        # super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer  # AutoTokenizer.from_pretrained(pretrained_path)
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.image_processor = FuyuImageProcessor()

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                if isinstance(text[0], list):
                    prompts = text
                else:
                    prompts = [text]
            batch_images = []
            if isinstance(images, list):
                image_unpadded_heights = []
                image_unpadded_widths = []
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
                    # FIXME add format CHW or HWC detection and cast here
                    image_unpadded_widths.append(image.shape[width_index])
                    image_unpadded_heights.append(image.shape[height_index])

                    # reproduct adept padding sampler
                    padded_image = self.image_processor.aspectratio_preserving_padding.apply_transformation(image)

                    # convert to tensor

                    tensor_img = torch.Tensor(padded_image).permute(2, 0, 1)
                    batch_images.append(tensor_img)
                image_unpadded_heights = torch.Tensor([image_unpadded_heights])
                image_unpadded_widths = torch.Tensor([image_unpadded_widths])
                batch_images = [batch_images]
            else:
                raise ValueError("images must be a list of ndarrays or PIL Images to be processed.")

            # Note: the original adept code has a handling of image_unpadded_h and w, but it doesn't seem to hold
            # when there are several different size subsequences per batch. The current implementation reflects
            # that limitation and should be documented.
            #
            first_image_sequence_length = len(batch_images[0])

            for image_sample, text_sample in zip(batch_images, prompts):
                # get length of sequences within a batch
                assert first_image_sequence_length == len(image_sample) == len(
                    text_sample), "The current implementation only supports batches with the same number of subsequences."

            self.subsequence_length = first_image_sequence_length

            assert len(prompts) == len(images)
            self.batch_size = len(batch_images)

            # FIXME max_tokens_to_generate is embedded into this processor's call.
            prompt_tokens, prompts_length = tokenize_prompts_with_images(tokenizer=self.tokenizer,
                                                                         prompts=prompts,
                                                                         transformed_images=[
                                                                             [tensor_img]],
                                                                         max_tokens_to_generate=self.max_tokens_to_generate,
                                                                         max_position_embeddings=self.max_position_embeddings,
                                                                         add_BOS=True,
                                                                         add_beginning_of_answer_token=True,
                                                                         rank=0)
            # same so far

            # FIXME the remainder of current image processing logic assumes batch_size = subsequence_size = 1.
            image_input = tensor_img

            # This is 1 if there is an image per subsequence, else 0. [batch, subsequence, presence]
            # the remainder of current image processing logic assumes batch_size = subsequence_size = 1.
            # Here it is OK as the model cannot handle > 1 subsequences
            # FIXME the image could be absent however and image presence should be inferred from user batch input
            image_present = torch.ones(self.batch_size, 1, 1)

            image_placeholder_id = self.tokenizer('|SPEAKER|', add_special_tokens=False)['input_ids'][1]
            image_newline_id = self.tokenizer('|NEWLINE|', add_special_tokens=False)['input_ids'][1]

            model_image_input = self.image_processor.process_images_for_model_input(
                image_input=image_input.unsqueeze(0).unsqueeze(0),
                image_present=image_present,
                image_unpadded_h=image_unpadded_heights,
                image_unpadded_w=image_unpadded_widths,
                image_patch_dim_h=30,
                image_patch_dim_w=30,
                image_placeholder_id=image_placeholder_id,
                image_newline_id=image_newline_id,
                variable_sized=True
            )

            image_padded_unpacked_tokens = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=prompt_tokens,
                image_tokens=model_image_input['image_input_ids'],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            # Construct inputs for image patch indices.
            unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=torch.full_like(prompt_tokens, -1),
                image_tokens=model_image_input['image_patch_indices_per_batch'],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
            max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
            all_bi_tokens_to_place = []
            for bi in range(self.batch_size):
                tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[bi].shape[0]))
                all_bi_tokens_to_place.append(tokens_to_place)

            image_padded_unpacked_tokens_tensor = full_unpacked_stream_to_tensor(
                all_bi_tokens_to_place=all_bi_tokens_to_place,
                full_unpacked_stream=image_padded_unpacked_tokens,
                fill_value=self.tokenizer.eos_token_id,
                batch_size=self.batch_size,
                new_seq_len=max_seq_len_batch,
                offset=0,
            )
            # Use same packing logic for the image patch indices.
            image_patch_input_indices = full_unpacked_stream_to_tensor(
                all_bi_tokens_to_place=all_bi_tokens_to_place,
                full_unpacked_stream=unpacked_image_patch_indices_per_batch,
                fill_value=-1,
                batch_size=self.batch_size,
                new_seq_len=max_seq_len_batch,
                offset=0,
            )
            return {"model_image_input": model_image_input,
                    "image_padded_unpacked_tokens": image_padded_unpacked_tokens,
                    "image_padded_unpacked_tokens_tensor": image_padded_unpacked_tokens_tensor,
                    "image_patch_input_indices": image_patch_input_indices}

        '''
        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
        '''

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]
