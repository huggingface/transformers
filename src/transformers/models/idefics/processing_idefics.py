# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processor class for IDEFICS.
"""

import warnings
from typing import Callable, List, Optional, Union
from urllib.parse import urlparse

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, is_tf_available, is_torch_available


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

IMAGE_TOKEN = "<image>"


# copied from m4.training.packing
def incremental_to_binary_attention_mask(incremental_mask, return_tensors, num_classes=-1):
    # Set elements >= num_classes to -1
    if num_classes != -1:
        if return_tensors == "pt":
            incremental_mask[incremental_mask >= num_classes] = -1
        elif return_tensors == "tf":
            incremental_mask = tf.where(incremental_mask >= num_classes, -1, incremental_mask)

    # Create mask for negative values
    if return_tensors == "pt":
        negatives = incremental_mask == -1
        incremental_mask[negatives] = 0
        attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
        attn_mask[negatives, :] = 0
    elif return_tensors == "tf":
        negatives = tf.equal(incremental_mask, -1)
        incremental_mask = tf.where(negatives, 0, incremental_mask)
        attn_mask = tf.one_hot(incremental_mask, depth=num_classes)
        # Reshape 'negatives' to add an extra dimension, making it [batch_size, seq_length, 1]
        negatives_expanded = tf.expand_dims(negatives, -1)
        attn_mask = tf.where(negatives_expanded, tf.zeros_like(attn_mask), attn_mask)

    return attn_mask


# copied from m4.training.packing
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer, return_tensors):
    if return_tensors == "pt":
        return image_attention_mask_for_packed_input_ids_pt(input_ids, tokenizer)
    elif return_tensors == "tf":
        return image_attention_mask_for_packed_input_ids_tf(input_ids, tokenizer)


def image_attention_mask_for_packed_input_ids_pt(input_ids, tokenizer):
    image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx, token_id in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count

            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1

            if token_id == eod_token_id:
                seen_eod = True

    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            token_id = input_ids[batch_idx][idx]
            if token_id == image_token_id:
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                next_image_attention_mask[batch_idx][idx] = count

            if token_id == eod_token_id:
                seen_eod = True

            if seen_eod:
                next_image_attention_mask[batch_idx][idx] = -1

        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1

    return image_attention_mask, next_image_attention_mask


def image_attention_mask_for_packed_input_ids_tf(input_ids, tokenizer):
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    batch_size = tf.shape(input_ids)[0]
    image_attention_mask = tf.fill(tf.shape(input_ids), -1)
    next_image_attention_mask = tf.fill(tf.shape(input_ids), -1)

    for batch_idx in range(batch_size):
        count = -1
        seen_eod = False
        seq_length = tf.shape(input_ids)[1]

        for idx in range(seq_length - 1, -1, -1):
            token_id = input_ids[batch_idx, idx].numpy()
            if token_id == image_token_id:
                count += 1
                indices = [[batch_idx, idx]]
                updates = [count]
                image_attention_mask = tf.tensor_scatter_nd_update(image_attention_mask, indices, updates)
                next_image_attention_mask = tf.tensor_scatter_nd_update(next_image_attention_mask, indices, updates)
            elif token_id == eod_token_id and not seen_eod:
                seen_eod = True
                count = 0
                indices = [[batch_idx, idx]]
                updates = [count]
                next_image_attention_mask = tf.tensor_scatter_nd_update(next_image_attention_mask, indices, updates)
            if seen_eod and token_id != eod_token_id:
                indices = [[batch_idx, idx]]
                updates = [-1]
                next_image_attention_mask = tf.tensor_scatter_nd_update(next_image_attention_mask, indices, updates)
    return image_attention_mask, next_image_attention_mask


def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class IdeficsProcessor(ProcessorMixin):
    r"""
    Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224): Image size (assuming a square image)
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["image_size", "add_end_of_utterance_token"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        self.tokenizer_was_trained_with_end_of_utterance_token = (
            True
            if "<end_of_utterance>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            else False
        )

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        prompts: Optional[Union[List[TextInput], List[List[TextInput]]]] = None,
        transform: Callable = None,
        add_eos_token=False,
        add_end_of_utterance_token=None,
        debug=False,
        **kwargs,
    ) -> BatchEncoding:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            prompts (`Union[List[TextInput], [List[List[TextInput]]]]`):
                either a single prompt or a batched list of prompts - see the detailed description immediately after
                the end of the arguments doc section.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `"longest"`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'`: No padding. This will raise an error if the input sequences are of different
                  lengths.
                Note: Unlike most processors, which set padding=`False` by default, `IdeficsProcessor` sets `padding="longest"`
                  by default. See https://github.com/huggingface/transformers/pull/29449#pullrequestreview-1925576061 for why.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            transform (`Callable`, *optional*):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple functions. If `None` a preset inference-specific
                set of transforms will be applied to the images
            add_eos_token (`bool`, *optional*, defaults to `False`):
                Adds `eos_token` at the end of the final prompt if True`
            add_end_of_utterance_token (`bool`, *optional*)
                Whether to automatically add `<end_of_utterance>` after each prompt's text input (unless followed by an
                image). If `None` the tokenizer will be checked instead and if this token is found in
                `additional_special_tokens` then the value will be `True`.
            debug (`bool`, *optional*, defaults to `False`):
                `True` value will help debug prompt generation by dumping useful information
            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`):
                The type of tensors to return. Can be one of:
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.

        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

        Detailed explanation:

        Each entry in `prompts` is either a text to be passed as is or an image that will be processed.

        An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

        When the processor encounters an image it'll inject `<fake_token_around_image><image><fake_token_around_image>`
        entry into the prompt.

        Example:

        ```python
        checkpoint = "HuggingFaceM4/idefics-9b"
        processor = AutoProcessor.from_pretrained(checkpoint)
        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
        img = processor.image_processor.fetch_images([url])[0]

        prompts = [
            "User:",
            img,
            "Describe this image.\nAssistant: An image of two kittens in grass.\n",
            "User:",
            "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
            "Describe this image.\nAssistant:",
        ]

        inputs = processor(prompts, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```

        In this example the `prompts` will be converted into:

        ```
        <s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant: An image of two kittens in grass.
        User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant:'
        ```

        and the two images will be massaged using [`IdeficsImageProcessor.__call__`] method and placed inside the
        `pixel_values` dict entry of the return value.

        This example also examplifies that images can be passed as objects or as text urls. It can be seen that the
        first image is passed as object and the second one as a url.

        To do training do:

        ```python
        image_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        inputs = processor(prompts, transform=image_transform, return_tensors="pt")
        ```

        In order to help debug prompt generation enable `debug=True` which will show you what's happening.

        """
        warnings.warn("The order of the arguments in the __call__ function have been changed from previous versions.")
        if prompts is None:
            # if the user didn't specify prompts=prompts in the call, we assume they want to use the old behavior with prompts as a first argument
            prompts = images
        elif prompts is None and (images is not None and text is not None):
            # Assuming image-text-to-text behavior: one prompt for all images
            # Check if batched images are provided
            if not isinstance(images, (list, tuple)):
                images = [images]
            if not isinstance(text, (list, tuple)):
                text = [text] * len(images)
            # Check if batched text is provided
            if isinstance(text, (list, tuple)) and len(text) != len(images):
                raise ValueError(
                    "When using the image-text-to-text behavior, the number of prompts should be the same as the number of images."
                )
            prompts = list(zip(images, text))

        # if the value isn't overriden by the user, check if the tokenizer was trained with this token and then use it
        if add_end_of_utterance_token is None:
            add_end_of_utterance_token = self.tokenizer_was_trained_with_end_of_utterance_token
        # turn non-batched prompts into batched
        if not any(isinstance(i, (list, tuple)) for i in prompts):
            prompts = [prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        end_of_utterance_token = "<end_of_utterance>"

        def image_tokens(last_was_image):
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token

        all_prompts = []
        all_images = []
        for sample in prompts:
            # the model was trained on samples starting with <s>
            full_text = f"{self.tokenizer.bos_token}"

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            image_objects = []
            last_was_image = False
            last_was_text = False
            for i, item in enumerate(sample):
                if i > 0:
                    last_was_text = True if not last_was_image else False

                if isinstance(item, str):
                    item = item.strip(" ")
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        image_objects.append(image)
                        last_was_image = True
                    else:
                        # we add end_of_utterance_token between each subsequent text prompts (but not at the last one!)
                        if add_end_of_utterance_token and last_was_text:
                            full_text += end_of_utterance_token
                        full_text += item
                        last_was_image = False
                else:
                    # must be an image obj
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item)
                    last_was_image = True

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            if debug is True:
                print(f"{full_text=}")

            image_objects = self.image_processor(image_objects, transform=transform, return_tensors=return_tensors)

            all_prompts.append(full_text)
            all_images.append(image_objects)

        text_encoding = self.tokenizer(
            text=all_prompts,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        all_texts = text_encoding["input_ids"]
        all_attention_masks = text_encoding["attention_mask"]

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        at_least_one_image = sum(len(x) for x in all_images) > 0
        output_input_ids = []
        output_images = []
        output_attention_masks = []

        for text, attention_mask, images in zip(all_texts, all_attention_masks, all_images):
            padded_input_ids = text
            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)

            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                if return_tensors == "pt":
                    padded_image_tensor = torch.zeros(max_num_images, *current_images.size()[1:])
                    padded_image_tensor[: current_images.size(0)] = current_images
                elif return_tensors == "tf":
                    # Assuming current_images is a TensorFlow tensor
                    # Get the shape of current_images, excluding the first dimension
                    image_shape = tf.shape(current_images)[1:]
                    # Create a shape for the padded_image_tensor
                    padded_shape = tf.concat([[max_num_images], image_shape], axis=0)
                    # Create the padded_image_tensor of zeros
                    padded_image_tensor = tf.zeros(padded_shape, dtype=current_images.dtype)
                    # Get the number of images (assuming current_images has shape [num_images, height, width, channels])
                    num_images = tf.shape(current_images)[0]
                    # Update the padded_image_tensor with the values from current_images
                    indices = tf.reshape(tf.range(num_images), (-1, 1))
                    updates = current_images
                    padded_image_tensor = tf.tensor_scatter_nd_update(padded_image_tensor, indices, updates)
            else:
                if return_tensors == "pt":
                    padded_image_tensor = torch.zeros(max_num_images, *self.default_image_dims)
                elif return_tensors == "tf":
                    padded_image_tensor = tf.zeros((max_num_images, *self.default_image_dims))

            output_images.append(padded_image_tensor)
            if return_tensors == "pt":
                output_input_ids.append(torch.tensor(padded_input_ids))
                output_attention_masks.append(torch.tensor(attention_mask))
            elif return_tensors == "tf":
                output_input_ids.append(tf.convert_to_tensor(padded_input_ids, dtype=tf.int32))
                output_attention_masks.append(attention_mask)

        if return_tensors == "pt":
            output_input_ids = torch.stack(output_input_ids)
            output_images = torch.stack(output_images)
            output_attention_masks = torch.stack(output_attention_masks)
        elif return_tensors == "tf":
            output_input_ids = tf.stack(output_input_ids)
            output_images = tf.stack(output_images)
            output_attention_masks = tf.stack(output_attention_masks)

        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(
                output_input_ids, self.tokenizer, return_tensors
            )
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, return_tensors, num_classes=max_num_images
            )
        else:
            # in full language mode we set the image mask to all-0s
            if return_tensors == "pt":
                image_attention_mask = torch.zeros(
                    output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool
                )
            elif return_tensors == "tf":
                image_attention_mask = tf.zeros(
                    (output_input_ids.shape[0], output_input_ids.shape[1], 1), dtype=tf.bool
                )
        return BatchFeature(
            data={
                "input_ids": output_input_ids,
                "attention_mask": output_attention_masks,
                "pixel_values": output_images,
                "image_attention_mask": image_attention_mask,
            }
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
