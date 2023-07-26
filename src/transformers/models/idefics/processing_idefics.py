"""
Processor class for IDEFICS.
"""

from typing import List, Optional, Union
from urllib.parse import urlparse

from transformers import is_torch_available
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    BatchEncoding,
    TextInput,
)
from transformers.utils import TensorType


if is_torch_available():
    import torch


IMAGE_TOKEN = "<image>"


# copied from m4.training.packing
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    # This function converts: [-1, 0, 1] => [[0, 0], [1, 0], [0, 1]]

    # If any of images index are more than num_classes, set them to -1.
    # Words after the max number of images allowed have been seen don't attend on anything
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1

    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask


# copied from m4.training.packing
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer):
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
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def __call__(
        self,
        prompts: Union[List[TextInput], List[List[TextInput]]],
        eval_mode: bool = False,
        device: str = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        debug=False,
    ) -> BatchEncoding:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            prompts (`Union[List[TextInput], [List[List[TextInput]]]]`):
                either a single prompt or a batched list of prompts - see the detailed description immediately after
                the end of the arguments doc section.
            eval_mode (`bool`, *optional*, defaults to `True`):
                `True` should be used for inference, and `False` for training - this option impacts how cropping is
                performed
            device (`str`, *optional*, defaults to `None`):
                 if `device` is passed, the return dict values will be placed on that device

            debug (`bool`, *optional*, defaults to `False`):
                `True` value will help debug prompt generation by dumping useful information

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

        inputs = processor(prompts, eval_mode=True, device=device, return_tensors="pt")
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

        In order to help debug prompt generation enable `debug=True` which will show you what's happening.

        """

        # turn non-batched prompts into batched
        if not any(isinstance(i, list) for i in prompts):
            prompts = [prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"

        def image_tokens(last_was_image):
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token

        all_texts = []
        all_images = []
        for sample in prompts:
            # the model was trained on samples starting with <s>
            full_text = f"{self.tokenizer.bos_token}"

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            real_images = []
            last_was_image = False
            for item in sample:
                if isinstance(item, str):
                    item = item.strip(" ")
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        real_images.append(image)
                        last_was_image = True
                    else:
                        full_text += item
                        last_was_image = False
                else:
                    # must be an image obj
                    full_text += image_tokens(last_was_image)
                    real_images.append(item)
                    last_was_image = True

            if debug is True:
                print(f"{full_text=}")

            real_images = self.image_processor(real_images, eval_mode=eval_mode, return_tensors=return_tensors)
            if len(real_images) > 0:
                real_images = torch.stack(real_images)

            encoding = self.tokenizer(
                text=full_text,
                add_special_tokens=False,
            )

            all_texts.append(encoding["input_ids"])
            all_images.append(real_images)

        max_num_images = max(len(x) for x in all_images)
        max_seq_len = max(len(x) for x in all_texts)

        output_input_ids = []
        output_images = []
        output_attention_masks = []
        for text, images in zip(all_texts, all_images):
            padded_input_ids = [self.tokenizer.pad_token_id] * max_seq_len
            unpadded_seq_len = len(text)
            start = max_seq_len - unpadded_seq_len
            padded_input_ids[start:] = text[:max_seq_len]

            attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
            attention_mask[start:] = 1

            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)

            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(max_num_images, *current_images.size()[1:])
                padded_image_tensor[: current_images.size(0)] = current_images
            else:
                default_image_size = (
                    3,
                    self.image_processor.image_size,
                    self.image_processor.image_size,
                )
                padded_image_tensor = torch.zeros(max_num_images, *default_image_size)

            output_images.append(padded_image_tensor)
            output_input_ids.append(torch.tensor(padded_input_ids))

            output_attention_masks.append(attention_mask)

        output_input_ids = torch.stack(output_input_ids)
        output_images = torch.stack(output_images)
        output_attention_masks = torch.stack(output_attention_masks)

        image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, self.tokenizer)
        image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)

        inputs = {
            "input_ids": output_input_ids,
            "attention_mask": output_attention_masks,
            "pixel_values": output_images,
            "image_attention_mask": image_attention_mask,
        }

        if device is not None:
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device)

        return inputs

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
