# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for IDEFICS2.
"""

from typing import TYPE_CHECKING, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AddedToken, BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Idefics2Processor(ProcessorMixin):
    r"""
    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            config.perceiver_config.resampler_n_latents value for the model used.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 64, chat_template: str = None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
        self.image_token = AddedToken("<image>", normalized=False, special=True)
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        self.image_seq_len = image_seq_len

        tokens_to_add = {
            "additional_special_tokens": [self.fake_image_token, self.image_token, self.end_of_utterance_token]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    def __call__(
        self,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        image_seq_len: Optional[int] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        is_split_into_words: bool = False,
        add_special_tokens: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import Idefics2Processor
        >>> from transformers.image_utils import load_image

        >>> processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", image_seq_len=2)
        >>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [[image1], [image2]]

        >>> text = [
        ...     "<image>In this image, we see",
        ...     "bla bla bla<image>",
        ... ]
        >>> outputs = processor(text=text, images=images, return_tensors="pt", padding=True)
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        ['<s><fake_token_around_image><image><image><fake_token_around_image> In this image, we see', '<s> bla bla bla<fake_token_around_image><image><image><fake_token_around_image>']
        ```

        Args:
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value is used.
            padding (`Union[bool, str, PaddingStrategy]`, *optional*, defaults to `False`):
                Padding strategy applied to the input ids. See [`PreTrainedTokenizerFast.pad`] for more information.
            truncation (`Union[bool, str, TruncationStrategy]`, *optional*):
                Truncation strategy applied to the input ids. See [`PreTrainedTokenizerFast.truncate`] for more information.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding/truncation length. See
                [`PreTrainedTokenizerFast.__call__`] for more information.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether the input text is split into words or not. If set to `True`, the tokenizer will skip the
                tokenization process and assume the input is already tokenized.
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether to add special tokens or not. See [`PreTrainedTokenizerFast.__call__`] for more information.
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        n_images_in_text = []
        inputs = BatchFeature()

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            fake_image_token = self.fake_image_token.content
            image_token = self.image_token.content
            image_str = f"{fake_image_token}{image_token * image_seq_len}{fake_image_token}"

            if self.image_processor.do_image_splitting:
                # A single image token is split into 4 patches + 1 original image
                image_str = image_str * 5

            prompt_strings = []
            for sample in text:
                n_images_in_text.append(sample.count(image_token))
                sample = sample.replace(image_token, image_str)
                # Remove any double fake tokens if images are adjacent
                sample = sample.replace(f"{fake_image_token}{fake_image_token}", f"{fake_image_token}")
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(
                text=prompt_strings,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                is_split_into_words=is_split_into_words,
                return_tensors=return_tensors,
            )
            inputs.update(text_inputs)

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                images = [images]
            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )

            n_images_in_images = [len(sample) for sample in images]
            if text is not None and not n_images_in_images == n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

            # Load images if they are URLs
            images = [[load_image(im) for im in sample] for sample in images]
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
            inputs.update(image_inputs)

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

    @property
    def default_chat_template(self):
        """
        This template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content can be a single string or a list of strings and images.
        * If the content element is an image, the template will output a sequence of <image> tokens and <fake_token_around_image> token before and after each image
        * The template will output an <end_of_utterance> token at the end of each message.

        Example:

        ```python
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image"},
                {"type": "image"},
                ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground."},]
        }]
        ```

        Will create outputs like:
        ```
        User: What is in this Image?<image><image><end_of_utterance>
        Assistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>
        ```
        """
        # fmt: off
        return (
            "{% for message in messages %}"
                "{{message['role'].capitalize()}}"
                "{% if message['content'][0]['type'] == 'image' %}"
                    "{{':'}}"
                "{% else %}"
                    "{{': '}}"
                "{% endif %}"
                "{% for line in message['content'] %}"
                    "{% if line['type'] == 'text' %}"
                        "{{line['text']}}"
                    "{% elif line['type'] == 'image' %}"
                        "{{ '<image>' }}"
                    "{% endif %}"
                "{% endfor %}"
                "<end_of_utterance>\n"
            "{% endfor %}"

            "{% if add_generation_prompt %}"
                "{{ 'Assistant:' }}"
            "{% endif %}"
        )
        # fmt: on
