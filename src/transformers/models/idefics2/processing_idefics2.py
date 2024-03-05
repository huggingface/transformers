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

from ...processing_utils import ProcessorMixin
from ...feature_extraction_utils import BatchFeature
from typing import List, Optional, Union



class IdeficsProcessor(ProcessorMixin):
    r"""
    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224): Image size (assuming a square image)
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics2ImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        prompts: Union[List[TextInput], List[List[TextInput]]] = None,
        images: Optional[Union[ImageInput, List[ImageInput], List[List[ImageInput]]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchEncoding:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            prompts (`Union[List[TextInput], [List[List[TextInput]]]]`):
                either a single prompt or a batched list of prompts.
            images (`Union[ImageInput, List[ImageInput], List[List[ImageInput]]]`, *optional*):
                either a single image or a batched list of images to process
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`):
                The type of tensors to return. Can be one of:
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.

        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

        Example:

        ```python
        checkpoint = "HuggingFaceM4/idefics2"
        processor = AutoProcessor.from_pretrained(checkpoint)
        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
        img = processor.image_processor.fetch_images([url])[0]

        image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )
        raw_images = [
            [image1],
            [image2, image3],
        ]
        prompts = [
            "<fake_token_around_image>{image_seq}<fake_token_around_image>In this image, we see",
            "bla bla<fake_token_around_image>{image_seq}<fake_token_around_image>{image_seq}<fake_token_around_image>",
        ]

        inputs = processor(prompts, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```
        """

        inputs = BatchFeature({}, return_tensors=return_tensors)

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]

            # Add to BOS token to the prompts
            prompts = [f"{self.tokenizer.bos_token}{p}" for p in prompts]

            text_inputs = self.tokenizer(
                text=prompts,
                add_special_tokens=False,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )
            inputs.update(text_inputs)

        if images is not None:
            image_inputs = self.image_processor(
                images,
                return_tensors=return_tensors,
            )
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
