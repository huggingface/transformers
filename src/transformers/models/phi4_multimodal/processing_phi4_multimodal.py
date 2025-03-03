# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
Processor class for Phi4Multimodal
"""

import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ImageInput,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

# Special tokens
COMPATIBLE_IMAGE_TOKEN_PATTERN = r"<\|image_\d+\|>"
COMPATIBLE_AUDIO_TOKEN_PATTERN = r"<\|audio_\d+\|>"


AudioInput = Tuple[Union[np.ndarray, torch.Tensor], int]
AudioInputs = List[AudioInput]


class Phi4MultimodalProcessor(ProcessorMixin):
    r"""
    Constructs a Phi4Multimodal processor which raps an image processor, a audio processor, and a GPT tokenizer into a single processor.

    [`Phi4MultimodalProcessor`] offers all the functionalities of [`Phi4MultimodalImageProcessor`] and [`GPT2Tokenizer`]. See the
    [`~Phi4MultimodalProcessor.__call__`] and [`~Phi4MultimodalProcessor.decode`] for more information.

    Args:
        image_processor ([`Phi4MultimodalImageProcessor`]):
            The image processor is a required input.
        audio_processor (`<fill_type>`): <fill_docstring>
        tokenizer ([`GPT2Tokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    tokenizer_class = "GPT2TokenizerFast"
    image_processor_class = "Phi4MultimodalImageProcessor"
    audio_processor_class = "Phi4MultimodalFeatureExtractor"
    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor, audio_processor, tokenizer, **kwargs):
        super().__init__(image_processor, audio_processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        images: Optional[ImageInput] = None,
        audios: Optional[AudioInputs] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forards the `text`
        and `kwargs` arguments to GPT2Tokenizer's [`~GPT2Tokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        Phi4MultimodalImageProcessor's [`~Phi4MultimodalImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
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
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **input_image_embeds** -- Pixel values to be fed to a model.
            - **image_sizes** -- List of tuples specifying the size of each image in `input_image_embeds`.
            - **image_attention_mask** -- List of attention masks for each image in `input_image_embeds`.
            - **input_audio_embeds** -- Audio embeddings to be fed to a model.
            - **audio_embed_sizes** -- List of integers specifying the size of each audio in `input_audio_embeds`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
        """
        image_inputs = self.image_processor(images, return_tensors=return_tensors) if images is not None else {}
        audio_inputs = self.audio_processor(audios, return_tensors=return_tensors) if audios is not None else {}
        inputs = self._convert_images_audios_text_to_inputs(
            image_inputs,
            audio_inputs,
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        return inputs

    def _convert_images_audios_text_to_inputs(
        self, images, audios, text, padding=False, truncation=None, max_length=None, return_tensors=None
    ):
        image_inputs = {}
        audio_inputs = {}
        num_img_tokens = []
        audio_embed_sizes = torch.tensor([])

        # image inputs
        if len(images) > 0:
            image_inputs_args = ["input_image_embeds", "image_sizes", "image_attention_mask"]
            image_inputs = {k: v for k, v in images.items() if k in image_inputs_args}
            num_img_tokens = images["num_img_tokens"]

        # Audio inputs
        audio_inputs = {}
        if len(audios) > 0:
            audio_inputs_args = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]
            audio_inputs = {k: v for k, v in audios.items() if k in audio_inputs_args}
            audio_embed_sizes = audios["audio_embed_sizes"]

        # Replace certain special tokens for compatibility
        if isinstance(text, str):
            text = [text]
        processed_text = [re.sub(COMPATIBLE_IMAGE_TOKEN_PATTERN, self.tokenizer.image_token, t) for t in text]
        processed_text = [
            re.sub(COMPATIBLE_AUDIO_TOKEN_PATTERN, self.tokenizer.audio_token, t) for t in processed_text
        ]

        input_ids_list = [self.tokenizer(t).input_ids for t in processed_text]

        img_cnt, audio_cnt = 0, 0  # only needed for later assertion
        image_token_count_iter = iter(num_img_tokens)
        audio_embed_size_iter = iter(audio_embed_sizes.tolist())
        new_input_ids_list = []
        for input_ids in input_ids_list:
            i = 0
            while i < len(input_ids):
                token_id = input_ids[i]
                if token_id == self.tokenizer.audio_token_id:
                    token_count = next(audio_embed_size_iter)
                    audio_cnt += 1
                elif token_id == self.tokenizer.image_token_id:
                    token_count = next(image_token_count_iter)
                    img_cnt += 1
                else:
                    i += 1
                    continue
                tokens = [token_id] * token_count
                input_ids = input_ids[:i] + tokens + input_ids[i + 1 :]
                i += token_count
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            new_input_ids_list.append(input_ids)
        lengths = torch.tensor([len(input_ids) for input_ids in new_input_ids_list])
        max_len = lengths.max()
        input_ids = input_ids.new_full((len(new_input_ids_list), max_len), self.tokenizer.pad_token_id)
        # batched inference requires left padding
        for i in range(len(new_input_ids_list)):
            input_ids[i, max_len - len(new_input_ids_list[i]) :] = new_input_ids_list[i]

        # If the below assertion fails, it might be that input pure-text
        # messages contain image/audio special tokens literally
        # (<|endoftext10|>, <|endoftext11|>).
        assert img_cnt == len(num_img_tokens), (
            f"Number of image tokens in prompt_token_ids ({img_cnt}) "
            f"does not match number of images ({len(num_img_tokens)})"
        )
        assert audio_cnt == len(audio_embed_sizes), (
            f"Number of audio tokens in prompt_token_ids ({audio_cnt}) "
            f"does not match number of audios ({len(audio_embed_sizes)})"
        )

        # prepare attention mask
        seq_range = torch.arange(max_len - 1, -1, -1)
        attention_mask = seq_range.unsqueeze(0) < lengths.unsqueeze(1)

        # prepare batch feature
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **image_inputs,
            **audio_inputs,
        }

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_processor_input_names))


__all__ = ["Phi4MultimodalProcessor"]
