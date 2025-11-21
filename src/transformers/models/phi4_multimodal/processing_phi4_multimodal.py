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
from typing import List, Optional, Union

from ...audio_utils import AudioInput
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class Phi4MultimodalProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "device": "cpu",
        },
    }


class Phi4MultimodalProcessor(ProcessorMixin):
    r"""
    Constructs a Phi4Multimodal processor which raps an image processor, a audio processor, and a GPT tokenizer into a single processor.

    [`Phi4MultimodalProcessor`] offers all the functionalities of [`Phi4MultimodalImageProcessorFast`] and [`GPT2Tokenizer`]. See the
    [`~Phi4MultimodalProcessor.__call__`] and [`~Phi4MultimodalProcessor.decode`] for more information.

    Args:
        image_processor (`Phi4MultimodalImageProcessorFast`):
            The image processor to use for images.
        audio_processor (`Phi4MultimodalFeatureExtractor`):
            The audio processor to use for audio inputs.
        tokenizer (`GPT2TokenizerFast`):
            The tokenizer to use for text.
        fake_image_token_pattern (`str`, *optional*, defaults to `r"<\|image_\d+\|>"`):
            The fake image token pattern.
        fake_audio_token_pattern (`str`, *optional*, defaults to `r"<\|audio_\d+\|>"`):
            The fake audio token pattern.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    tokenizer_class = "GPT2TokenizerFast"
    image_processor_class = "Phi4MultimodalImageProcessorFast"
    audio_processor_class = "Phi4MultimodalFeatureExtractor"
    valid_kwargs = ["chat_template"]

    def __init__(
        self,
        image_processor,
        audio_processor,
        tokenizer,
        **kwargs,
    ):
        super().__init__(image_processor, audio_processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        images: Optional[ImageInput] = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forards the `text`
        and `kwargs` arguments to GPT2Tokenizer's [`~GPT2Tokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        Phi4MultimodalImageProcessorFast's [`~Phi4MultimodalImageProcessorFast.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            audio (`List[Union[np.ndarray, torch.Tensor]]`):
                List of the audios to be prepared.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **input_image_embeds** -- Pixel values to be fed to a model.
            - **image_sizes** -- List of tuples specifying the size of each image in `input_image_embeds`.
            - **image_attention_mask** -- List of attention masks for each image in `input_image_embeds`.
            - **input_audio_embeds** -- Audio embeddings to be fed to a model.
            - **audio_embed_sizes** -- List of integers specifying the size of each audio in `input_audio_embeds`.
        """

        output_kwargs = self._merge_kwargs(Phi4MultimodalProcessorKwargs, self.tokenizer.init_kwargs, **kwargs)
        image_kwargs = output_kwargs["images_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]

        image_inputs = self.image_processor(images, **image_kwargs) if images is not None else {}
        audio_inputs = self.audio_processor(audio, **audio_kwargs) if audio is not None else {}

        # We pop here for images as we don't need it later
        num_img_tokens = image_inputs.pop("num_img_tokens", [])
        audio_embed_sizes = audio_inputs.get("audio_embed_sizes", [])

        # Replace certain special tokens for compatibility
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_token = self.tokenizer.image_token
        audio_token = self.tokenizer.audio_token

        # Check that the number of special tokens is sound
        concatenated_prompt = "".join(text)
        if concatenated_prompt.count(image_token) != len(num_img_tokens):
            raise ValueError(
                "You should add as much image tokens `<|image|>` in your prompt as you pass `images` to the processor. ",
                f"Input contains {concatenated_prompt.count(image_token)} tokens != {len(num_img_tokens)} images",
            )
        if concatenated_prompt.count(audio_token) != len(audio_embed_sizes):
            raise ValueError(
                "You should add as much audio tokens `<|audio|>` in your prompt as you pass `audios` to the processor. "
                f"Input contains {concatenated_prompt.count(audio_token)} tokens != {len(audio_embed_sizes)} audios"
            )

        # Add appropriate number of image/audio tokens (note that the count of replacement is dynamic)
        image_count_iter = iter(num_img_tokens)
        audio_count_iter = iter(audio_embed_sizes)
        processed_text = [
            re.sub(re.escape(image_token), lambda _: image_token * next(image_count_iter), t) for t in text
        ]
        processed_text = [
            re.sub(re.escape(audio_token), lambda _: audio_token * next(audio_count_iter), t) for t in processed_text
        ]

        text_inputs = self.tokenizer(processed_text, **text_kwargs)

        # prepare batch feature
        data = {
            **text_inputs,
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
