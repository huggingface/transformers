"""
Processor class for VideoLLaMA3.
"""

from typing import Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    AllKwargsForChatTemplate,
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class Videollama3ImagesKwargs(ImagesKwargs):
    min_tokens: Optional[int]
    max_tokens: Optional[int]
    patch_size: Optional[int]
    image_merge_size: Optional[int]
    video_merge_size: Optional[int]


class Videollama3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Videollama3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class Videollama3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Videollama3ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Videollama3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            num_image_tokens = image_inputs["grid_sizes"].prod(dim=1) // (image_inputs["merge_sizes"] ** 2)
        else:
            num_image_tokens = []

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            timestamps = videos_inputs.pop("timestamps")
            num_frame_tokens = videos_inputs["grid_sizes_videos"][:, 1:].prod(dim=1) // (
                videos_inputs["merge_sizes_videos"] ** 2
            )
        else:
            timestamps = []
            num_frame_tokens = []

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            image_index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens[image_index], 1)
                    image_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    frame_prompts = [
                        f"Time {t:.1f}s:" + "<|placeholder|>" * num_frame_tokens[video_index]
                        for t in timestamps[video_index]
                    ]
                    text[i] = text[i].replace(self.video_token, ",".join(frame_prompts), 1)
                    video_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        return super().apply_chat_template(
            conversation, chat_template, image_token=self.image_token, video_token=self.video_token, **kwargs
        )


__all__ = ["Videollama3Processor"]
