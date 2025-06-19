# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Processor class for SmolVLM.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import AllKwargsForChatTemplate, ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import is_num2words_available, is_vision_available, logging
from ...video_utils import VideoInput


if is_vision_available():
    from .video_processing_smolvlm import (
        DEFAULT_MEDIA_OUTTRO,
        DEFAULT_VIDEO_INTRO,
        FRAME_TIMESTAMP_MESSAGE,
    )

if is_vision_available():
    from .video_processing_smolvlm import (
        DEFAULT_MEDIA_OUTTRO,
        DEFAULT_VIDEO_INTRO,
        FRAME_TIMESTAMP_MESSAGE,
    )

if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


if is_num2words_available():
    from num2words import num2words
else:
    num2words = None


# The correct chat template to be used for videos after #38105
DEFAULT_CHAT_TEMPLATE = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def _prompt_split_image(
    image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_image_token
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_image_token):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_image_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_image_token=global_image_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_image_token
    )


class SmolVLMImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: Optional[bool]
    max_image_size: Optional[dict[str, int]]


class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SmolVLMImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
    }


class SmolVLMProcessor(ProcessorMixin):
    r"""
    Constructs a SmolVLM processor which wraps a LLama tokenizer and SmolVLM image processor into a single processor.

    [`SmolVLMProcessor`] offers all the functionalities of [`SmolVLMImageProcessor`] and [`SmolVLMTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`SmolVLMImageProcessor`):
            An instance of [`SmolVLMImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        video_processor (`SmolVLMImageProcessor`):
            n instance of [`SmolVLMImageProcessor`]. The video processor is a required input.
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "SmolVLMImageProcessor"
    video_processor_class = "SmolVLMVideoProcessor"  # NOTE: uses different interpolation than slow processors
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        image_seq_len: int = 169,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.fake_image_token = getattr(tokenizer, "fake_image_token", "<fake_token_around_image>")
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.end_of_utterance_token = getattr(tokenizer, "end_of_utterance_token", "<end_of_utterance>")
        self.global_image_token = getattr(tokenizer, "global_image_token", "<global-img>")
        self.image_seq_len = image_seq_len
        self.video_token = getattr(tokenizer, "video_token", "<video>")

        if not num2words:
            raise ImportError(
                "Package `num2words` is required to run SmolVLM processor. Install it with `pip install num2words`."
            )

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    def process_vision(self, text, images, output_kwargs):
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        n_images_in_images = [len(sublist) for sublist in images]
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is None:
            return None, image_inputs

        if n_images_in_images != n_images_in_text:
            raise ValueError(
                f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
            )
        image_rows = image_inputs.pop("rows", [[0] * len(text)])
        image_cols = image_inputs.pop("cols", [[0] * len(text)])

        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    self.image_seq_len,
                    image_token=self.image_token,
                    fake_token_around_image=self.fake_image_token,
                    global_image_token=self.global_image_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(self.image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        return prompt_strings, image_inputs

    def process_video(self, text, videos, output_kwargs):
        if text is not None:
            n_videos_in_text = [sample.count(self.video_token) for sample in text]

        n_videos_in_videos = [len(sublist) for sublist in videos]
        video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

        num_frames = video_inputs["pixel_values"].shape[1]
        batch_timestamps = iter(video_inputs.pop("timestamps"))
        batch_durations = iter(video_inputs.pop("durations"))

        if text is None:
            return None, video_inputs

        if n_videos_in_videos != n_videos_in_text:
            raise ValueError(
                f"The number of videos in the text {n_videos_in_text} and videos {n_videos_in_videos} should be the same."
            )

        prompt_strings = []
        for sample in text:
            while self.video_token in sample:
                timestamps = next(batch_timestamps)
                duration = next(batch_durations)
                duration_td = timedelta(seconds=int(duration))
                image_prompt_strings = DEFAULT_VIDEO_INTRO.format(
                    frame_count=num2words(num_frames), video_duration=str(duration_td)
                )
                for timestamp in timestamps:
                    image_prompt_string = _prompt_single_image(
                        self.image_seq_len,
                        image_token=self.image_token,
                        fake_token_around_image=self.fake_image_token,
                        global_image_token=self.global_image_token,
                    )
                    timestamp = f"{timestamp[0]:02d}:{timestamp[1]:02d}"
                    image_prompt_string = FRAME_TIMESTAMP_MESSAGE.format(timestamp=timestamp) + image_prompt_string
                    image_prompt_strings += image_prompt_string

                image_prompt_strings += DEFAULT_MEDIA_OUTTRO
                sample = sample.replace(self.video_token, image_prompt_strings, 1)
            prompt_strings.append(sample)
        return prompt_strings, video_inputs

    def __call__(
        self,
        images: Union[ImageInput, list[ImageInput], list[list[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[SmolVLMProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import SmolVLMProcessor
        >>> from transformers.image_utils import load_image

        >>> processor = SmolVLMProcessor.from_pretrained("HuggingFaceM4/SmolVLM2-256M-Video-Instruct")
        >>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [[image1], [image2]]

        >>> text = [
        ...     "<image>In this image, we see",
        ...     "bla bla bla<image>",
        ... ]
        >>> outputs = processor(images=images, text=text, return_tensors="pt", padding=True)
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        ['<|begin_of_text|><fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image> In this image, we see', '<|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|><|begin_of_text|>bla bla bla<fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image>']
        ```

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            videos (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The video or batch of videos to be prepared. Each video can be a list of PIL frames, NumPy array or PyTorch
                tensor. If is of type `list[VideoInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is None and images is None and videos is None:
            raise ValueError("You must provide one of `text`, `images` or `videos'.")

        if text is None and ((images is None) ^ (videos is not None)):
            raise ValueError("You must specify exactly one of `images` or `videos`")

        output_kwargs = self._merge_kwargs(
            SmolVLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = sum([sample.count(self.image_token) for sample in text])
            if n_images_in_text > 0 and (images is None and videos is None):
                raise ValueError(f"We detected {n_images_in_text} tokens in the text but no images/videos were passed")

        inputs = {}
        # Images and videos are mutually exclusive, so process one which is present
        if images is not None:
            images = make_nested_list_of_images(images)
            text, vision_inputs = self.process_vision(
                text,
                images,
                output_kwargs,
            )
            inputs.update(vision_inputs)
        elif videos is not None:
            text, vision_inputs = self.process_video(
                text,
                videos,
                output_kwargs,
            )
            inputs.update(vision_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
            inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return batched_decode_output

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return decode_output

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + tokenizer_input_names))

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            conversations = conversation
        else:
            conversations = [conversation]

        has_video = any(
            (isinstance(content, dict) and content["type"] == "video")
            for conversation in conversations
            for message in conversation
            for content in message["content"]
        )
        if chat_template is None and has_video:
            # re-assign to the correct default template for BC, if user is not requesting their own template
            chat_template = DEFAULT_CHAT_TEMPLATE
        return super().apply_chat_template(conversation, chat_template, **kwargs)


__all__ = ["SmolVLMProcessor"]
