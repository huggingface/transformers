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
Processor class for SmolVLM.
"""

import re
from datetime import timedelta
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import AddedToken, BatchEncoding, TextInput
from ...utils import logging
from .video_processing_smolvlm import (
    DEFAULT_MEDIA_OUTTRO,
    DEFAULT_VIDEO_INTRO,
    FRAME_TIMESTAMP_MESSAGE,
    load_smolvlm_video,
)


try:
    from num2words import num2words
except ImportError as e:
    raise ImportError(
        "Please install `num2words` to use smolvlm. For example:\n\n"
        "  pip install num2words\n"
    ) from e

if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_str(val) -> bool:
    return isinstance(val, str)


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token):
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
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_img_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token
    )


class SmolVLMImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: Optional[bool]
    max_image_size: Optional[Dict[str, int]]


class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SmolVLMImagesKwargs

    _defaults = {
        "add_generation_prompt": False,
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
        "videos_kwargs": {
            "max_frames": None,
            "sampling_fps": None,
            "video_duration": None,
        },
    }


SmolVLMProcessorKwargs.__annotations__["images_kwargs"] = SmolVLMImagesKwargs  # python 3.8 compatibility


class SmolVLMProcessor(ProcessorMixin):
    r"""
    Constructs a SmolVLM processor which wraps a LLama tokenizer and SmolVLM image processor into a single processor.

    [`SmolVLMProcessor`] offers all the functionalities of [`SmolVLMImageProcessor`] and [`SmolVLMTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`SmolVLMImageProcessor`):
            An instance of [`SmolVLMImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["image_seq_len", "chat_template"]
    image_processor_class = "SmolVLMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 169, chat_template: str = None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")

        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
        self.image_token = AddedToken("<image>", normalized=False, special=True)
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        self.global_image_tag = "<global-img>"  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len

        self.video_sampling_fps = image_processor.video_sampling["fps"]
        self.video_frame_size = image_processor.video_sampling["video_size"]
        self.max_frames = image_processor.video_sampling["max_frames"]

        # This regex matches one or more occurrences of <global-img> tags (optionally surrounded by newline characters)
        # or <row_x_col_y> tags (where x and y are digits, also optionally surrounded by newline characters).
        self._regex_to_remove_extra_special_tokens = re.compile(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

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
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        audio=None,
        videos: Union[str, List[ImageInput], List[List[ImageInput]]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import SmolVLMProcessor
        >>> from transformers.image_utils import load_image

        >>> processor = SmolVLMProcessor.from_pretrained("HuggingFaceM4/SmolVLM-8B-Llama3")
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
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
                image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is not None and messages is not None:
            raise ValueError("You must provide only one of `text` or `messages'.")

        if text is None and images is None and videos is None:
            raise ValueError("You must provide one of `text`, `images` or `videos'.")

        if images is not None and videos is not None:
            raise ValueError("You must provide either `images` or `videos', not both.")

        output_kwargs = self._merge_kwargs(
            SmolVLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        video_sampling_fps = output_kwargs["videos_kwargs"]["sampling_fps"]
        max_frames = output_kwargs["videos_kwargs"]["max_frames"]
        add_generation_prompt = kwargs["add_generation_prompt"] if "add_generation_prompt" in kwargs else False

        video_sampling_fps = video_sampling_fps if video_sampling_fps is not None else self.video_sampling_fps
        max_frames = max_frames if max_frames is not None else self.max_frames
        add_generation_prompt = add_generation_prompt if add_generation_prompt is not None else False

        inputs = BatchFeature()
        if images is not None:
            if messages is not None and text is None:
                text = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

            self.process_images(inputs, text, images, output_kwargs)

        elif videos is not None:
            if is_str(videos) or is_url(videos):
                # Single path/URL
                frames, timestamps, duration_sec = load_smolvlm_video(
                    videos, sampling_fps=video_sampling_fps, max_frames=max_frames
                )
                images = [frames]

            elif isinstance(videos, (list, tuple)):
                if videos and is_image_or_image_url(videos[0]) and messages is not None:
                    # => single list of frames => wrap as [video]
                    frames = list(videos)
                    images = [frames]
                    # Possibly the user provides an explicit video_duration in "videos_kwargs"
                    duration_sec = output_kwargs["videos_kwargs"].get("video_duration", None)

                    # If not provided, do a naive approach: (len - 1) / fps
                    if duration_sec is None:
                        duration_sec = max(len(frames) - 1, 0) / float(video_sampling_fps)

                    if messages is not None and text is None:
                        timestamps = []
                        n_frames = len(frames)
                        for i in range(n_frames):
                            sec = i / (n_frames - 1) * duration_sec
                            mm = int(sec // 60)
                            ss = int(sec % 60)
                            ts_str = f"{mm:02d}:{ss:02d}"
                            timestamps.append(ts_str)

                elif videos and is_image_or_image_url(videos[0]) and text is not None:
                    frames = list(videos)
                    images = [frames]

                else:
                    raise ValueError("Invalid format for `videos` argument when it's a list/tuple.")

            if messages is not None and text is None:
                self.process_video_token(
                    messages, num_frames=len(frames), timestamps=timestamps, duration_sec=duration_sec
                )
                text = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

            self.process_images(
                inputs,
                text,
                images,
                output_kwargs,
                do_image_splitting=False,
                image_processor_size=self.video_frame_size,
            )

        elif text is not None:
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        return inputs

    def process_images(self, inputs, text, images, output_kwargs, do_image_splitting=None, image_processor_size=None):
        image_seq_len = self.image_seq_len
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token.content) for sample in text]

        if is_image_or_image_url(images):
            images = [[images]]
        elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
            if text is not None:
                if sum(n_images_in_text) != len(images):
                    raise ValueError(
                        f"The total number of {self.image_token.content} tokens in the prompts should be the same as the number of images passed."
                        f" Found {sum(n_images_in_text)} {self.image_token.content} tokens and {len(images)} images."
                    )
                # Reorganize the images to match the prompts
                cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                images = [
                    images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                    for i in range(len(n_images_in_text))
                ]
            else:
                images = [images]
        elif (
            not isinstance(images, (list, tuple))
            and not isinstance(images[0], (list, tuple))
            and not is_image_or_image_url(images[0][0])
        ):
            raise ValueError(
                "Invalid input images. Please provide a single image or a list of images or a list of list of images."
            )
        n_images_in_images = [len(sample) for sample in images]

        # Load images if they are URLs
        images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

        image_inputs = self.image_processor(
            images, do_image_splitting=do_image_splitting, size=image_processor_size, **output_kwargs["images_kwargs"]
        )
        inputs.update(image_inputs)

        if text is not None:
            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                )

            image_rows = inputs.pop("rows", [[0] * len(text)])
            image_cols = inputs.pop("cols", [[0] * len(text)])

            fake_image_token = self.fake_image_token.content
            image_token = self.image_token.content
            global_img_token = self.global_image_tag

            prompt_strings = []
            for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                image_prompt_strings = []
                for n_rows, n_cols in zip(sample_rows, sample_cols):
                    image_prompt_string = get_image_prompt_string(
                        n_rows,
                        n_cols,
                        image_seq_len,
                        image_token=image_token,
                        fake_token_around_image=fake_image_token,
                        global_img_token=global_img_token,
                    )
                    image_prompt_strings.append(image_prompt_string)

                split_sample = sample.split(image_token)
                if len(split_sample) == 0:
                    raise ValueError("The image token should be present in the text.")

                # Place in the image prompt strings where the image tokens are
                sample = split_sample[0]
                for i, image_prompt_string in enumerate(image_prompt_strings):
                    sample += image_prompt_string + split_sample[i + 1]
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

    def process_video_token(self, messages, num_frames, timestamps, duration_sec):
        """
        converts the video token into the interleaved text and image tokens the model was trained with.
        Args:
            messages (List[Dict]): A list of conversation turns, where each turn has
                a "role" and a "content" (which is a list of blocks).
            frames (int): The number of frames that were extracted from a video.
            timestamps (List[str]): Parallel list of timestamps for the frames.
            duration_sec (float): The total video duration in seconds.
        """
        if not num_frames or not timestamps or duration_sec is None:
            raise ValueError("process_video_token requires valid frames, timestamps, and duration_sec.")

        # For each message, scan content for {"type": "video"}
        for msg in messages:
            if "content" not in msg:
                continue

            new_content = []
            for block in msg["content"]:
                if block.get("type") == "video":
                    # frames, timestamps, duration_sec must be provided
                    # (Alternatively, you could dynamically load them here.)
                    if num_frames is None or timestamps is None or duration_sec is None:
                        # If user didn't pass these, raise or skip
                        raise ValueError(
                            "Must provide num_frames, timestamps, and duration_sec to insert 'video' blocks."
                        )

                    # Build the video intro texts
                    td = timedelta(seconds=duration_sec)
                    new_content.append(
                        {
                            "type": "text",
                            "text": DEFAULT_VIDEO_INTRO.format(
                                frame_count=num2words(num_frames), video_duration=str(td)
                            ),
                        }
                    )

                    # 2) Insert per-frame lines: "Frame from {timestamp}:", then an "image" block
                    for i, ts in enumerate(timestamps):
                        new_content.append({"type": "text", "text": FRAME_TIMESTAMP_MESSAGE.format(timestamp=ts)})
                        new_content.append({"type": "image"})

                    # 3) Optionally add an outro (e.g. "Now answer the question:")
                    new_content.append({"type": "text", "text": DEFAULT_MEDIA_OUTTRO})
                    # Do NOT add the original block => we skip it (since we've replaced it)
                else:
                    # keep original block
                    new_content.append(block)

            # update the content
            msg["content"] = new_content

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return [self._regex_to_remove_extra_special_tokens.sub("<image>", s) for s in batched_decode_output]

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return self._regex_to_remove_extra_special_tokens.sub("<image>", decode_output)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + tokenizer_input_names))


__all__ = ["SmolVLMProcessor"]
