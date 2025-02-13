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

import copy
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import (
    ImageInput,
    VideoInput,
    is_valid_image,
    make_batched_videos,
    make_nested_list_of_images,
)
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import is_num2words_available, logging
from .video_processing_smolvlm import (
    DEFAULT_MEDIA_OUTTRO,
    DEFAULT_VIDEO_INTRO,
    FRAME_TIMESTAMP_MESSAGE,
    smolvlm_sample_indices_fn,
)


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


if is_num2words_available():
    from num2words import num2words
else:
    logger.warn("Please install `num2words` to use smolvlm. For example:\n\n" "  pip install num2words\n")


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_str(val) -> bool:
    return isinstance(val, str)


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


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
    max_image_size: Optional[Dict[str, int]]


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
        self.fake_image_token = tokenizer.fake_image_token
        self.image_token = tokenizer.image_token
        self.end_of_utterance_token = tokenizer.end_of_utterance_token
        self.global_image_token = tokenizer.global_image_token  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len
        self.video_frame_size = image_processor.video_sampling["video_size"]

        # Matches one or more occurrences of <row_x_col_y> tags (where x and y are digits, optionally surrounded by newline characters
        self._regex_to_remove_extra_special_tokens = re.compile(r"(<row_\d+_col_\d+>\n?)+")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def process_vision(self, text, images, output_kwargs, do_image_splitting=None, image_processor_size=None):
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        n_images_in_images = [len(sublist) for sublist in images]
        image_inputs = self.image_processor(
            images, do_image_splitting=do_image_splitting, size=image_processor_size, **output_kwargs["images_kwargs"]
        )

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

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
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
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
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

        inputs = BatchFeature()
        # Images and videos are mutually exclusive, so process one which is present
        if images is not None:
            images = make_nested_list_of_images(images)
            text, vision_inputs = self.process_vision(text, images, output_kwargs)
            inputs.update(vision_inputs)
        elif videos is not None:
            videos = make_batched_videos(videos)
            text, vision_inputs = self.process_vision(
                text,
                videos,
                output_kwargs,
                do_image_splitting=False,  # False only if videos
                image_processor_size=self.video_frame_size,
            )
            inputs.update(vision_inputs)

        if text is not None:
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        return inputs

    def _process_messaged_for_chat_template(
        self,
        conversations: List[List[Dict[str, str]]],
        batch_images: List[ImageInput],
        batch_videos: List[VideoInput],
        batch_video_metadata: List[List[Dict[str, any]]],
        **chat_template_kwargs,
    ):
        """
        Used within `apply_chat_template` when a model has special way to process conversation history. For example,
        video models might want to specify in the prompt the duratin of video or which frame indices ate which timestamps
        were sampled. This information cannot be accessed before the video is loaded.
        For most models it is a no-op, must be overriden by model processors which require special processing.
        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to process. Always comes in batched format.
            batch_images (`List[List[ImageInput]]`):
                Batch of images that were loaded from url/path defined in the conversation. The images
                are ordered in the samm way as in the conversation. Comes in nested list format, one list of `PIL` images
                per batch.
            batch_videos (`List[List[ImageInput]]`):
                Batch of videos that were loaded from url/path defined in the conversation. The videos
                are ordered in the samm way as in the conversation. Comes in nested list format, one list of 4D video arrays
                per batch.
            batch_video_metadata (`List[List[Dict[[str, any]]]]`):
                Batch of metadata returned from loading videos. That includes video fps, duration and total number of framer in original video.
                Metadata are ordered in the samm way as `batch_videos`. Comes in nested list format, one list of 4D video arrays
                per batch.
        """
        # We don't want to modify in-place the messages passed by user
        # The user might want to add new turn on conv and continue geenration
        conversations = copy.deepcopy(conversations)
        batch_num_frames, batch_timestamps = [], []
        for metadata_list, video_list in zip(batch_video_metadata, batch_videos):
            for metadata, video in zip(metadata_list, video_list):
                duration_sec = metadata["duration"]
                frames_idx = metadata["frames_indices"]
                fps = metadata["fps"]

                timestamps = []
                for idx, frame_np in zip(frames_idx, video):
                    sec = idx / fps
                    mm = int(sec // 60)
                    ss = int(sec % 60)
                    timestamps.append(f"{mm:02d}:{ss:02d}")
                batch_timestamps.append(timestamps)
                batch_num_frames.append(len(video))

        for conversation in conversations:
            # For each message, scan content for {"type": "video"}
            for msg in conversation:
                if "content" not in msg:
                    continue

                new_content = []
                for block in msg["content"]:
                    if block.get("type") == "video":
                        curr_timestamps = batch_timestamps.pop(0)
                        curr_num_frames = batch_num_frames.pop(0)

                        # Build the video intro texts
                        td = timedelta(seconds=int(duration_sec))
                        new_content.append(
                            {
                                "type": "text",
                                "text": DEFAULT_VIDEO_INTRO.format(
                                    frame_count=num2words(curr_num_frames), video_duration=str(td)
                                ),
                            }
                        )

                        # 2) Insert per-frame lines: "Frame from {timestamp}:", then an "image" block
                        for i, ts in enumerate(curr_timestamps):
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
        return conversations

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

    # Add model-specific video sampling method when applying the template
    def apply_chat_template(self, conversation, max_frames=64, target_fps=1, skip_secs=1, **kwargs):
        def sample_indices_fn_func(metadata, **fn_kwargs):
            return smolvlm_sample_indices_fn(
                metadata, max_frames=max_frames, target_fps=target_fps, skip_secs=skip_secs, **fn_kwargs
            )

        sample_indices_fn = sample_indices_fn_func
        return super().apply_chat_template(
            conversation, sample_indices_fn=sample_indices_fn, video_load_backend="decord", **kwargs
        )


__all__ = ["SmolVLMProcessor"]
