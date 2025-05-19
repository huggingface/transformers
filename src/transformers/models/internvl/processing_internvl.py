# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from typing import List, Optional, Union

import numpy as np

from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
    concatenate_list,
    make_flat_list_of_images,
)
from ...video_utils import VideoInput, VideoMetadata, load_video, make_batched_videos


class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
        "videos_kwargs": {},
    }


class InternVLProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternVLProcessor.__call__`] and [`~InternVLProcessor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`AutoVideoProcessor`], *optional*):
            The video processor is a required input.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch. it should be set so that:
            image_seq_length = (config.image_size // config.patch_size) ** 2 * (config.scale_factor**2)
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = [
        "chat_template",
        "image_seq_length",
    ]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        image_seq_length: int = 256,
        chat_template=None,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.start_image_token = tokenizer.start_image_token
        self.end_image_token = tokenizer.end_image_token
        self.image_token = tokenizer.context_image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.context_image_token_id

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    def _insert_media_placeholders(
        self,
        text: list[str],
        image_pixel_values,
        video_pixel_values,
        image_num_patches: list[int],
        video_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
        video_num_patches_indices: np.ndarray,
        video_patch_indices: np.ndarray,
    ):
        """
        Processes interleaved text with <image> and <video> placeholders, replacing them with appropriate
        image and video tokens while keeping track of the patches used.
        """
        image_index = 0
        video_index = 0
        processed_text = []
        image_video_patches = []
        replace_strings = []
        # Support interleaved image and video in prompts:
        # Processed patches of images and videos are inserted in `image_video_patches` in the order they appear in the prompts
        for prompt in text:
            new_prompt = prompt
            while self.image_token in new_prompt or self.video_token in new_prompt:
                if self.image_token in new_prompt and (
                    self.video_token not in new_prompt
                    or new_prompt.index(self.image_token) < new_prompt.index(self.video_token)
                ):
                    # Get the slice of patches corresponding to the current image
                    start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                    end_index = image_num_patches_indices[image_index]
                    image_video_patches.append(image_pixel_values[start_index:end_index])
                    # Replace the corresponding image placeholder with the correct number of image tokens
                    new_prompt = new_prompt.replace(self.image_token, "<placeholder>", 1)
                    replace_strings.append(
                        f"{self.start_image_token}{self.image_token * self.image_seq_length * image_num_patches[image_index]}{self.end_image_token}"
                    )
                    image_index += 1
                else:
                    # Get the slice of patches corresponding to the current video
                    # Here we need to account for both the multiple video frames and the potential multiple patches per frame
                    # As of now, InternVL only supports one patch per frame, but we keep the code flexible for future updates
                    current_patch_index = video_patch_indices[video_index - 1] if video_index > 0 else 0
                    end_patch_index = video_patch_indices[video_index]
                    start_index = video_num_patches_indices[current_patch_index] if video_index > 0 else 0
                    end_index = video_num_patches_indices[end_patch_index - 1]
                    image_video_patches.append(video_pixel_values[start_index:end_index])
                    # Get the number of patches per frame and replace the video placeholder with the correct number of image tokens
                    num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                    video_prompt = "\n".join(
                        f"Frame{i + 1}: {self.start_image_token}{self.image_token * self.image_seq_length * num_patches[i]}{self.end_image_token}"
                        for i in range(len(num_patches))
                    )
                    replace_strings.append(video_prompt)
                    new_prompt = new_prompt.replace(self.video_token, "<placeholder>", 1)
                    video_index += 1
            while "<placeholder>" in new_prompt:
                replace_str = replace_strings.pop(0)
                new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)
            processed_text.append(new_prompt)

        return processed_text, image_video_patches, image_index, video_index

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images and videos separately, as videos don't support crop_to_patches
        image_num_patches = []
        video_num_patches = []
        image_videos_inputs = {}
        image_pixel_values = None
        video_pixel_values = None
        image_num_patches_indices = np.array([0])
        video_patch_indices = np.array([0])
        video_num_patches_indices = np.array([0])
        if images is not None:
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)
        if videos is not None:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            video_patch_indices = np.cumsum(num_frames_per_video)
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_num_patches = [1 for frames in num_frames_per_video for _ in range(frames)]
            video_pixel_values = video_inputs.pop("pixel_values_videos").flatten(0, 1)
            video_num_patches_indices = np.cumsum(video_num_patches)

        if images is not None or videos is not None:
            text, image_video_patches, image_index, video_index = self._insert_media_placeholders(
                text,
                image_pixel_values,
                video_pixel_values,
                image_num_patches,
                video_num_patches,
                image_num_patches_indices,
                video_num_patches_indices,
                video_patch_indices,
            )
            if images is not None and image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")
            if videos is not None and video_index != len(videos):
                raise ValueError("Number of video placeholders in the prompt does not match the number of videos.")

            # Concatenate the interleaved image and video patches (function agnostic to the patches type (list, numpy array, torch tensor))
            image_videos_inputs = {"pixel_values": concatenate_list(image_video_patches)}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(data={**text_inputs, **image_videos_inputs}, tensor_type=return_tensors)

    def sample_indices_fn(
        self, metadata: VideoMetadata, num_frames: Optional[int] = None, initial_shift: Union[bool, float, int] = True
    ):
        """
        The function to generate indices of frames to sample from a video.

        Args:
            metadata (`VideoMetadata`):
                `VideoMetadata` object containing metadata about the video, such as "total_num_frames" or "fps".
            num_frames (`int`, *optional*):
                Number of frames to sample uniformly. If None, all frames are sampled.
            initial_shift (`bool`, `float` or `int`, defaults to `0`):
                The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

        Returns:
            `np.ndarray`: Array of frame indices to sample.
        """
        num_frames = num_frames if num_frames is not None else metadata.total_num_frames

        if initial_shift is True:
            initial_shift = metadata.total_num_frames / num_frames / 2
        indices = np.arange(initial_shift, metadata.total_num_frames, metadata.total_num_frames / num_frames).astype(
            int
        )
        return indices

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)

    # TODO: raushan, has to be public method under `VideoProcessorBase` when API is added
    def _load_video_for_model(
        self,
        video: Union[str, "VideoInput"],
        num_frames: Optional[int],
        backend: str = "pyav",
        initial_shift: bool = True,
        **kwargs,
    ) -> np.array:
        """
        Loads `video` to a numpy array.

        Args:
            video (`str` or `VideoInput`):
                The video to convert to the numpy array format. Can be a link to video or local path.
            num_frames (`int`, *optional*):
                Number of frames to sample uniformly. If not passed, the whole video is loaded.
            backend (`str`, *optional*, defaults to `"pyav"`):
                The backend to use when loading the video. Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "pyav".
            initial_shift (`bool`, *optional*, defaults to `True`):
                The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

        Returns:
            Tuple[`np.array`, Dict]: A tuple containing:
                - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
                - Metadata dictionary.
        """

        def sample_indices_fn_func(metadata, **fn_kwargs):
            return self.sample_indices_fn(metadata, num_frames=num_frames, initial_shift=initial_shift, **fn_kwargs)

        video, metadata = load_video(video, backend=backend, sample_indices_fn=sample_indices_fn_func)
        return video, metadata


__all__ = ["InternVLProcessor"]
