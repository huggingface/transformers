# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
Processor class for Qwen2.5Omni.
"""

import logging
import re
from typing import List, Optional, Union

import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput


class Qwen2_5_VLVideosKwargs(VideosKwargs):
    fps: Optional[List[int]] = None
    use_audio_in_video: Optional[bool] = None
    seconds_per_chunk: Optional[float] = None
    position_id_per_seconds: Optional[int] = None


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_VLVideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


class Qwen2_5OmniProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2.5Omni processor.
    [`Qwen2_5OmniProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5OmniProcessor.__call__`] and [`~Qwen2_5OmniProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwargs` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].pop("fps", None)

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1)) // 2 + 1
            audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            audio_inputs = {}
            audio_lengths = None

        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
        else:
            images_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            if fps is None:
                fps = [2.0] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        # Extend mm token length
        merge_length = self.image_processor.merge_size**2
        audio_index = 0
        image_index = 0
        video_index = 0

        for i in range(len(text)):
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, text[i])])

            positions.sort(key=lambda x: x[0])
            for _, special_token in positions:
                if audio is not None and special_token == self.audio_token:
                    text[i] = text[i].replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * audio_lengths[audio_index],
                        1,
                    )
                    audio_index += 1
                elif images is not None and special_token == self.image_token:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|image_placeholder|>" * (image_grid_thw[image_index].prod() // merge_length),
                        1,
                    )
                    image_index += 1
                elif videos is not None and special_token == self.video_token:
                    if not use_audio_in_video:
                        text[i] = text[i].replace(
                            self.video_token,
                            "<|video_placeholder|>" * (video_grid_thw[video_index].prod() // merge_length),
                            1,
                        )
                        video_index += 1
                    else:
                        audio_t_index = torch.arange(audio_lengths[audio_index])
                        video_t_index = (
                            torch.arange(video_grid_thw[video_index][0])
                            .view(-1, 1, 1)
                            .expand(
                                -1,
                                video_grid_thw[video_index][1] // self.image_processor.merge_size,
                                video_grid_thw[video_index][2] // self.image_processor.merge_size,
                            )
                            .flatten()
                            * videos_inputs["video_second_per_grid"][video_index]
                            * position_id_per_seconds
                        ).long()
                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                        placeholder_string = str()
                        placeholder_string += self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                placeholder_string += "<|video_placeholder|>" * (
                                    video_chunk_index[1] - video_chunk_index[0]
                                )
                            if audio_chunk_index is not None:
                                placeholder_string += "<|audio_placeholder|>" * (
                                    audio_chunk_index[1] - audio_chunk_index[0]
                                )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        text[i] = text[i].replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )
                        audio_index += 1
                        video_index += 1

            text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)
            text[i] = text[i].replace("<|image_placeholder|>", self.image_token)
            text[i] = text[i].replace("<|video_placeholder|>", self.video_token)

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def get_chunked_index(self, t_index, t_ntoken_per_chunk):
        """
        Divides a list of indices into chunks based on token count per chunk.
        Args:
            t_index: A list of integers representing token indices.
            t_ntoken_per_chunk: The maximum number of tokens allowed in each chunk.
        Returns:
            A list of tuples, where each tuple represents the start and end
            indices of a chunk.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(t_index):  # skip eos token
                if t_index[i] >= current_chunk * t_ntoken_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(t_index))

        return list(_iter())

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

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            if (
                conversation[0]["role"] != "system"
                or conversation[0]["content"][0]["text"]
                != "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ):
                logging.warning(
                    "System prompt modified, audio output may not work as expected. "
                    + "Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'"
                )
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen2_5OmniProcessor"]
