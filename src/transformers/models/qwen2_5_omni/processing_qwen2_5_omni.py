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
from typing import Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ...video_utils import VideoInput


# Redefine kwargs for videos because Qwen-Omni uses some kwargs for processing omni
# and does not use them in video processor class
class Qwen2_5_OmniVideosKwargs(VideosKwargs, total=False):
    """
    min_pixels (`int`, *optional*):
        Minimum number of pixels (height × width) for video frames after resizing. Frames smaller than this
        threshold will be upscaled to meet the minimum requirement.
    max_pixels (`int`, *optional*):
        Maximum number of pixels (height × width) for video frames after resizing. Frames larger than this
        threshold will be downscaled to fit within the limit.
    patch_size (`int`, *optional*):
        The spatial patch size used by the vision encoder. Video frames are divided into patches of this size
        in both height and width dimensions.
    temporal_patch_size (`int`, *optional*):
        The temporal patch size used by the vision encoder. This determines how many consecutive frames are
        grouped together as a single temporal patch.
    merge_size (`int`, *optional*):
        The merge size used for combining spatial patches. Multiple patches are merged together to reduce the
        sequence length while maintaining spatial information.
    min_frames (`int`, *optional*):
        Minimum number of frames to extract from the video. Videos with fewer frames will be padded or repeated
        to meet this requirement.
    max_frames (`int`, *optional*):
        Maximum number of frames to extract from the video. Longer videos will be truncated or sampled to fit
        within this limit.
    use_audio_in_video (`bool`, *optional*, defaults to `False`):
        Whether to incorporate audio information when processing videos. When enabled, audio tokens are
        interleaved with video tokens based on temporal alignment, creating a unified multimodal representation.
    seconds_per_chunk (`float`, *optional*, defaults to `2.0`):
        The duration (in seconds) of each video chunk when splitting long videos. This parameter controls how
        videos are divided into temporal segments for processing.
    position_id_per_seconds (`int` or `float`, *optional*, defaults to `25`):
        The number of position IDs allocated per second of video. This parameter controls the temporal resolution
        of position embeddings and is used to align video tokens with audio tokens when `use_audio_in_video=True`.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    use_audio_in_video: bool
    seconds_per_chunk: float
    position_id_per_seconds: Union[int, float]


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_OmniVideosKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
            "size": {
                "shortest_edge": 128 * 28 * 28,
                "longest_edge": 768 * 28 * 28,
            },
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


@auto_docstring
class Qwen2_5OmniProcessor(ProcessorMixin):
    def __init__(
        self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    @auto_docstring
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: Optional[ImageInput] = None,
        videos: Optional[VideoInput] = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
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

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            audio_lengths = iter((input_lengths - 2) // 2 + 1)
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])

            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            video_grid_thw = videos_inputs["video_grid_thw"]
            second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            videos_inputs["video_second_per_grid"] = second_per_grid_ts

            video_grid_thw = iter(video_grid_thw)
            video_second_per_grid = iter(second_per_grid_ts)
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        if images is not None or videos is not None or audio is not None:
            text = self.replace_multimodal_special_tokens(
                text,
                audio_lengths,
                image_grid_thw,
                video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                use_audio_in_video=use_audio_in_video,
                position_id_per_seconds=position_id_per_seconds,
                seconds_per_chunk=seconds_per_chunk,
            )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        tokens_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                placeholder_string += "<|video_placeholder|>" * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                placeholder_string += "<|audio_placeholder|>" * audio_seq_length
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        is_batched = False
        if isinstance(conversations[0], dict):
            conversations = [conversations]
            is_batched = True

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
        if is_batched:
            conversations = conversations[0]

        return super().apply_chat_template(conversations, chat_template, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs[0], skip_special_tokens=skip_special_tokens, **kwargs)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[Inion[str, np.ndarray]]`: The decoded text or generated audio.
        """
        if generation_mode is None or generation_mode == "text":
            return self.post_process_image_text_to_text(
                generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
            )

        elif generation_mode == "audio":
            # model supports only bs=1, so we will never get several audio outputs
            audio = generated_outputs[1].reshape(-1).detach().cpu().numpy()
            return [audio]

        else:
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only `text` and `audio"
            )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen2_5OmniProcessor"]
