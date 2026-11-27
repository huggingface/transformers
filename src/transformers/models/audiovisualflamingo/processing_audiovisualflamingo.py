# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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

import math
from itertools import islice

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import OmniModalProcessingKwargs, OmniModalProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...video_utils import VideoInput, VideoMetadata


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@auto_docstring
class AudioVisualFlamingoProcessor(OmniModalProcessorMixin):
    text_kwargs = {"padding": True, "padding_side": "left", "return_tensors": "pt"}
    videos_kwargs = {"return_metadata": True}
    # Frame and audio token times are kept as Python floats, so that casting the inputs to a lower precision doesn't
    # affect them: their time embedding needs more precision
    skip_tensor_conversion = OmniModalProcessorMixin.skip_tensor_conversion + ["video_timestamps", "audio_timestamps"]
    # The audio encoder outputs one token every 4 mel frames
    audio_frames_per_token = 4

    def __init__(
        self,
        image_processor,
        video_processor,
        feature_extractor,
        tokenizer,
        chat_template=None,
        image_token="<image>",
        video_token="<vila/video>",
        audio_token="<sound>",
        image_seq_length=256,
        video_temporal_pool_size=8,
        video_segment_duration=30,
        max_audio_len=600,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Placeholder token of images.
        video_token (`str`, *optional*, defaults to `"<vila/video>"`):
            Placeholder token of videos.
        audio_token (`str`, *optional*, defaults to `"<sound>"`):
            Placeholder token of audio.
        image_seq_length (`int`, *optional*, defaults to 256):
            Number of tokens of an image tile, and of a group of video frames.
        video_temporal_pool_size (`int`, *optional*, defaults to 8):
            Number of consecutive video frames whose tokens are averaged together by the model.
        video_segment_duration (`int`, *optional*, defaults to 30):
            Duration in seconds of the segments in which the visual and audio tokens of a video are interleaved.
        max_audio_len (`int`, *optional*, defaults to 600):
            Maximum duration in seconds of an audio. Longer audio is truncated.
        """
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        # Each modality span starts and ends with these tokens
        self.span_tokens = {
            image_token: ("<|image_bos|>", "<|image_eos|>"),
            video_token: ("<|video_bos|>", "<|video_eos|>"),
            audio_token: ("<|sound_bos|>", "<|sound_eos|>"),
        }
        self.image_seq_length = image_seq_length
        self.video_temporal_pool_size = video_temporal_pool_size
        self.video_segment_duration = video_segment_duration
        self.max_audio_len = max_audio_len
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[OmniModalProcessingKwargs],
    ):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        # Images are grouped by prompt, since only single-image prompts are tiled
        if text is not None and images is not None:
            images = make_flat_list_of_images(images)
            num_images = [prompt.count(self.image_token) for prompt in text]
            if sum(num_images) == len(images):
                images = iter(images)
                images = [list(islice(images, count)) for count in num_images]
        return images, text, videos, audio

    def get_text_with_replacements(self, text, images_replacements=(), videos_replacements=(), audio_replacements=()):
        # As in training, all the images (resp. audio) of a prompt are wrapped by a single pair of begin/end tokens
        images_replacements, audio_replacements = list(images_replacements), list(audio_replacements)
        for token, replacements in ((self.image_token, images_replacements), (self.audio_token, audio_replacements)):
            bos, eos = self.span_tokens[token]
            start = 0
            for prompt in text:
                if (count := prompt.count(token)) and replacements:
                    replacements[start] = bos + replacements[start]
                    replacements[start + count - 1] += eos
                start += count
        return super().get_text_with_replacements(text, images_replacements, videos_replacements, audio_replacements)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # The image is encoded at the resolution of its last S2 scale
        num_rows, num_cols = image_inputs["image_grid_hw"][image_idx, -1].tolist()
        # As in training, each image ends with a newline
        return self.image_token * (num_rows * num_cols * self.image_seq_length) + "\n"

    def _process_audio(self, audio: AudioInput, **kwargs):
        # Each audio is padded to a multiple of the 30 s encoder window and its log-mel spectrogram is normalized as a
        # whole, before being split into windows by the model. Audio is processed separately, so that its features
        # don't depend on the other audio of the batch.
        sampling_rate = self.feature_extractor.sampling_rate
        window_size = self.feature_extractor.n_samples
        # Audio kwargs (e.g. `device`) are passed to the feature extractor, but its padding is set to whole windows
        audio_kwargs = {"sampling_rate": sampling_rate, **kwargs, "padding": "max_length", "return_tensors": "pt"}
        input_features = []
        for waveform in audio:
            if len(waveform) > self.max_audio_len * sampling_rate:
                logger.warning_once(f"Audio longer than {self.max_audio_len} seconds is truncated.")
            waveform = waveform[: self.max_audio_len * sampling_rate]
            audio_kwargs["max_length"] = window_size * max(1, math.ceil(len(waveform) / window_size))
            features = self.feature_extractor(waveform, **audio_kwargs).input_features
            input_features.append(features[0].T)

        num_frames = torch.tensor([len(features) for features in input_features])
        # Each audio token is placed at the middle of its mel frames
        token_duration = self.feature_extractor.hop_length / sampling_rate * self.audio_frames_per_token
        num_tokens = int(num_frames.max()) // self.audio_frames_per_token
        timestamps = [index * token_duration + token_duration / 2 for index in range(num_tokens)]
        audio_inputs = BatchFeature(
            {
                "input_features": torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True).transpose(1, 2),
                "input_features_mask": torch.arange(num_frames.max()) < num_frames[:, None],
                "audio_timestamps": [list(timestamps) for _ in input_features],
            }
        )
        num_audio_tokens = (num_frames // self.audio_frames_per_token).tolist()
        return audio_inputs, [self.audio_token * num_tokens for num_tokens in num_audio_tokens]

    def _process_videos(
        self,
        videos: VideoInput,
        audio: AudioInput | None = None,
        processed_audio: dict | None = None,
        video_to_audio_indices: list | None = None,
        **kwargs: Unpack[OmniModalProcessingKwargs],
    ):
        processed_videos = self.video_processor(videos, **kwargs["videos_kwargs"])
        metadata = processed_videos["video_metadata"]
        # Frames are assumed to be one second apart when the frame rate is unknown
        frame_rates = [video_metadata.fps or 1.0 for video_metadata in metadata]
        processed_videos["video_timestamps"] = [
            [index / fps for index in video_metadata.frames_indices]
            for video_metadata, fps in zip(metadata, frame_rates)
        ]

        video_replacements = []
        for video_idx, (video_metadata, fps) in enumerate(zip(metadata, frame_rates)):
            num_groups = math.ceil(len(video_metadata.frames_indices) / self.video_temporal_pool_size)
            num_video_tokens = num_groups * self.image_seq_length
            audio_idx = video_to_audio_indices[video_idx] if video_to_audio_indices else None
            if audio_idx is None:
                video_bos, video_eos = self.span_tokens[self.video_token]
                video_replacements.append(video_bos + self.video_token * num_video_tokens + video_eos)
                continue

            audio_mask = processed_audio["input_features_mask"][audio_idx]
            num_audio_tokens = int(audio_mask.sum()) // self.audio_frames_per_token
            audio_duration = min(len(audio[audio_idx]) / self.feature_extractor.sampling_rate, self.max_audio_len)
            replacement, num_used_audio_tokens = self._interleave_video_and_audio(
                video_metadata, fps, num_video_tokens, num_audio_tokens, audio_duration
            )
            # Only the audio overlapping the video is kept
            audio_mask[num_used_audio_tokens * self.audio_frames_per_token :] = False
            video_replacements.append(replacement)
        return processed_videos, video_replacements

    def _interleave_video_and_audio(
        self,
        metadata: VideoMetadata,
        fps: float,
        num_video_tokens: int,
        num_audio_tokens: int,
        audio_duration: float,
    ) -> tuple[str, int]:
        """
        Splits a video in segments of `video_segment_duration` seconds, and returns the placeholder string in which the
        visual tokens of each segment are followed by its audio tokens, along with the number of used audio tokens.
        """
        frames_indices = metadata.frames_indices
        duration = metadata.total_num_frames / fps
        mel_frames_per_second = self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        num_frames = len(frames_indices)

        segments = []
        frame = video_end = audio_end = 0
        for segment in range(math.ceil(duration / self.video_segment_duration)):
            segment_start = segment * self.video_segment_duration
            segment_end = segment_start + self.video_segment_duration
            while frame < num_frames and frames_indices[frame] < segment_end * fps:
                frame += 1
            new_video_end = math.ceil(frame * num_video_tokens / num_frames)
            new_audio_end = audio_end
            overlap_end = min(segment_end, duration, audio_duration)
            if segment_start < overlap_end:
                num_mel_frames = round(overlap_end * mel_frames_per_second)
                new_audio_end = min(math.ceil(num_mel_frames / self.audio_frames_per_token), num_audio_tokens)
            segments.append(
                self.video_token * (new_video_end - video_end)
                + "\n"
                + self.audio_token * (new_audio_end - audio_end)
                + "\n"
            )
            video_end, audio_end = new_video_end, new_audio_end

        # As in training, the audio track keeps an empty span of its own since its tokens are placed in the video
        audio_bos, audio_eos = self.span_tokens[self.audio_token]
        video_bos, video_eos = self.span_tokens[self.video_token]
        return audio_bos + audio_eos + video_bos + "".join(segments) + video_eos, audio_end

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["video_timestamps", "input_features_mask", "audio_timestamps"]


__all__ = ["AudioVisualFlamingoProcessor"]
