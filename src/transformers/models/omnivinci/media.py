# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import cv2
import decord
import librosa
import numpy as np
import PIL
import PIL.Image
import whisper
from decord import AudioReader, cpu

from transformers import PretrainedConfig
from transformers.image_utils import load_image

from .configuration_omnivinci import MEDIA_TOKENS


class Media:
    """Base class for media objects."""

    __slots__ = ()


@dataclass(slots=True)
class File(Media):
    """File-based media object."""

    path: str | BytesIO


class Image(File):
    """Image media object."""

    pass


class Video(File):
    """Video media object."""

    pass


@dataclass(slots=True)
class Sound(File):
    """Sound/music audio media object."""

    extension: str | None = None


def _extract_image(image: Image | PIL.Image.Image) -> PIL.Image.Image:
    """Extract PIL image from Image object or return PIL image as-is."""
    if isinstance(image, Image):
        image = load_image(image.path)
    return image.convert("RGB")


def _load_video_bytesio(
    video_bytesio: BytesIO, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, Any]]:
    """Load video from BytesIO object by writing to temporary file."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        video_bytesio.seek(0)
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames, load_aud=load_aud, config=config)


def _load_video(
    video_path: str, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, Any]]:
    """Load video frames (and optionally aligned audio features) from file or frame directory."""
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        if not frame_paths:
            raise ValueError(f"Video frame directory '{video_path}' is empty.")
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        output_frames = []
        for index in indices:
            with PIL.Image.open(frame_paths[index]) as frame:
                output_frames.append(frame.convert("RGB"))
        output_frame_times = [float(index) for index in indices]
        video_info = {
            "video_path": video_path,
            "has_audio": False,
            "video_duration": float(len(frame_paths)),
            "audio_info": None,
            "video_frame_times": output_frame_times,
        }
        return output_frames, None, video_info

    vidcap = cv2.VideoCapture(video_path)
    try:
        # Load audio if available and needed.
        aud_feature = None
        audio_info = None
        if load_aud:
            try:
                aud_feature, audio_info = _load_speech(video_path, config)
            except Exception:
                aud_feature = None

        # Find the last valid frame since cv2 frame_count may be inaccurate.
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count > 0:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if vidcap.grab():
                break
            frame_count -= 1
        else:
            raise ValueError(f"Video '{video_path}' has no frames.")

        # Extract frames uniformly.
        indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 1.0
        video_duration = frame_count / fps

        segment_vis_indices_list = None
        segment_aud_indices_list = None

        # When loading interleaved visual/audio clips, build segment indices for both modalities.
        if config.load_audio_in_video and config.interleaved_vis_aud_in_video and aud_feature is not None:
            segment_duration = config.interleaved_video_segment_duration
            if segment_duration == -1:
                raise ValueError("video_segment_duration is not set")

            segment_vis_indices_list = []
            segment_aud_indices_list = []
            segment_counts = np.ceil(video_duration / segment_duration).astype(int)

            audio_start_sec = audio_info["audio_start_sec"]
            audio_end_sec = audio_info["audio_end_sample_sec"]
            stft_frames_per_second = config.audio_sampling_rate // config.audio_hop_length

            idx = 0
            aud_sample_start_idx = 0
            for i in range(segment_counts):
                end_frame = min((i + 1) * segment_duration * fps, frame_count)

                segment_indices = []
                while idx < len(indices) and indices[idx] < end_frame:
                    segment_indices.append(indices[idx])
                    idx += 1
                segment_vis_indices_list.append(segment_indices)

                clip_start_sec = i * segment_duration
                clip_end_sec = min(clip_start_sec + segment_duration, video_duration)

                # get the audio indices for the current clip
                overlap_start = max(clip_start_sec, audio_start_sec)
                overlap_end = min(clip_end_sec, audio_end_sec)
                overlap = (overlap_start, overlap_end) if overlap_start < overlap_end else None
                if overlap is not None:
                    aud_sample_end_idx = round((overlap[1] - audio_start_sec) * stft_frames_per_second)
                    segment_aud_indices_list.append([aud_sample_start_idx, aud_sample_end_idx])
                    aud_sample_start_idx = aud_sample_end_idx
                else:
                    segment_aud_indices_list.append([])

        frames = {}
        frame_times = {}
        for index in indices:
            if index in frames:
                continue
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = vidcap.read()
            if not success:
                print(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[index] = PIL.Image.fromarray(frame)
            frame_times[index] = index / fps

        output_frames = [frames[index] for index in indices if index in frames]
        output_frame_times = [frame_times[index] for index in indices if index in frame_times]

        video_info = {
            "video_path": video_path,
            "has_audio": aud_feature is not None,
            "video_duration": video_duration,
            "audio_info": audio_info,
            "video_frame_times": output_frame_times,
        }
        if audio_info is not None:
            audio_info["video_path"] = video_path

        if segment_vis_indices_list is not None:
            new_segment_vis_indices_list = []
            processed_frame_index = 0
            for segment_indices in segment_vis_indices_list:
                new_segment_vis_indices_list.append([])
                for index in segment_indices:
                    if index in frames:
                        new_segment_vis_indices_list[-1].append(processed_frame_index)
                        processed_frame_index += 1

            video_info.update(
                {
                    "segment_vis_indices_list": new_segment_vis_indices_list,
                    "segment_aud_indices_list": segment_aud_indices_list,
                    "expected_frame_count": len(indices),
                }
            )

        return output_frames, aud_feature, video_info
    finally:
        vidcap.release()


def _extract_video(
    video: Video, config: PretrainedConfig
) -> tuple[list[PIL.Image.Image], dict[str, Any]] | tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, Any]]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        print("Extracting frames from video with specified FPS is not supported yet. Ignored.")

    if isinstance(video.path, BytesIO):
        frames, aud_fea, video_info = _load_video_bytesio(
            video.path, num_frames=num_frames, config=config, load_aud=config.load_audio_in_video
        )
    else:
        frames, aud_fea, video_info = _load_video(
            video.path, num_frames=num_frames, config=config, load_aud=config.load_audio_in_video
        )

    if config.load_audio_in_video:
        return frames, aud_fea, video_info
    else:
        return frames, video_info


def _load_speech(speech: str | Sound, config: PretrainedConfig) -> tuple[np.ndarray, dict[str, Any]] | None:
    speech_path = speech if isinstance(speech, str) else speech.path
    if speech_path is None:
        return None

    sampling_rate = config.audio_sampling_rate
    audio_chunk_length = config.audio_chunk_length
    load_max_audio = isinstance(audio_chunk_length, str) and "max" in audio_chunk_length
    if load_max_audio:
        if "_" in audio_chunk_length:
            max_audio_chunk_length = int(audio_chunk_length.split("_", maxsplit=1)[1])
            audio_n_samples_limit = max_audio_chunk_length * sampling_rate
        else:
            audio_n_samples_limit = None
    else:
        try:
            audio_n_samples_limit = int(audio_chunk_length) * sampling_rate
        except Exception as error:
            raise ValueError(f"Error setting audio_chunk_length: {error}") from error

    def _load_wav(path_or_file: str | BytesIO) -> tuple[np.ndarray, float]:
        audio, loaded_sampling_rate = librosa.load(path_or_file, sr=sampling_rate)
        return audio, audio.shape[0] / loaded_sampling_rate

    def _slice_audio_window(audio_data: decord.audio_reader.AudioReader | np.ndarray) -> tuple[np.ndarray, int, int]:
        if isinstance(audio_data, decord.audio_reader.AudioReader):
            ori_n_samples = audio_data.shape[1]
        else:
            ori_n_samples = audio_data.shape[0]

        if audio_n_samples_limit is None:
            target_samples = ori_n_samples
        else:
            target_samples = min(audio_n_samples_limit, ori_n_samples)

        audio_start_sample_id = 0
        if (
            bool(getattr(config, "random_audio_sample", False))
            and not load_max_audio
            and ori_n_samples > target_samples
        ):
            audio_start_sample_id = random.randint(0, ori_n_samples - target_samples)
        audio_end_sample_id = audio_start_sample_id + target_samples

        if isinstance(audio_data, decord.audio_reader.AudioReader):
            audio_data = audio_data[audio_start_sample_id:audio_end_sample_id].asnumpy()[0]
        else:
            audio_data = audio_data[audio_start_sample_id:audio_end_sample_id]
        return audio_data, audio_start_sample_id, audio_end_sample_id

    if isinstance(speech_path, BytesIO):
        if getattr(speech, "extension", None) != ".wav":
            raise ValueError(f"Unsupported audio extension: {getattr(speech, 'extension', None)}")
        speech_data, ori_audio_duration = _load_wav(speech_path)
        speech_data, audio_start_sample_id, audio_end_sample_id = _slice_audio_window(speech_data)
    elif isinstance(speech_path, str) and speech_path.lower().endswith(".mp4"):
        audio_reader = AudioReader(speech_path, ctx=cpu(0), sample_rate=sampling_rate, mono=True)
        ori_audio_duration = audio_reader.shape[1] / sampling_rate
        speech_data, audio_start_sample_id, audio_end_sample_id = _slice_audio_window(audio_reader)
    else:
        if not isinstance(speech_path, str) or not os.path.exists(speech_path):
            raise ValueError(f"File {speech_path} does not exist")
        speech_data, ori_audio_duration = _load_wav(speech_path)
        speech_data, audio_start_sample_id, audio_end_sample_id = _slice_audio_window(speech_data)

    speech_data = speech_data.astype(np.float32, copy=False)
    audio_n_samples = int(np.ceil(speech_data.shape[0] / (sampling_rate * 30)) * (sampling_rate * 30))
    speech_data = whisper.pad_or_trim(speech_data, length=audio_n_samples)

    audio_info = {
        "new_audio_chunk_length": int(audio_n_samples // sampling_rate),
        "new_audio_n_samples": audio_n_samples,
        "ori_audio_duration": ori_audio_duration,
        "audio_start_sec": audio_start_sample_id / sampling_rate,
        "audio_end_sample_sec": audio_end_sample_id / sampling_rate,
    }
    return speech_data, audio_info


def extract_media(
    messages: list[dict[str, Any]],
    config: PretrainedConfig | None = None,
) -> dict[str, list[Any]]:
    if config is None:
        raise ValueError("`config` must be provided for media extraction.")

    media = defaultdict(list)

    if not hasattr(config, "load_audio_in_video"):
        print("Warning: load_audio_in_video not in config, set to False")
        config.load_audio_in_video = False

    def _strip_media_tokens(part: str) -> str:
        for token in MEDIA_TOKENS.values():
            if token in part:
                print(f"Media token '{token}' found in text: '{part}'. Removed.")
                part = part.replace(token, "").strip()
        return part

    for message in messages:
        text = ""
        parts = message["value"] if isinstance(message["value"], list) else [message["value"]]
        for part in parts:
            if isinstance(part, str):
                text += _strip_media_tokens(part)
            elif isinstance(part, (Image, PIL.Image.Image)):
                media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if config.load_audio_in_video:
                    output, aud_fea, video_info = _extract_video(part, config)
                    media["video"].append(output)
                    media["video_info"].append(video_info)
                    if aud_fea is not None:
                        media["sound"].append(aud_fea)
                        media["audio_info"].append(video_info["audio_info"])
                        text += MEDIA_TOKENS["sound"]
                else:
                    output, video_info = _extract_video(part, config)
                    media["video"].append(output)
                    media["video_info"].append(video_info)
                text += MEDIA_TOKENS["video"]
            elif isinstance(part, Sound):
                speech = _load_speech(part, config)
                if speech is not None:
                    output, audio_info = speech
                    media["sound"].append(output)
                    media["audio_info"].append(audio_info)
                    text += MEDIA_TOKENS["sound"]
            else:
                print(f"part: {part}")
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
