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

    pass


class File(Media):
    """File-based media object."""

    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    """Image media object."""

    pass


class Video(File):
    """Video media object."""

    pass


class Sound(File):
    """Sound/music audio media object."""

    def __init__(self, path, extension: str | None = None) -> None:
        self.path = path
        self.extension = extension


def make_list(obj: Any) -> list:
    """Convert object to list if not already a list."""
    return obj if isinstance(obj, list) else [obj]


def _extract_image(image: Image | PIL.Image.Image) -> PIL.Image.Image:
    """Extract PIL image from Image object or return PIL image as-is."""
    if isinstance(image, Image):
        image = load_image(image.path)
    return image.convert("RGB")


def _load_video_bytesio(
    video_bytesio: BytesIO, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> list[PIL.Image.Image]:
    """Load video from BytesIO object by writing to temporary file."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames, load_aud=load_aud, config=config)


def get_overlap(inp1, inp2):
    """Return overlapping [start, end) interval for two [start, end] pairs."""
    overlap_start = max(inp1[0], inp2[0])
    overlap_end = min(inp1[1], inp2[1])
    return (overlap_start, overlap_end) if overlap_start < overlap_end else None


def _load_video(
    video_path: str, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> list[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    vidcap = cv2.VideoCapture(video_path)
    try:
        # Load audio if available and needed
        audio_info = None
        if load_aud:
            try:
                aud_feature, audio_info = _load_speech(video_path, config)
            except Exception:
                aud_feature = None
        else:
            aud_feature = None

        # Find the last frame as frame count might not be accurate
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count > 0:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if vidcap.grab():
                break
            frame_count -= 1
        else:
            raise ValueError(f"Video '{video_path}' has no frames.")

        # Extract frames uniformly
        indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_duration = frame_count / fps

        segment_vis_indices_list = None
        segment_aud_indices_list = None

        # When load_audio_in_video and interleaved_vis_aud_in_video is True, we need to load frames for each video segment
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
                overlap = get_overlap([clip_start_sec, clip_end_sec], [audio_start_sec, audio_end_sec])
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


def _extract_video(video: Video, config: PretrainedConfig) -> list[PIL.Image.Image]:
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


def _load_speech(speech, config: PretrainedConfig):
    speech_path = speech if isinstance(speech, str) else speech.path

    if speech_path is None:
        return None

    if config.audio_chunk_length and not (
        isinstance(config.audio_chunk_length, str) and "max" in config.audio_chunk_length
    ):
        try:
            config.audio_chunk_length = int(config.audio_chunk_length)
        except Exception as e:
            print(f"Error setting audio_chunk_length: {e}")
            raise e

    audio_n_samples_limit = config.audio_chunk_length * config.audio_sampling_rate

    def load_wav(path_or_file):
        audio, sample_rate = librosa.load(path_or_file, sr=config.audio_sampling_rate)
        ori_audio_duration = audio.shape[0] / sample_rate
        return audio, ori_audio_duration

    def get_audio(audio_data, audio_n_samples):
        if isinstance(audio_data, decord.audio_reader.AudioReader):
            ori_n_samples = audio_data.shape[1]
        else:
            ori_n_samples = audio_data.shape[0]

        audio_start_sample_id = 0
        audio_end_sample_id = ori_n_samples

        load_max_audio = isinstance(config.audio_chunk_length, str) and "max" in config.audio_chunk_length
        if hasattr(config, "random_audio_sample") and not load_max_audio:
            if ori_n_samples > audio_n_samples:
                audio_start_sample_id = random.randint(0, ori_n_samples - audio_n_samples)
                audio_end_sample_id = audio_start_sample_id + audio_n_samples
        else:
            if load_max_audio:
                if "_" in config.audio_chunk_length:
                    max_audio_chunk_length = int(config.audio_chunk_length.split("_")[1])
                    max_audio_n_samples = max_audio_chunk_length * config.audio_sampling_rate
                    audio_n_samples = min(ori_n_samples, max_audio_n_samples)
                    audio_end_sample_id = audio_n_samples
                else:
                    audio_n_samples = ori_n_samples
                    audio_end_sample_id = audio_n_samples
            else:
                audio_end_sample_id = min(audio_n_samples, ori_n_samples)

        if isinstance(audio_data, decord.audio_reader.AudioReader):
            audio_data = audio_data[audio_start_sample_id:audio_end_sample_id].asnumpy()[0]
        else:
            audio_data = audio_data[audio_start_sample_id:audio_end_sample_id]

        return audio_data, audio_n_samples, audio_start_sample_id, audio_end_sample_id

    if isinstance(speech_path, BytesIO):
        if getattr(speech, "extension", None) != ".wav":
            raise ValueError(f"Unsupported audio extension: {getattr(speech, 'extension', None)}")
        speech_data, ori_audio_duration = load_wav(speech_path)
        speech_data, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(
            speech_data, audio_n_samples_limit
        )
    elif isinstance(speech_path, str) and ".mp4" in speech_path:
        audio_reader = AudioReader(speech_path, ctx=cpu(0), sample_rate=config.audio_sampling_rate, mono=True)
        ori_audio_duration = audio_reader.shape[1] / config.audio_sampling_rate
        speech_data, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(
            audio_reader, audio_n_samples_limit
        )
    else:
        if not isinstance(speech_path, str) or not os.path.exists(speech_path):
            raise ValueError(f"File {speech_path} does not exist")
        speech_data, ori_audio_duration = load_wav(speech_path)
        speech_data, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(
            speech_data, audio_n_samples_limit
        )

    speech_data = speech_data.astype(np.float32)
    audio_n_samples = int(
        np.ceil(speech_data.shape[0] / (config.audio_sampling_rate * 30)) * (config.audio_sampling_rate * 30)
    )

    speech_data = whisper.pad_or_trim(speech_data, length=audio_n_samples)

    audio_info = {
        "new_audio_chunk_length": int(audio_n_samples // config.audio_sampling_rate),
        "new_audio_n_samples": audio_n_samples,
        "ori_audio_duration": ori_audio_duration,
        "audio_start_sec": audio_start_sample_id / config.audio_sampling_rate,
        "audio_end_sample_sec": audio_end_sample_id / config.audio_sampling_rate,
    }

    return speech_data, audio_info


def _extract_sound(sound: Sound, config: PretrainedConfig):
    frames, audio_info = _load_speech(sound, config)
    return frames, audio_info


def extract_media(
    messages: list[dict[str, Any]],
    config: PretrainedConfig | None = None,
) -> dict[str, list[Any]]:
    media = defaultdict(list)

    if not hasattr(config, "load_audio_in_video"):
        print("Warning: load_audio_in_video not in config, set to False")
        config.load_audio_in_video = False

    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        print(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
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
                output, audio_info = _extract_sound(part, config)
                if output is not None:
                    media["sound"].append(output)
                    media["audio_info"].append(audio_info)
                    text += MEDIA_TOKENS["sound"]
            else:
                print(f"part: {part}")
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
