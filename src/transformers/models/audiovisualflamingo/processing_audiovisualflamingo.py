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

import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import PIL.Image
import torch

from transformers import WhisperFeatureExtractor
from transformers.audio_utils import load_audio, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import load_image
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.video_utils import load_video

from .configuration_audiovisualflamingo import AudioVisualFlamingoConfig


MEDIA_TOKENS = AudioVisualFlamingoConfig.media_tokens
MM_BOS_EOS_TOKENS = AudioVisualFlamingoConfig.mm_bos_eos_tokens


_AUDIOVISUALFLAMINGO_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"
    "{% endif %}"
    "{% for message in messages if message['content'] is not none %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for c in message['content'] %}"
    "{% if c.get('type') == 'text' %}{{ c['text'] }}"
    "{% elif c.get('type') == 'image' %}{{ '<image>' }}"
    "{% elif c.get('type') == 'video' %}{{ '<vila/video>' }}"
    "{% elif c.get('type') in ['audio', 'sound'] %}{{ '<sound>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

_VIDEO_METADATA_KEYS = {"fps", "frames_indices", "total_num_frames", "video_path", "video_url"}


def _looks_like_video_metadata(meta) -> bool:
    if meta is None:
        return False
    if isinstance(meta, dict):
        return bool(_VIDEO_METADATA_KEYS & set(meta.keys()))
    return any(hasattr(meta, key) for key in _VIDEO_METADATA_KEYS)


def _is_packed_media_item(item) -> bool:
    return isinstance(item, (tuple, list)) and len(item) == 2 and _looks_like_video_metadata(item[1])


def _is_audio_like(value) -> bool:
    return isinstance(value, (str, np.ndarray, torch.Tensor))


def _merge_media_config(target: defaultdict, source: defaultdict) -> None:
    for modality, config in source.items():
        for key, value in config.items():
            if isinstance(value, list):
                target[modality].setdefault(key, []).extend(value)
            elif key not in target[modality]:
                target[modality][key] = value
            elif target[modality][key] != value:
                raise ValueError(
                    f"Conflicting `{modality}` media config for key `{key}`: {target[modality][key]!r} != {value!r}"
                )


def _expand2square(pil_img, background_color):
    """Expand a non-square PIL image with padding to make it square."""
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = background_color[0]
    if width == height:
        return pil_img
    if width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = PIL.Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from candidate ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_s2_preprocess(image, s2_scales: list[int] | None = None, max_num=12, image_size=384):
    """Dynamically preprocess image using multi-scale S2 tiling."""
    if s2_scales is None:
        s2_scales = [384, 768, 1152]
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (s2_scales[-1] // s2_scales[0]) ** 2

    processed_images = []

    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0]) ** 2
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    return processed_images, (target_aspect_ratio[1], target_aspect_ratio[0])


def _process_image(image_input, data_args, enable_dynamic_s2=False):
    processor = data_args.image_processor
    image = load_image(image_input) if isinstance(image_input, str) else image_input
    image = image.convert("RGB")
    crop_size = getattr(data_args.image_processor, "crop_size", None)
    if crop_size is None:
        crop_size = getattr(data_args.image_processor, "size", None)
    if crop_size is None:
        raise ValueError("AudioVisualFlamingo image processor must define either `crop_size` or `size`.")
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = _dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    elif data_args.image_aspect_ratio == "pad":
        image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def _process_images(images, image_processor, model_cfg):
    """Process a batch of images using the model image processor."""
    model_cfg.image_processor = image_processor
    new_images = [_process_image(image, model_cfg) for image in images]

    if not all(x.shape == new_images[0].shape for x in new_images):
        raise ValueError("The shape of images in new_images is different!")
    if len(new_images[0].shape) == 4:
        return torch.cat(new_images, dim=0)
    if len(new_images[0].shape) == 3:
        return torch.stack(new_images, dim=0)
    raise ValueError(f"new_images rank does not equal to 4, rank: {len(new_images[0].shape)}")


def _add_mm_bos_eos_tokens(text: str) -> str:
    for k in ("image", "video", "sound"):
        _bos, _eos = MM_BOS_EOS_TOKENS[k]
        _media_token = MEDIA_TOKENS[k]
        if _media_token in text:
            text_parts = text.split(_media_token)
            text_parts[0] = text_parts[0] + _bos
            text_parts[-1] = _eos + text_parts[-1]
            text = _media_token.join(text_parts)
    return text


def _pad_or_trim_audio(audio: np.ndarray, length: int) -> np.ndarray:
    current_length = int(audio.shape[0])
    if current_length > length:
        return audio[:length]
    if current_length < length:
        return np.pad(audio, (0, length - current_length), mode="constant")
    return audio


def _resolve_sound_feature_size(config) -> int:
    audio_config = getattr(config, "audio_config", None)
    if isinstance(audio_config, dict):
        feature_size = audio_config.get("num_mel_bins")
    else:
        feature_size = getattr(audio_config, "num_mel_bins", None)
    if feature_size is None:
        feature_size = 128
    return int(feature_size)


def _resolve_target_audio_samples(sound: np.ndarray, audio_info, config) -> int:
    sampling_rate = config.audio_sampling_rate
    audio_n_samples = sound.shape[0]
    if isinstance(audio_info, dict) and audio_info.get("new_audio_n_samples") is not None:
        return int(audio_info["new_audio_n_samples"])

    target = int(np.ceil(audio_n_samples / (sampling_rate * 30)) * (sampling_rate * 30))
    if config.audio_chunk_length and not (
        isinstance(config.audio_chunk_length, str) and "max" in config.audio_chunk_length
    ):
        target = min(target, int(config.audio_chunk_length) * sampling_rate)
    return int(target)


def _extract_sound_features(
    sound_media: list,
    audio_infos: list | None,
    config,
    feature_extractor: WhisperFeatureExtractor | None = None,
) -> list:
    if audio_infos is None:
        audio_infos = []
    if audio_infos and len(audio_infos) != len(sound_media):
        raise ValueError("The number of audio info does not match the number of audio samples.")

    feature_size = _resolve_sound_feature_size(config)
    sampling_rate = config.audio_sampling_rate
    hop_length = config.audio_hop_length
    if feature_extractor is not None:
        feature_size = getattr(feature_extractor, "feature_size", feature_size)
        sampling_rate = getattr(feature_extractor, "sampling_rate", sampling_rate)
        hop_length = getattr(feature_extractor, "hop_length", hop_length)
    new_media = []

    for idx, sound in enumerate(sound_media):
        audio_info = audio_infos[idx] if idx < len(audio_infos) else None
        if isinstance(sound, dict) and "input_features" in sound:
            stft_features = sound
        else:
            if isinstance(sound, torch.Tensor):
                audio = sound.detach().cpu().float().numpy()
            else:
                audio = np.asarray(sound, dtype=np.float32)
            if audio.ndim != 1:
                audio = np.squeeze(audio)
            if audio.ndim != 1:
                raise ValueError(f"Expected mono waveform for sound input, got shape {audio.shape}.")

            cur_audio_n_samples = _resolve_target_audio_samples(audio, audio_info, config)
            cur_audio_duration = cur_audio_n_samples // sampling_rate
            whisper_feature_extractor = feature_extractor
            if (
                whisper_feature_extractor is None
                or getattr(whisper_feature_extractor, "chunk_length", None) != cur_audio_duration
            ):
                whisper_feature_extractor = WhisperFeatureExtractor(
                    feature_size=feature_size,
                    chunk_length=cur_audio_duration,
                    sampling_rate=sampling_rate,
                    hop_length=hop_length,
                )
            audio = _pad_or_trim_audio(audio, length=cur_audio_n_samples)
            stft_features = whisper_feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
            )

            if isinstance(audio_info, dict):
                audio_info["new_audio_chunk_length"] = cur_audio_duration
                audio_info["new_audio_n_samples"] = cur_audio_n_samples
                audio_info["audio_end_sample_sec"] = audio_info["audio_start_sec"] + cur_audio_duration
                audio_info["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]

        if isinstance(audio_info, dict) and "new_audio_n_stft_frames" not in audio_info:
            audio_info["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]
        new_media.append(stft_features)

    return new_media


def _load_audio_track_with_pyav(audio_path: str, sampling_rate: int) -> np.ndarray:
    import av

    with av.open(audio_path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream found in media container: {audio_path}")

        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
        chunks = []

        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]
            for resampled_frame in resampled_frames:
                chunks.append(resampled_frame.to_ndarray())

        flushed_frames = resampler.resample(None)
        if flushed_frames is not None:
            if not isinstance(flushed_frames, list):
                flushed_frames = [flushed_frames]
            for flushed_frame in flushed_frames:
                chunks.append(flushed_frame.to_ndarray())

    if not chunks:
        raise ValueError(f"No audio samples could be decoded from media container: {audio_path}")

    return np.concatenate(chunks, axis=-1)[0].astype(np.float32, copy=False)


def _load_audio_hf_with_info(audio_input, config) -> tuple[np.ndarray, dict[str, float | int]]:
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

    def _resolve_window(ori_n_samples: int) -> tuple[int, int]:
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
        return audio_start_sample_id, audio_end_sample_id

    if isinstance(audio_input, torch.Tensor):
        speech_data = audio_input.detach().cpu().float().numpy()
    elif isinstance(audio_input, np.ndarray):
        speech_data = audio_input
    elif isinstance(audio_input, str):
        try:
            speech_data = load_audio(audio_input, sampling_rate=sampling_rate)
        except Exception as audio_error:
            try:
                speech_data = _load_audio_track_with_pyav(audio_input, sampling_rate)
            except Exception:
                raise audio_error
    else:
        raise TypeError(
            "AudioVisualFlamingo audio inputs must be a path/URL, a numpy array, or a torch tensor. "
            f"Got {type(audio_input)!r}."
        )

    speech_data = np.asarray(speech_data, dtype=np.float32)
    if speech_data.ndim != 1:
        speech_data = np.squeeze(speech_data)
    if speech_data.ndim != 1:
        raise ValueError(f"Expected mono waveform for sound input, got shape {speech_data.shape}.")

    ori_n_samples = int(speech_data.shape[0])
    audio_start_sample_id, audio_end_sample_id = _resolve_window(ori_n_samples)
    ori_audio_duration = ori_n_samples / sampling_rate
    speech_data = speech_data[audio_start_sample_id:audio_end_sample_id]

    audio_n_samples = int(np.ceil(speech_data.shape[0] / (sampling_rate * 30)) * (sampling_rate * 30))
    speech_data = _pad_or_trim_audio(speech_data, length=audio_n_samples)

    audio_info = {
        "new_audio_chunk_length": int(audio_n_samples // sampling_rate),
        "new_audio_n_samples": audio_n_samples,
        "ori_audio_duration": ori_audio_duration,
        "audio_start_sec": audio_start_sample_id / sampling_rate,
        "audio_end_sample_sec": audio_end_sample_id / sampling_rate,
    }
    return speech_data, audio_info


def _coerce_video_frames_to_pil(video_frames) -> list[PIL.Image.Image]:
    if isinstance(video_frames, np.ndarray):
        if video_frames.ndim == 3:
            video_frames = np.expand_dims(video_frames, axis=0)
        if video_frames.ndim != 4:
            raise TypeError(f"Expected video array with 4 dimensions, got shape {video_frames.shape}.")
        return [PIL.Image.fromarray(frame).convert("RGB") for frame in video_frames]

    if isinstance(video_frames, (list, tuple)):
        output_frames = []
        for frame in video_frames:
            if isinstance(frame, PIL.Image.Image):
                output_frames.append(frame.convert("RGB"))
            else:
                output_frames.append(PIL.Image.fromarray(np.asarray(frame)).convert("RGB"))
        return output_frames

    raise TypeError(f"Unsupported video payload type for frame conversion: {type(video_frames)!r}")


def _extract_video_hf(
    video_input, config
) -> (
    tuple[list[PIL.Image.Image], dict[str, object]]
    | tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, object]]
):
    num_frames = config.num_video_frames

    def _unpack_video_item(video_item):
        frames_obj = video_item
        item_metadata = None

        for _ in range(4):
            if isinstance(frames_obj, np.ndarray) and frames_obj.ndim == 0:
                frames_obj = frames_obj.item()
                continue

            if (
                isinstance(frames_obj, (tuple, list))
                and len(frames_obj) == 2
                and _looks_like_video_metadata(frames_obj[1])
            ):
                item_metadata = frames_obj[1]
                frames_obj = frames_obj[0]
                continue

            break

        return frames_obj, item_metadata

    def _resolve_video_source(
        video_item,
        video_metadata,
    ) -> str | None:
        if isinstance(video_item, str):
            return video_item

        metadata_candidates = []
        if video_metadata is not None:
            metadata_candidates.append(video_metadata)
        _, packed_metadata = _unpack_video_item(video_item)
        if packed_metadata is not None:
            metadata_candidates.append(packed_metadata)

        for metadata_obj in metadata_candidates:
            if isinstance(metadata_obj, dict):
                video_path = metadata_obj.get("video_path")
                video_url = metadata_obj.get("video_url")
            else:
                video_path = getattr(metadata_obj, "video_path", None)
                video_url = getattr(metadata_obj, "video_url", None)

            if isinstance(video_path, str) and video_path:
                return video_path

            if isinstance(video_url, str) and video_url:
                if video_url.startswith("file://"):
                    from urllib.parse import urlparse
                    from urllib.request import url2pathname

                    parsed = urlparse(video_url)
                    return url2pathname((parsed.netloc or "") + (parsed.path or ""))
                return video_url

        return None

    def _meta_get(meta, key, default=None):
        if isinstance(meta, dict):
            return meta.get(key, default)
        return getattr(meta, key, default)

    unpacked_frames, unpacked_metadata = _unpack_video_item(video_input)
    if isinstance(unpacked_frames, str):
        frames_array, metadata = load_video(unpacked_frames, num_frames=num_frames)
    else:
        frames_array = unpacked_frames
        metadata = unpacked_metadata

    if frames_array is None:
        raise TypeError(
            "Unsupported video payload for AudioVisualFlamingo video extraction: "
            f"video_input_type={type(video_input)!r}, "
            f"unpacked_type={type(unpacked_frames)!r}, "
            f"unpacked_metadata_type={type(unpacked_metadata)!r}, "
            f"unpacked_repr={repr(unpacked_frames)[:200]}"
        )
    output_frames = _coerce_video_frames_to_pil(frames_array)

    fps = float(_meta_get(metadata, "fps", None) or 1.0)
    sampled_frame_indices = _meta_get(metadata, "frames_indices", None) if metadata is not None else None
    if sampled_frame_indices is None:
        frame_indices = list(range(len(output_frames)))
    else:
        frame_indices = list(np.asarray(sampled_frame_indices).tolist())

    metadata_total_frames = _meta_get(metadata, "total_num_frames", None) if metadata is not None else None
    frame_count = int(frame_indices[-1] + 1) if frame_indices else int(metadata_total_frames or len(output_frames))
    video_duration = _meta_get(metadata, "duration", None) if metadata is not None else None
    if video_duration is None:
        video_duration = float(frame_count / fps if fps > 0 else len(output_frames))
    else:
        video_duration = float(video_duration)
    # Keep np.float64 timestamps for parity with legacy timing dtype used by the original AudioVisualFlamingo path.
    output_frame_times = list(np.asarray(frame_indices, dtype=np.float64) / np.float64(fps if fps > 0 else 1.0))

    video_source = _resolve_video_source(video_input, metadata)

    aud_feature = None
    audio_info = None
    if config.load_audio_in_video and video_source is not None:
        try:
            aud_feature, audio_info = _load_audio_hf_with_info(video_source, config)
        except Exception:
            aud_feature, audio_info = None, None

    video_info = {
        "video_path": video_source,
        "has_audio": aud_feature is not None,
        "video_duration": video_duration,
        "audio_info": audio_info,
        "video_frame_times": output_frame_times,
    }
    if audio_info is not None and video_source is not None:
        audio_info["video_path"] = video_source

    if config.load_audio_in_video and config.interleaved_vis_aud_in_video and aud_feature is not None:
        segment_duration = config.interleaved_video_segment_duration
        if segment_duration == -1:
            raise ValueError("video_segment_duration is not set")

        segment_vis_indices_list = []
        segment_aud_indices_list = []
        segment_counts = int(np.ceil(video_duration / segment_duration))

        audio_start_sec = audio_info["audio_start_sec"]
        audio_end_sec = audio_info["audio_end_sample_sec"]
        stft_frames_per_second = config.audio_sampling_rate // config.audio_hop_length

        idx = 0
        aud_sample_start_idx = 0
        for i in range(segment_counts):
            end_frame = min((i + 1) * segment_duration * fps, frame_count)

            segment_indices = []
            while idx < len(frame_indices) and frame_indices[idx] < end_frame:
                segment_indices.append(frame_indices[idx])
                idx += 1
            segment_vis_indices_list.append(segment_indices)

            clip_start_sec = i * segment_duration
            clip_end_sec = min(clip_start_sec + segment_duration, video_duration)
            overlap_start = max(clip_start_sec, audio_start_sec)
            overlap_end = min(clip_end_sec, audio_end_sec)
            if overlap_start < overlap_end:
                aud_sample_end_idx = round((overlap_end - audio_start_sec) * stft_frames_per_second)
                segment_aud_indices_list.append([aud_sample_start_idx, aud_sample_end_idx])
                aud_sample_start_idx = aud_sample_end_idx
            else:
                segment_aud_indices_list.append([])

        new_segment_vis_indices_list = []
        processed_frame_index = 0
        for segment_indices in segment_vis_indices_list:
            new_segment_vis_indices_list.append([])
            for _ in segment_indices:
                new_segment_vis_indices_list[-1].append(processed_frame_index)
                processed_frame_index += 1

        video_info.update(
            {
                "segment_vis_indices_list": new_segment_vis_indices_list,
                "segment_aud_indices_list": segment_aud_indices_list,
                "expected_frame_count": len(frame_indices),
            }
        )

    if config.load_audio_in_video:
        return output_frames, aud_feature, video_info
    return output_frames, video_info


class AudioVisualFlamingoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        },
    }


class AudioVisualFlamingoProcessor(ProcessorMixin):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    valid_kwargs = [
        "padding_side",
        "image_aspect_ratio",
        "s2_scales",
        "max_tiles",
        "num_video_frames",
        "load_audio_in_video",
        "interleaved_vis_aud_in_video",
        "interleaved_video_segment_duration",
        "mm_use_bos_eos_tokens",
        "audio_sampling_rate",
        "audio_chunk_length",
        "audio_hop_length",
    ]

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        padding_side="left",
        image_aspect_ratio=None,
        s2_scales=None,
        max_tiles=12,
        num_video_frames=None,
        load_audio_in_video=True,
        interleaved_vis_aud_in_video=True,
        interleaved_video_segment_duration=30,
        mm_use_bos_eos_tokens=False,
        audio_sampling_rate=16000,
        audio_chunk_length=120,
        audio_hop_length=60,
        **kwargs,
    ):
        if chat_template is None:
            chat_template = _AUDIOVISUALFLAMINGO_CHAT_TEMPLATE
        self.image_token = MEDIA_TOKENS["image"]
        self.video_token = MEDIA_TOKENS["video"]
        self.sound_token = MEDIA_TOKENS["sound"]
        self.image_aspect_ratio = image_aspect_ratio
        self.s2_scales = s2_scales
        self.max_tiles = max_tiles
        self.num_video_frames = num_video_frames
        self.load_audio_in_video = load_audio_in_video
        self.interleaved_vis_aud_in_video = interleaved_vis_aud_in_video
        self.interleaved_video_segment_duration = interleaved_video_segment_duration
        self.mm_use_bos_eos_tokens = mm_use_bos_eos_tokens
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_chunk_length = audio_chunk_length
        self.audio_hop_length = audio_hop_length
        self.image_processor = image_processor
        if feature_extractor is None:
            chunk_length = audio_chunk_length if isinstance(audio_chunk_length, int) else 30
            feature_extractor = WhisperFeatureExtractor(
                feature_size=128,
                chunk_length=chunk_length,
                sampling_rate=audio_sampling_rate,
                hop_length=audio_hop_length,
            )
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        if tokenizer is not None:
            self.tokenizer.padding_side = padding_side
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
            self.sound_token_id = self.tokenizer.convert_tokens_to_ids(self.sound_token)
            self.pad_token_id = self.tokenizer("<|endoftext|>").input_ids[0]
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.image_token_id = 0
            self.video_token_id = 0
            self.sound_token_id = 0
            self.pad_token_id = 0
            self.eos_token_id = 0
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        **kwargs: Unpack[AudioVisualFlamingoProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("`text` is required.")
        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and (len(text) == 0 or isinstance(text[0], str))):
            raise ValueError("`text` must be a string or a list/tuple of strings.")
        else:
            text = list(text)

        processor_kwargs = {name: kwargs.pop(name) for name in self.valid_kwargs if name in kwargs}
        output_kwargs = self._merge_kwargs(
            AudioVisualFlamingoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if self.tokenizer is not None else None,
            **kwargs,
        )
        runtime_config = self._get_runtime_config(output_kwargs, **processor_kwargs)
        return self._call_native(
            text=text,
            images=images,
            videos=videos,
            audio=audio,
            runtime_config=runtime_config,
            text_kwargs=output_kwargs["text_kwargs"],
        )

    def _get_runtime_config(self, output_kwargs: dict[str, dict], **overrides) -> SimpleNamespace:
        runtime_kwargs = {
            "audio_chunk_length": self.audio_chunk_length,
            "audio_hop_length": self.audio_hop_length,
            "audio_sampling_rate": self.audio_sampling_rate,
            "feature_extractor": self.feature_extractor,
            "image_aspect_ratio": self.image_aspect_ratio,
            "image_processor": self.image_processor,
            "interleaved_video_segment_duration": self.interleaved_video_segment_duration,
            "interleaved_vis_aud_in_video": self.interleaved_vis_aud_in_video,
            "load_audio_in_video": self.load_audio_in_video,
            "max_tiles": self.max_tiles,
            "mm_use_bos_eos_tokens": self.mm_use_bos_eos_tokens,
            "num_video_frames": self.num_video_frames,
            "padding_side": self.padding_side,
            "random_audio_sample": getattr(self, "random_audio_sample", False),
            "s2_scales": self.s2_scales,
        }
        runtime_kwargs.update(
            {
                "audio_chunk_length": output_kwargs["audio_kwargs"].get(
                    "chunk_length", runtime_kwargs["audio_chunk_length"]
                ),
                "audio_hop_length": output_kwargs["audio_kwargs"].get(
                    "hop_length", runtime_kwargs["audio_hop_length"]
                ),
                "audio_sampling_rate": output_kwargs["audio_kwargs"].get(
                    "sampling_rate", runtime_kwargs["audio_sampling_rate"]
                ),
                "num_video_frames": output_kwargs["videos_kwargs"].get(
                    "num_frames", runtime_kwargs["num_video_frames"]
                ),
                "padding_side": output_kwargs["text_kwargs"].get("padding_side", runtime_kwargs["padding_side"]),
            }
        )
        runtime_kwargs.update(overrides)
        if isinstance(runtime_kwargs["s2_scales"], str):
            runtime_kwargs["s2_scales"] = [int(scale) for scale in runtime_kwargs["s2_scales"].split(",")]
        return SimpleNamespace(**runtime_kwargs)

    def _normalize_nested_media(self, values, batch_size: int) -> list[list]:
        if values is None:
            return [[] for _ in range(batch_size)]

        if batch_size == 1 and _is_packed_media_item(values):
            return [[values]]

        if batch_size == 1 and (
            not isinstance(values, (list, tuple)) or (values and not isinstance(values[0], (list, tuple)))
        ):
            if isinstance(values, (list, tuple)):
                return [list(values)]
            return [[values]]

        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            raise ValueError(f"Expected batched media list with length {batch_size}, got {type(values)}")

        normalized = []
        for item in values:
            if item is None:
                normalized.append([])
            elif _is_packed_media_item(item):
                normalized.append([item])
            elif isinstance(item, (list, tuple)):
                normalized.append(list(item))
            else:
                normalized.append([item])
        return normalized

    def _normalize_audio_sample(self, sample_audio) -> list:
        if sample_audio is None:
            return []
        if _is_audio_like(sample_audio):
            return [sample_audio]
        if isinstance(sample_audio, (list, tuple)):
            if not sample_audio:
                return []
            if all(_is_audio_like(item) for item in sample_audio):
                if all(isinstance(item, (np.ndarray, torch.Tensor)) for item in sample_audio):
                    return list(make_list_of_audio(list(sample_audio)))
                return list(sample_audio)
        raise ValueError(f"Unsupported audio sample type: {type(sample_audio)!r}")

    def _normalize_audio_batches(self, audio, prompts: list[str]) -> list[list]:
        batch_size = len(prompts)
        if audio is None:
            return [[] for _ in range(batch_size)]

        if batch_size == 1:
            return [self._normalize_audio_sample(audio)]

        if (
            isinstance(audio, (list, tuple))
            and len(audio) == batch_size
            and all(
                item is None
                or _is_audio_like(item)
                or (isinstance(item, (list, tuple)) and all(_is_audio_like(sub_item) for sub_item in item))
                for item in audio
            )
        ):
            return [self._normalize_audio_sample(sample_audio) for sample_audio in audio]

        flat_audio = self._normalize_audio_sample(audio)
        audio_counts = [prompt.count(self.sound_token) for prompt in prompts]
        if sum(audio_counts) != len(flat_audio):
            raise ValueError(
                "Batched audio inputs must either be grouped per sample or match the number of `<sound>` tokens in "
                f"the prompts. Got {len(flat_audio)} audio inputs for token counts {audio_counts}."
            )

        audio_batches = []
        cursor = 0
        for audio_count in audio_counts:
            audio_batches.append(flat_audio[cursor : cursor + audio_count])
            cursor += audio_count
        return audio_batches

    def _prepare_sample(
        self,
        text: str,
        runtime_config: SimpleNamespace,
        images: list | None = None,
        videos: list | None = None,
        audio: list | None = None,
    ) -> tuple[str, defaultdict, defaultdict]:
        media = defaultdict(list)
        media_config = defaultdict(dict)
        raw_sounds = []
        video_infos = []

        if images:
            if len(images) == 1 and runtime_config.image_aspect_ratio == "dynamic_s2":
                image_tensor, block_sizes = _process_image(images[0], runtime_config, enable_dynamic_s2=True)
                media["image"] = list(image_tensor.half())
                media_config["image"]["block_sizes"] = [block_sizes]
            else:
                media["image"] = list(_process_images(images, runtime_config.image_processor, runtime_config).half())

        audio_info_list = []
        if videos:
            for video in videos:
                if runtime_config.load_audio_in_video:
                    frames, audio_waveform, video_info = _extract_video_hf(video, runtime_config)
                    if audio_waveform is not None:
                        raw_sounds.append(audio_waveform)
                        audio_info_list.append(video_info["audio_info"])
                else:
                    frames, video_info = _extract_video_hf(video, runtime_config)
                media["video"].append(_process_images(frames, runtime_config.image_processor, runtime_config).half())
                video_infos.append(video_info)
            media["video_info"] = [video_infos]

        explicit_audio_count = len(audio) if audio else 0
        if audio:
            for audio_item in audio:
                audio_waveform, audio_info = _load_audio_hf_with_info(audio_item, runtime_config)
                raw_sounds.append(audio_waveform)
                audio_info_list.append(audio_info)

        if raw_sounds:
            media["sound"] = _extract_sound_features(
                raw_sounds,
                audio_info_list,
                runtime_config,
                feature_extractor=runtime_config.feature_extractor,
            )

        if audio_info_list:
            media["audio_info"] = [audio_info_list]

        if video_infos and runtime_config.load_audio_in_video:
            expected_sound_tokens = explicit_audio_count + sum(
                1 for video_info in video_infos if video_info.get("has_audio", False)
            )
            missing_sound_tokens = expected_sound_tokens - text.count(self.sound_token)
            if missing_sound_tokens > 0:
                rebuilt = []
                cursor = 0
                for video_info in video_infos:
                    pos = text.find(self.video_token, cursor)
                    if pos < 0:
                        break
                    rebuilt.append(text[cursor:pos])
                    if video_info.get("has_audio", False) and missing_sound_tokens > 0:
                        rebuilt.append(self.sound_token)
                        missing_sound_tokens -= 1
                    rebuilt.append(self.video_token)
                    cursor = pos + len(self.video_token)
                rebuilt.append(text[cursor:])
                text = "".join(rebuilt)

        if runtime_config.mm_use_bos_eos_tokens:
            text = _add_mm_bos_eos_tokens(text)

        return text, media, media_config

    def _call_native(
        self,
        text: list[str],
        runtime_config: SimpleNamespace,
        text_kwargs: dict,
        images=None,
        videos=None,
        audio=None,
    ) -> BatchFeature:
        if not text:
            raise ValueError("`text` must contain at least one prompt.")

        image_batches = self._normalize_nested_media(images, len(text))
        video_batches = self._normalize_nested_media(videos, len(text))
        audio_batches = self._normalize_audio_batches(audio, text)

        processed_text = []
        media = defaultdict(list)
        media_config = defaultdict(dict)

        for prompt, sample_images, sample_videos, sample_audio in zip(
            text, image_batches, video_batches, audio_batches
        ):
            sample_text, sample_media, sample_media_config = self._prepare_sample(
                prompt,
                runtime_config=runtime_config,
                images=sample_images,
                videos=sample_videos,
                audio=sample_audio,
            )
            processed_text.append(sample_text)
            for name in sample_media:
                media[name].extend(sample_media[name])
            _merge_media_config(media_config, sample_media_config)

        text_inputs = self.tokenizer(processed_text, **text_kwargs)
        if "attention_mask" in text_inputs and isinstance(text_inputs["attention_mask"], torch.Tensor):
            text_inputs["attention_mask"] = text_inputs["attention_mask"].to(dtype=torch.bool)
        self._check_special_mm_tokens(processed_text, text_inputs, modalities=["image", "video", "sound"])

        return BatchFeature(
            data={
                **text_inputs,
                "media": media,
                "media_config": media_config,
            }
        )

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = (
            self.feature_extractor.model_input_names if self.feature_extractor is not None else []
        )
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + image_processor_input_names
                + feature_extractor_input_names
                + ["media", "media_config"]
            )
        )


__all__ = [
    "AudioVisualFlamingoProcessor",
    "AudioVisualFlamingoProcessorKwargs",
]
