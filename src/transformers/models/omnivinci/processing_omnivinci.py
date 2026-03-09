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

import json
import os
import random
from collections import defaultdict

import numpy as np
import PIL.Image
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import WhisperFeatureExtractor
from transformers.audio_utils import load_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import load_image
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.video_utils import load_video

from .configuration_omnivinci import MEDIA_TOKENS, MM_BOS_EOS_TOKENS, OmniVinciConfig


_OMNIVINCI_CHAT_TEMPLATE = (
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


def _collect_encoder_boundary_tokens(config) -> list[str]:
    token_keys = {"start_tokens", "end_tokens", "sep_tokens"}
    collected = []
    seen = set()

    def _maybe_add(token):
        if not isinstance(token, str) or token == "None" or token in seen:
            return
        seen.add(token)
        collected.append(token)

    def _visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_keys:
                    _maybe_add(value)
                _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    # Encoder implementations default `end_tokens` to "\n" when the config omits it.
    _maybe_add("\n")

    for attr in ("image_encoder", "video_encoder", "sound_encoder"):
        encoder_config = getattr(config, attr, None)
        if isinstance(encoder_config, str):
            try:
                encoder_config = json.loads(encoder_config)
            except Exception:
                continue
        _visit(encoder_config)

    return collected


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


def _process_image(image_file, data_args, image_folder, enable_dynamic_s2=False):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = load_image(os.path.join(image_folder, image_file))
        else:
            image = load_image(image_file)
    else:
        image = image_file
    image = image.convert("RGB")
    crop_size = getattr(data_args.image_processor, "crop_size", None)
    if crop_size is None:
        crop_size = getattr(data_args.image_processor, "size", None)
    if crop_size is None:
        raise ValueError("OmniVinci image processor must define either `crop_size` or `size`.")
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
    new_images = [_process_image(image, model_cfg, None) for image in images]

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


def _pad_fn(input_ids_list: list[torch.Tensor], padding_value=0, target_len=None, padding_side="left") -> torch.Tensor:
    if not input_ids_list:
        raise ValueError("input_ids_list must not be empty")

    sequences = [ids.squeeze(0) for ids in input_ids_list]

    if padding_side == "right":
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    elif padding_side == "left":
        reversed_sequences = [torch.flip(ids, dims=[0]) for ids in sequences]
        padded = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
        padded = torch.flip(padded, dims=[1])
    else:
        raise ValueError(f"Unsupported padding_side: {padding_side}")

    if target_len is not None:
        assert target_len >= padded.shape[1], "target_len must be greater than or equal to max_len"
        if target_len > padded.shape[1]:
            pad_width = target_len - padded.shape[1]
            pad_tensor = padded.new_full((padded.shape[0], pad_width), padding_value)
            if padding_side == "right":
                padded = torch.cat((padded, pad_tensor), dim=1)
            else:
                padded = torch.cat((pad_tensor, padded), dim=1)

    return padded


def _pad_or_trim_audio(audio: np.ndarray, length: int) -> np.ndarray:
    current_length = int(audio.shape[0])
    if current_length > length:
        return audio[:length]
    if current_length < length:
        return np.pad(audio, (0, length - current_length), mode="constant")
    return audio


def _resolve_sound_feature_size(config) -> int:
    sound_tower_cfg = getattr(config, "sound_tower_cfg", None)
    if isinstance(sound_tower_cfg, dict):
        feature_size = sound_tower_cfg.get("num_mel_bins")
    else:
        feature_size = getattr(sound_tower_cfg, "num_mel_bins", None)
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

    if isinstance(audio_input, np.ndarray):
        speech_data = audio_input.astype(np.float32, copy=False)
        ori_n_samples = int(speech_data.shape[0])
        audio_start_sample_id, audio_end_sample_id = _resolve_window(ori_n_samples)
        ori_audio_duration = ori_n_samples / sampling_rate
        speech_data = speech_data[audio_start_sample_id:audio_end_sample_id]
    elif isinstance(audio_input, str) and audio_input.lower().endswith(".mp4"):
        from decord import AudioReader, cpu

        audio_reader = AudioReader(audio_input, ctx=cpu(0), sample_rate=sampling_rate, mono=True)
        ori_n_samples = int(audio_reader.shape[1])
        audio_start_sample_id, audio_end_sample_id = _resolve_window(ori_n_samples)
        ori_audio_duration = ori_n_samples / sampling_rate
        speech_data = (
            audio_reader[audio_start_sample_id:audio_end_sample_id].asnumpy()[0].astype(np.float32, copy=False)
        )
    else:
        speech_data = load_audio(audio_input, sampling_rate=sampling_rate).astype(np.float32, copy=False)
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


def _extract_video_hf(
    video_input, config
) -> (
    tuple[list[PIL.Image.Image], dict[str, object]]
    | tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, object]]
):
    num_frames = config.num_video_frames

    def _looks_like_video_metadata(meta) -> bool:
        if meta is None:
            return False
        if isinstance(meta, dict):
            return bool({"fps", "frames_indices", "total_num_frames", "video_path", "video_url"} & set(meta.keys()))
        return any(
            hasattr(meta, key) for key in ("fps", "frames_indices", "total_num_frames", "video_path", "video_url")
        )

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

    def _make_legacy_uniform_indices(video_source_for_sampling):
        def _legacy_uniform_indices(metadata, **kwargs):
            total_num_frames = int(getattr(metadata, "total_num_frames", 0) or 0)
            if total_num_frames <= 0:
                return np.array([], dtype=int)

            # Match legacy OmniVinci sampling by locating the last readable frame first.
            last_valid_frame_count = total_num_frames
            if isinstance(video_source_for_sampling, str):
                import cv2

                video_capture = cv2.VideoCapture(video_source_for_sampling)
                try:
                    while last_valid_frame_count > 0:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, last_valid_frame_count - 1)
                        if video_capture.grab():
                            break
                        last_valid_frame_count -= 1
                finally:
                    video_capture.release()

            if last_valid_frame_count <= 0:
                return np.array([], dtype=int)
            return np.round(np.linspace(0, last_valid_frame_count - 1, num_frames)).astype(int)

        return _legacy_uniform_indices

    unpacked_frames, unpacked_metadata = _unpack_video_item(video_input)
    unpacked_source = _resolve_video_source(video_input, unpacked_metadata)
    if unpacked_metadata is not None:
        # Re-run OmniVinci's native frame sampling path when source is available.
        # This keeps parity with string-path inputs and avoids downstream drift when
        # upstream loaders return fewer frames due terminal-frame decode failures.
        if isinstance(unpacked_source, str) and unpacked_source:
            try:
                frames_array, metadata = load_video(
                    unpacked_source,
                    backend="opencv",
                    sample_indices_fn=_make_legacy_uniform_indices(unpacked_source),
                )
                if isinstance(metadata, list):
                    metadata = None
            except Exception:
                frames_array = np.asarray(unpacked_frames)
                metadata = unpacked_metadata
        else:
            frames_array = np.asarray(unpacked_frames)
            metadata = unpacked_metadata
    else:
        frames_array, metadata = load_video(
            video_input,
            backend="opencv",
            sample_indices_fn=_make_legacy_uniform_indices(video_input if isinstance(video_input, str) else None),
        )
        if isinstance(metadata, list):
            metadata = None

    frames_array = np.asarray(frames_array)
    if frames_array.ndim == 0:
        raise TypeError(
            "Unsupported video payload for OmniVinci video extraction: "
            f"video_input_type={type(video_input)!r}, "
            f"unpacked_type={type(unpacked_frames)!r}, "
            f"unpacked_metadata_type={type(unpacked_metadata)!r}, "
            f"unpacked_repr={repr(unpacked_frames)[:200]}"
        )
    output_frames = [PIL.Image.fromarray(frame).convert("RGB") for frame in frames_array]

    fps = float(_meta_get(metadata, "fps", None) or 1.0)
    sampled_frame_indices = _meta_get(metadata, "frames_indices", None) if metadata is not None else None
    if sampled_frame_indices is None:
        frame_indices = list(range(len(output_frames)))
    else:
        frame_indices = list(np.asarray(sampled_frame_indices).tolist())

    metadata_total_frames = _meta_get(metadata, "total_num_frames", None) if metadata is not None else None
    frame_count = int(frame_indices[-1] + 1) if frame_indices else int(metadata_total_frames or len(output_frames))
    video_duration = float(frame_count / fps if fps > 0 else len(output_frames))
    # Keep np.float64 timestamps for parity with legacy timing dtype used by the original OmniVinci path.
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


class OmniVinciProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class OmniVinciProcessor(ProcessorMixin):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = []

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        config=None,
        padding_side="left",
        **kwargs,
    ):
        if isinstance(config, dict):
            config = OmniVinciConfig(**config)
        if chat_template is None:
            chat_template = _OMNIVINCI_CHAT_TEMPLATE
        self.image_token = MEDIA_TOKENS["image"]
        self.video_token = MEDIA_TOKENS["video"]
        self.sound_token = MEDIA_TOKENS["sound"]
        self.config = config
        self.image_processor = image_processor
        if feature_extractor is None:
            default_chunk_length = getattr(config, "audio_chunk_length", 30) if config is not None else 30
            if not isinstance(default_chunk_length, int):
                default_chunk_length = 30
            feature_extractor = WhisperFeatureExtractor(
                feature_size=_resolve_sound_feature_size(config) if config is not None else 80,
                chunk_length=default_chunk_length,
                sampling_rate=getattr(config, "audio_sampling_rate", 16000) if config is not None else 16000,
                hop_length=getattr(config, "audio_hop_length", 160) if config is not None else 160,
            )
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.tokenizer.padding_side = padding_side

        # Use <|endoftext|> token as padding token for Qwen models
        self.pad_token_id = self.tokenizer("<|endoftext|>").input_ids[0]
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.config is not None:
            self.config.padding_side = self.padding_side
            self.config.pad_token_id = self.pad_token_id
            self.config.eos_token_id = self.eos_token_id
            if getattr(self.config, "bos_token_id", None) is None:
                self.config.bos_token_id = self.tokenizer.bos_token_id
            if getattr(self.config, "model_max_length", None) is None:
                self.config.model_max_length = getattr(self.tokenizer, "model_max_length", 2048)

            media_token_ids = {}
            for name, token in self.config.media_tokens.items():
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is None or token_id < 0:
                    tokenized = self.tokenizer(token, add_special_tokens=False).input_ids
                    if len(tokenized) != 1:
                        raise ValueError(f"Media token `{token}` must map to a single tokenizer id.")
                    token_id = tokenized[0]
                media_token_ids[name] = int(token_id)
            self.config.media_token_ids = media_token_ids

            self.config.encoder_text_token_ids = {
                token_text: [int(token_id) for token_id in self.tokenizer(token_text).input_ids]
                for token_text in _collect_encoder_boundary_tokens(self.config)
            }

        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def __repr__(self):
        return (
            f"OmniVinciProcessor(image_processor=SigLip, feature_extractor={self.feature_extractor}, "
            f"tokenizer={self.tokenizer}, config={self.config})"
        )

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        **kwargs: Unpack[OmniVinciProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("`text` is required.")
        if not isinstance(text, str) and not (
            isinstance(text, (list, tuple)) and (len(text) == 0 or isinstance(text[0], str))
        ):
            raise ValueError("`text` must be a string or a list/tuple of strings.")
        return self._call_native(text=text, images=images, videos=videos, audio=audio, **kwargs)

    def _normalize_nested_media(self, values, batch_size: int) -> list[list]:
        def _is_packed_media_item(item) -> bool:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return False
            meta = item[1]
            if isinstance(meta, dict):
                return bool(
                    {"fps", "frames_indices", "total_num_frames", "video_path", "video_url"} & set(meta.keys())
                )
            return any(
                hasattr(meta, key) for key in ("fps", "frames_indices", "total_num_frames", "video_path", "video_url")
            )

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

    def _single_native_call(
        self,
        text: str,
        images: list | None = None,
        videos: list | None = None,
        audio: list | None = None,
    ) -> BatchFeature:
        media = defaultdict(list)
        media_config = defaultdict(dict)
        raw_sounds = []
        video_infos = []

        if images:
            if len(images) == 1 and self.config.image_aspect_ratio == "dynamic_s2":
                self.config.image_processor = self.image_processor
                if isinstance(self.config.s2_scales, str):
                    self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                image_tensor, block_sizes = _process_image(images[0], self.config, None, enable_dynamic_s2=True)
                media["image"] = list(image_tensor.half())
                media_config["image"]["block_sizes"] = [block_sizes]
            else:
                media["image"] = list(_process_images(images, self.image_processor, self.config).half())

        audio_info_list = []
        if videos:
            for video in videos:
                if self.config.load_audio_in_video:
                    frames, audio_waveform, video_info = _extract_video_hf(video, self.config)
                    if audio_waveform is not None:
                        raw_sounds.append(audio_waveform)
                        audio_info_list.append(video_info["audio_info"])
                else:
                    frames, video_info = _extract_video_hf(video, self.config)
                media["video"].append(_process_images(frames, self.image_processor, self.config).half())
                video_infos.append(video_info)
            media["video_info"] = [video_infos]

        explicit_audio_count = len(audio) if audio else 0
        if audio:
            for audio_item in audio:
                audio_waveform, audio_info = _load_audio_hf_with_info(audio_item, self.config)
                raw_sounds.append(audio_waveform)
                audio_info_list.append(audio_info)

        if raw_sounds:
            media["sound"] = _extract_sound_features(
                raw_sounds, audio_info_list, self.config, feature_extractor=self.feature_extractor
            )

        if audio_info_list:
            media["audio_info"] = [audio_info_list]

        if video_infos and self.config.load_audio_in_video:
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

        if getattr(self.config, "mm_use_bos_eos_tokens", False):
            text = _add_mm_bos_eos_tokens(text)

        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask.to(dtype=torch.bool)

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )

    def _call_native(self, text, images=None, videos=None, audio=None, **kwargs) -> BatchFeature:
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            raise ValueError("`text` must contain at least one prompt.")

        image_batches = self._normalize_nested_media(images, len(texts))
        video_batches = self._normalize_nested_media(videos, len(texts))

        if audio is None:
            audio_batches = [[] for _ in range(len(texts))]
        elif len(texts) == 1:
            audio_batches = [[audio]] if not isinstance(audio, (list, tuple)) else [list(audio)]
        else:
            raise ValueError(
                "Batched `audio` with native `apply_chat_template(tokenize=True)` is not supported in OmniVinciProcessor yet."
            )

        padding_side = kwargs.get("padding_side", self.padding_side)
        input_ids_list = []
        media = defaultdict(list)
        media_config = defaultdict(dict)

        for prompt, sample_images, sample_videos, sample_audio in zip(
            texts, image_batches, video_batches, audio_batches
        ):
            feat = self._single_native_call(prompt, images=sample_images, videos=sample_videos, audio=sample_audio)
            input_ids_list.append(feat.input_ids)
            for name in feat.media:
                media[name] += feat.media[name]
            for name in feat.media_config:
                media_config[name].update(feat.media_config[name])

        input_ids = _pad_fn(input_ids_list, padding_value=self.pad_token_id, padding_side=padding_side)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[input_ids == self.pad_token_id] = False

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )

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
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + feature_extractor_input_names))


__all__ = [
    "OmniVinciProcessor",
    "OmniVinciProcessorKwargs",
]
