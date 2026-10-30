# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for MiMo V2 multimodal models."""

import json
import math
import re
from itertools import groupby

import numpy as np

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import TextInput
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import CONFIG_NAME, auto_docstring, cached_file
from ...video_utils import VideoInput, VideoMetadata
from .feature_extraction_mimo_v2 import MiMoV2FeatureExtractor
from .image_processing_mimo_v2_vl import MiMoV2ImageProcessor
from .video_processing_mimo_v2_vl import MiMoV2VideoProcessor


class MiMoV2AudioKwargs(AudioKwargs, total=False):
    """
    is_tokenized (`bool`, *optional*, defaults to `False`):
        Interpret audio inputs as integer codec arrays instead of waveforms.
    """

    is_tokenized: bool


class MiMoV2VideosKwargs(VideosKwargs, total=False):
    """
    use_audio_in_video (`bool` or `list[bool]`, *optional*, defaults to `False`):
        Whether each video has an audio input. Audio inputs must be supplied in the
        order of standalone audio and audio-enabled video placeholders across the text batch.
    video_audio_interleave_length (`float` or `int`, *optional*):
        Override the processor's audio/video grouping duration in seconds.
    use_per_grid_t_timestamps (`bool`, *optional*):
        Override whether timestamps are repeated for every video/audio grid.
    """

    use_audio_in_video: bool | list[bool]
    video_audio_interleave_length: float | int
    use_per_grid_t_timestamps: bool


class MiMoV2ProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: MiMoV2AudioKwargs
    videos_kwargs: MiMoV2VideosKwargs
    _defaults = {}


@auto_docstring(
    custom_intro="Constructs a multimodal processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2Processor(ProcessorMixin):
    valid_processor_kwargs = MiMoV2ProcessorKwargs
    _special_token_names = (
        "image",
        "video",
        "audio",
        "vision_start",
        "vision_end",
        "audio_start",
        "audio_end",
        "video_start",
        "video_end",
    )

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        audio_token="<|audio_pad|>",
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
        audio_start_token="<|mimo_audio_start|>",
        audio_end_token="<|mimo_audio_end|>",
        video_start_token="<|mimo_video_start|>",
        video_end_token="<|mimo_video_end|>",
        video_audio_interleave_length=0,
        use_per_grid_t_timestamps=False,
    ):
        r"""
        image_processor ([`MiMoV2ImageProcessor`], *optional*):
            Image processor. Defaults to a new `MiMoV2ImageProcessor` when unset.
        video_processor ([`MiMoV2VideoProcessor`], *optional*):
            Video processor. Defaults to a new `MiMoV2VideoProcessor` when unset.
        feature_extractor ([`MiMoV2FeatureExtractor`], *optional*):
            Audio feature extractor. Defaults to a new `MiMoV2FeatureExtractor` when unset.
        chat_template (`str`, *optional*):
            Jinja template for chat messages. Defaults to the tokenizer's chat template when available.
        image_token (`str`, *optional*, defaults to `"<|image_pad|>"`):
            Image placeholder token.
        video_token (`str`, *optional*, defaults to `"<|video_pad|>"`):
            Video placeholder token.
        audio_token (`str`, *optional*, defaults to `"<|audio_pad|>"`):
            Audio placeholder token.
        vision_start_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            Start of an image or video-frame feature span.
        vision_end_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            End of an image or video-frame feature span.
        audio_start_token (`str`, *optional*, defaults to `"<|mimo_audio_start|>"`):
            Start of an audio feature span.
        audio_end_token (`str`, *optional*, defaults to `"<|mimo_audio_end|>"`):
            End of an audio feature span.
        video_start_token (`str`, *optional*, defaults to `"<|mimo_video_start|>"`):
            Start of the complete timestamped video sequence.
        video_end_token (`str`, *optional*, defaults to `"<|mimo_video_end|>"`):
            End of the complete timestamped video sequence.
        video_audio_interleave_length (`float`, *optional*, defaults to 0):
            Group one video grid followed by audio when zero, all video followed by audio
            when -1, or group grids into intervals of this many seconds when positive.
        use_per_grid_t_timestamps (`bool`, *optional*, defaults to `False`):
            Emit a timestamp for every video and audio grid rather than once per group.
        """
        image_processor = image_processor if image_processor is not None else MiMoV2ImageProcessor()
        video_processor = video_processor if video_processor is not None else MiMoV2VideoProcessor()
        feature_extractor = feature_extractor if feature_extractor is not None else MiMoV2FeatureExtractor()
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.video_start_token = video_start_token
        self.video_end_token = video_end_token
        for name in self._special_token_names:
            token = getattr(self, f"{name}_token")
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                token_id = None
            setattr(self, f"{name}_token_id", token_id)
        self._validate_interleave_length(video_audio_interleave_length)
        self.video_audio_interleave_length = video_audio_interleave_length
        self.use_per_grid_t_timestamps = use_per_grid_t_timestamps

    @classmethod
    def get_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor settings, falling back to the checkpoint's model configuration for missing components."""
        processor_dict, instantiation_kwargs = super().get_processor_dict(pretrained_model_name_or_path, **kwargs)
        component_names = ("image_processor", "video_processor", "feature_extractor")
        if all(key in processor_dict for key in component_names):
            return processor_dict, instantiation_kwargs

        # Legacy MiMo V2 VL checkpoints keep their authoritative preprocessing settings in config.json.
        config_file = cached_file(
            pretrained_model_name_or_path,
            CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ("cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder")
            },
        )
        config = {}
        if config_file is not None:
            with open(config_file, encoding="utf-8") as reader:
                config = json.load(reader)
        checkpoint_processor_dict = cls._get_processor_config(config)
        for name, settings in checkpoint_processor_dict.items():
            if name in component_names:
                processor_dict[name] = {**settings, **processor_dict.get(name, {})}
            else:
                processor_dict.setdefault(name, settings)
        return processor_dict, instantiation_kwargs

    @classmethod
    def _get_processor_config(cls, config: dict) -> dict:
        """Translate model configuration fields into image, video, and audio component settings."""
        processor_config = config.get("processor_config") or {}
        vision_config = config.get("vision_config") or {}
        audio_config = config.get("audio_config") or {}
        spatial_config = {
            "patch_size": processor_config.get(
                "patch_size", vision_config.get("patch_size", MiMoV2ImageProcessor.patch_size)
            ),
            "merge_size": processor_config.get(
                "merge_size", vision_config.get("spatial_merge_size", MiMoV2ImageProcessor.merge_size)
            ),
            "temporal_patch_size": processor_config.get(
                "temporal_patch_size",
                vision_config.get("temporal_patch_size", MiMoV2ImageProcessor.temporal_patch_size),
            ),
        }
        factor = spatial_config["patch_size"] * spatial_config["merge_size"]
        processor_dict = {}
        for modality in ("image", "video"):
            processor_dict[f"{modality}_processor"] = {
                **spatial_config,
                "size": {
                    "shortest_edge": processor_config.get(f"{modality}_min_pixels", 4 * factor**2),
                    "longest_edge": processor_config.get(f"{modality}_max_pixels", 4096 * factor**2),
                },
            }
        processor_dict["video_processor"].update(
            temporal_compression_ratio=processor_config.get(
                "temporal_compression_ratio", MiMoV2VideoProcessor.temporal_compression_ratio
            ),
            total_max_pixels=processor_config.get("video_total_max_pixels", 16384 * factor**2),
            fps=processor_config.get("fps", MiMoV2VideoProcessor.fps),
            num_frames=processor_config.get("num_frames"),
            min_frames=processor_config.get("min_frames") or MiMoV2VideoProcessor.min_frames,
            max_frames=processor_config.get("max_frames") or MiMoV2VideoProcessor.max_frames,
        )
        audio_config_mapping = {
            "audio_sampling_rate": "sampling_rate",
            "audio_nfft": "n_fft",
            "audio_hop_length": "hop_length",
            "audio_window_size": "win_length",
            "audio_fmin": "f_min",
            "audio_fmax": "f_max",
            "audio_n_mels": "feature_size",
            "audio_channels": "audio_channels",
            "audio_group_size": "audio_group_size",
            "audio_segment_size": "audio_segment_size",
            "audio_kernel_size": "audio_kernel_size",
            "audio_stride_size": "audio_stride_size",
            "audio_avg_pooler": "audio_avg_pooler",
            "audio_input_id_per_second": "audio_input_id_per_second",
        }
        processor_dict["feature_extractor"] = {
            "audio_channels": audio_config.get("audio_channels", 8),
            "audio_group_size": audio_config.get("group_size", 4),
            "audio_segment_size": audio_config.get("audio_segment_size", 6000),
            **{
                target: processor_config[source]
                for source, target in audio_config_mapping.items()
                if source in processor_config
            },
        }
        for key in ("video_audio_interleave_length", "use_per_grid_t_timestamps"):
            if key in processor_config:
                processor_dict[key] = processor_config[key]
        token_config = {**config, **processor_config}
        processor_dict["_checkpoint_token_ids"] = {
            f"{name}_token_id": token_config[f"{name}_token_id"]
            for name in cls._special_token_names
            if token_config.get(f"{name}_token_id") is not None
        }
        return processor_dict

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        # Keep the tokenizer.json normalization and post-processing rules from XiaomiMiMo/MiMo-V2.6-Flash-RL.
        component_kwargs = {key: value for key, value in kwargs.items() if key != "return_unused_kwargs"}
        tokenizer = TokenizersBackend.from_pretrained(pretrained_model_name_or_path, **component_kwargs)
        for key, token_id in processor_dict.pop("_checkpoint_token_ids", {}).items():
            token_name = key.removesuffix("_id")
            if token_name in processor_dict:
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is None or tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Checkpoint {key}={token_id} is not a single token in the tokenizer.")
            processor_dict[token_name] = token
        return [
            MiMoV2ImageProcessor.from_dict(processor_dict["image_processor"], **component_kwargs),
            MiMoV2VideoProcessor.from_dict(processor_dict["video_processor"], **component_kwargs),
            MiMoV2FeatureExtractor.from_dict(processor_dict["feature_extractor"], **component_kwargs),
            tokenizer,
        ]

    def _validate_modality_tokens(self, modality: str) -> None:
        names = {
            "image": ("image", "vision_start", "vision_end"),
            "video": ("video", "vision_start", "vision_end", "video_start", "video_end"),
            "audio": ("audio", "audio_start", "audio_end"),
        }[modality]
        for name in names:
            token = getattr(self, f"{name}_token")
            token_id = getattr(self, f"{name}_token_id")
            if token_id is None or self.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Tokenizer is missing the MiMo V2 series {modality} special token {token!r}.")

    @staticmethod
    def _validate_interleave_length(length: float) -> None:
        if not math.isfinite(length) or (length < 0 and length != -1):
            raise ValueError("video_audio_interleave_length must be -1 or a finite nonnegative duration.")

    def _get_multimodal_token_pattern(self) -> re.Pattern:
        """Match bare placeholders and placeholders already framed by the checkpoint's chat template."""
        patterns = []
        for modality in ("image", "video", "audio"):
            prefix = "audio" if modality == "audio" else "vision"
            token = re.escape(getattr(self, f"{modality}_token"))
            start, end = (
                re.escape(getattr(self, f"{prefix}_start_token")),
                re.escape(getattr(self, f"{prefix}_end_token")),
            )
            patterns.append(f"(?P<{modality}>{start}(?:{token})+{end}|{token})")
        return re.compile("|".join(patterns))

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[MiMoV2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Args:
            audio (`AudioInput`, *optional*):
                Audio waveforms, or codec IDs when `is_tokenized=True`. Pass video audio tracks explicitly with
                `use_audio_in_video=True` or one boolean per video, in placeholder order across the text batch.
            videos (`VideoInput`, *optional*):
                Videos to preprocess. Decoded frame arrays require `video_metadata` containing the original frame
                rate and frame indices. Set `do_sample_frames=False` for already sampled frames.

        Returns:
            [`BatchFeature`]: Tokenized text and the supplied modalities' features. Waveforms produce `input_features`
            of shape `(batch, mel_time, mel_bins)`; codec inputs produce `input_audio_codes` of shape
            `(batch, grouped_time, group_size, codec_channels)`. `audio_feature_lens` counts audio tokens and
            `feature_attention_mask` marks valid feature frames. `audio_segment_offsets` contains
            `(audio_index, start, length)` rows in grouped-token units, ordered as the audio spans in the prompt.
        """
        output_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        video_kwargs = output_kwargs["videos_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        use_audio_in_video = video_kwargs.pop("use_audio_in_video", False)
        interleave_length = video_kwargs.pop("video_audio_interleave_length", self.video_audio_interleave_length)
        use_per_grid_t_timestamps = video_kwargs.pop("use_per_grid_t_timestamps", self.use_per_grid_t_timestamps)
        self._validate_interleave_length(interleave_length)
        is_tokenized_audio = audio_kwargs.get("is_tokenized", False)
        images, text, videos, fetched_audio = self.prepare_inputs_layout(
            images=images,
            text=text,
            videos=videos,
            audio=None if is_tokenized_audio else audio,
            sampling_rate=audio_kwargs.get("sampling_rate", self.feature_extractor.sampling_rate),
        )
        audio = audio if is_tokenized_audio else fetched_audio
        self.validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        for modality, inputs in (("image", images), ("video", videos), ("audio", audio)):
            if inputs is not None:
                self._validate_modality_tokens(modality)
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"]) if images is not None else {}
        video_kwargs["return_metadata"] = True
        video_inputs = self.video_processor(videos, **video_kwargs) if videos is not None else {}
        audio_inputs = self.feature_extractor(audio, **audio_kwargs) if audio is not None else {}
        if "attention_mask" in audio_inputs:
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        num_videos = len(video_inputs.get("video_grid_thw", []))
        if len(video_inputs.get("video_metadata", [])) != num_videos:
            raise ValueError("video_metadata must contain one entry per video.")
        use_audio_in_video = (
            [use_audio_in_video] * num_videos if isinstance(use_audio_in_video, bool) else list(use_audio_in_video)
        )
        if len(use_audio_in_video) != num_videos or not all(isinstance(flag, bool) for flag in use_audio_in_video):
            raise ValueError("use_audio_in_video must contain one flag per video.")
        if text is None and any(use_audio_in_video):
            raise ValueError("Text is required to align audio inputs with video placeholders.")
        text_inputs = {}
        audio_segment_offsets = []
        if text is not None:
            return_replacement_offsets = text_kwargs.pop("return_text_replacement_offsets", False)
            text, audio_segment_offsets, replacement_offsets = self._expand_multimodal_tokens(
                text,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                audio_inputs=audio_inputs,
                images_kwargs=output_kwargs["images_kwargs"],
                videos_kwargs=video_kwargs,
                use_audio_in_video=use_audio_in_video,
                interleave_length=interleave_length,
                use_per_grid_t_timestamps=use_per_grid_t_timestamps,
                return_replacement_offsets=return_replacement_offsets,
            )
            return_token_types = text_kwargs.pop("return_mm_token_type_ids", False)
            text_inputs = self.tokenizer(text, **text_kwargs)
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])
            if return_token_types:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
            if return_replacement_offsets:
                text_inputs["text_replacement_offsets"] = replacement_offsets
        data = {**text_inputs, **image_inputs, **video_inputs, **audio_inputs}
        if audio_segment_offsets:
            data["audio_segment_offsets"] = audio_segment_offsets
        if not kwargs.get("return_metadata", kwargs.get("videos_kwargs", {}).get("return_metadata", False)):
            data.pop("video_metadata", None)
        return BatchFeature(
            data, tensor_type=text_kwargs.get("return_tensors"), skip_tensor_conversion=self.skip_tensor_conversion
        )

    def _expand_multimodal_tokens(
        self,
        text: list[str],
        image_inputs: BatchFeature | dict,
        video_inputs: BatchFeature | dict,
        audio_inputs: BatchFeature | dict,
        images_kwargs: dict,
        videos_kwargs: dict,
        use_audio_in_video: list[bool],
        interleave_length: float,
        use_per_grid_t_timestamps: bool,
        return_replacement_offsets: bool = False,
    ) -> tuple[list[str], list[tuple[int, int, int]], list[list[dict]]]:
        """Expand placeholders in prompt order and collect the corresponding audio segments."""
        image_grid_thw = image_inputs.get("image_grid_thw", [])
        video_grid_thw = video_inputs.get("video_grid_thw", [])
        audio_lengths = audio_inputs.get("audio_feature_lens", [])
        image_merge_size = images_kwargs.get("merge_size", self.image_processor.merge_size)
        video_merge_size = videos_kwargs.get("merge_size", self.video_processor.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.video_processor.temporal_patch_size)
        temporal_compression_ratio = videos_kwargs.get(
            "temporal_compression_ratio", self.video_processor.temporal_compression_ratio
        )
        counts = {"image": 0, "video": 0, "audio": 0}
        expected_counts = {"image": len(image_grid_thw), "video": len(video_grid_thw), "audio": len(audio_lengths)}
        audio_segment_offsets = []

        def replace_placeholder(match):
            modality = match.lastgroup
            index = counts[modality]
            if index >= expected_counts[modality]:
                raise ValueError(f"More {modality} placeholders than {modality} inputs.")
            counts[modality] += 1

            if modality == "image":
                num_image_tokens = math.prod(int(size) for size in image_grid_thw[index]) // image_merge_size**2
                return self.vision_start_token + self.image_token * num_image_tokens + self.vision_end_token

            if modality == "video":
                grid_t, grid_h, grid_w = (int(size) for size in video_grid_thw[index])
                timestamps, video_end_time = self._get_video_timestamps(
                    video_inputs["video_metadata"][index],
                    grid_t=grid_t,
                    temporal_patch_size=temporal_patch_size,
                    temporal_compression_ratio=temporal_compression_ratio,
                )
                num_video_tokens = grid_h * grid_w // video_merge_size**2
                if not use_audio_in_video[index]:
                    frames = [
                        self._format_timestamp(timestamp)
                        + self.vision_start_token
                        + self.video_token * num_video_tokens
                        + self.vision_end_token
                        for timestamp in timestamps
                    ]
                    return self.video_start_token + "".join(frames) + self.video_end_token

                audio_index = counts["audio"]
                if audio_index >= len(audio_lengths):
                    raise ValueError("Missing audio for an audio-enabled video.")
                counts["audio"] += 1
                video_text, video_audio_segments = self._build_video_audio_text(
                    timestamps,
                    video_end_time=video_end_time,
                    num_video_tokens=num_video_tokens,
                    audio_length=int(audio_lengths[audio_index]),
                    audio_index=audio_index,
                    interleave_length=interleave_length,
                    use_per_grid_t_timestamps=use_per_grid_t_timestamps,
                )
                audio_segment_offsets.extend(video_audio_segments)
                return video_text

            audio_length = int(audio_lengths[index])
            audio_segment_offsets.append((index, 0, audio_length))
            return self.audio_start_token + self.audio_token * audio_length + self.audio_end_token

        pattern = self._get_multimodal_token_pattern()
        expanded_text, batch_replacement_offsets = [], []
        for sample in text:
            if not return_replacement_offsets:
                expanded_text.append(pattern.sub(replace_placeholder, sample))
                batch_replacement_offsets.append([])
                continue
            offset = 0
            replacement_offsets = []

            def replace_with_offsets(match):
                nonlocal offset
                replacement = replace_placeholder(match)
                start, end = match.span()
                replacement_offsets.append(
                    {
                        "type": match.lastgroup,
                        "span": (start, end),
                        "new_span": (start + offset, start + offset + len(replacement)),
                        "text": match.group(),
                        "replacement": replacement,
                    }
                )
                offset += len(replacement) - (end - start)
                return replacement

            expanded_text.append(pattern.sub(replace_with_offsets, sample))
            batch_replacement_offsets.append(replacement_offsets)
        if counts != expected_counts:
            raise ValueError("Every image, video and audio input must have a matching placeholder.")
        return expanded_text, audio_segment_offsets, batch_replacement_offsets

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

    @staticmethod
    def _get_video_timestamps(
        metadata: VideoMetadata, grid_t: int, temporal_patch_size: int, temporal_compression_ratio: int
    ) -> tuple[np.ndarray, float]:
        """Return timestamps for compressed temporal grids and the exclusive end of the sampled video."""
        if metadata.fps is None or not math.isfinite(metadata.fps) or metadata.fps <= 0:
            raise ValueError("Video timestamps require video_metadata with a finite positive fps.")
        timestamps = np.asarray(metadata.timestamps, dtype=np.float32)
        if (
            not len(timestamps)
            or not np.isfinite(timestamps).all()
            or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError("Video timestamps must be finite and nondecreasing.")
        if grid_t % temporal_compression_ratio:
            raise ValueError("Video grid length must be divisible by temporal_compression_ratio.")
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        num_timestamp_groups = grid_t // temporal_compression_ratio
        if math.ceil(len(timestamps) / temporal_factor) != num_timestamp_groups:
            raise ValueError("Video metadata frame indices do not match the processed frames.")
        indices = np.minimum(np.arange(num_timestamp_groups) * temporal_factor, len(timestamps) - 1)
        interval = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
        if interval <= 0:
            interval = 1.0
        return timestamps[indices], timestamps[-1] + interval

    def _build_video_audio_text(
        self,
        timestamps: np.ndarray,
        video_end_time: float,
        num_video_tokens: int,
        audio_length: int,
        audio_index: int,
        interleave_length: float,
        use_per_grid_t_timestamps: bool,
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Build an interleaved video/audio prompt and its grouped-codec offsets."""
        audio_tokens_per_second = self.feature_extractor.audio_tokens_per_second
        end_timestamps = list(timestamps[1:]) + [video_end_time]
        units = []
        audio_segment_offsets = []
        for timestamp, next_timestamp in zip(timestamps, end_timestamps):
            start = int(timestamp * audio_tokens_per_second)
            length = min(int(next_timestamp * audio_tokens_per_second), audio_length) - start
            if length < 0:
                raise ValueError("Video timestamps extend beyond the audio token sequence.")
            units.append((timestamp, length))
            audio_segment_offsets.append((audio_index, start, length))
        if interleave_length == -1:
            groups = [units]
        elif interleave_length == 0:
            groups = [[unit] for unit in units]
        else:
            groups = [
                list(group)
                for _, group in groupby(units, key=lambda unit: math.floor(float(unit[0]) / interleave_length))
            ]
        output = [self.video_start_token]
        for group in groups:
            if not use_per_grid_t_timestamps:
                output.append(self._format_timestamp(group[0][0]))
            video_tokens = []
            audio_tokens = []
            for timestamp, audio_count in group:
                prefix = self._format_timestamp(timestamp) if use_per_grid_t_timestamps else ""
                video_tokens.append(
                    prefix + self.vision_start_token + self.video_token * num_video_tokens + self.vision_end_token
                )
                audio_tokens.append(prefix + self.audio_token * audio_count)
            output.extend(("".join(video_tokens), self.audio_start_token, "".join(audio_tokens), self.audio_end_token))
        output.append(self.video_end_token)
        return "".join(output), audio_segment_offsets

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
                + [
                    "input_features",
                    "input_audio_codes",
                    "feature_attention_mask",
                    "audio_feature_lens",
                    "audio_segment_offsets",
                    "mm_token_type_ids",
                ]
            )
        )


__all__ = ["MiMoV2Processor"]
