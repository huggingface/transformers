# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Processor for the NemotronH Omni model (image + video + audio + text)."""

import math
from typing import Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available
from ...video_utils import VideoInput


if is_torch_available():
    import torch


# Audio input type - file paths, numpy arrays, or torch tensors (or lists thereof).
AudioInput = Union[str, "np.ndarray", "torch.Tensor", list]


class NemotronH_Omni_Reasoning_V3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ImagesKwargs
    videos_kwargs: VideosKwargs
    # `ProcessingKwargs` is a TypedDict, so `_defaults` is not inherited by subclasses and must be
    # (re)declared. Left empty on purpose — no defaults are overridden here (`padding` already
    # defaults to `False` in the tokenizer), so nothing needs to live in `processor_config.json`.
    _defaults = {}


@auto_docstring
class NemotronH_Omni_Reasoning_V3Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        feature_extractor=None,
        chat_template=None,
        audio_sampling_rate: int = 16000,
        audio_subsampling_factor: int = 8,
        audio_hop_length: int = 160,
        video_temporal_patch_dim: int = 2,
        **kwargs,
    ):
        r"""
        audio_sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate, in Hz, the audio waveforms are expected to be at.
        audio_subsampling_factor (`int`, *optional*, defaults to 8):
            Factor by which the sound encoder subsamples the mel frames, used to size the audio
            placeholder run.
        audio_hop_length (`int`, *optional*, defaults to 160):
            Hop length, in samples, between consecutive mel frames.
        video_temporal_patch_dim (`int`, *optional*, defaults to 2):
            Number of frames collapsed into a single temporal patch by the model's video embedder.
        """
        self.video_temporal_patch_dim = video_temporal_patch_dim
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<video>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.audio_token = "<so_embedding>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        self.audio_start_token = "<so_start>"
        self.audio_end_token = "<so_end>"
        self.image_start_token = (
            "<img>" if not hasattr(tokenizer, "image_start_token") else tokenizer.image_start_token
        )
        self.image_end_token = "</img>" if not hasattr(tokenizer, "image_end_token") else tokenizer.image_end_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_hop_length = audio_hop_length

        super().__init__(image_processor, video_processor, tokenizer, feature_extractor, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[NemotronH_Omni_Reasoning_V3ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            NemotronH_Omni_Reasoning_V3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs, videos_inputs, audio_inputs = {}, {}, {}

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_num_patches = [sum(videos_inputs["num_patches"])]

        audio_num_tokens = []
        if audio is not None:
            audio_features, audio_mask, audio_num_tokens = self._process_audio(
                audio, output_kwargs.get("audio_kwargs", {})
            )
            audio_inputs["input_features"] = audio_features
            if audio_mask is not None:
                audio_inputs["input_features_mask"] = audio_mask

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        images_replacements, videos_replacements, audio_replacements = [], [], []
        if images is not None:
            images_replacements = [
                self.replace_image_token(image_inputs, idx) for idx in range(len(image_inputs["image_grid_hw"]))
            ]
        if videos is not None:
            assert len(text) == 1, "Video is not supported for batch size > 1"
            videos_replacements = [
                self.replace_video_token(
                    videos_inputs,
                    0,
                    num_frames=video_num_patches[0],
                    video_metadata=output_kwargs.get("videos_kwargs", {}).get("video_metadata", None),
                )
            ]
        if audio is not None:
            audio_replacements = [
                self.replace_audio_token({"num_tokens": audio_num_tokens}, idx) for idx in range(len(audio_num_tokens))
            ]

        text, _ = self.get_text_with_replacements(
            text,
            images_replacements=images_replacements,
            videos_replacements=videos_replacements,
            audio_replacements=audio_replacements,
        )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # the video processor also returns the per-frame token counts used for the placeholder expansion above
        videos_inputs = {k: v for k, v in videos_inputs.items() if k in self.video_processor.model_input_names}
        output_data = {**text_inputs, **image_inputs, **videos_inputs}
        result = BatchFeature(data=output_data, tensor_type=return_tensors)

        for key, value in audio_inputs.items():
            result[key] = value

        return result

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        grid_height, grid_width = image_inputs["image_grid_hw"][image_idx].tolist()
        merge_size = round(1 / self.image_processor.downsample_ratio)
        n_tokens = grid_height * grid_width // merge_size**2
        return self.image_start_token + self.image_token * n_tokens + self.image_end_token

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        """Expand `<video>` into one `<img>...</img>` chunk per temporal patch (tubelet).

        Each chunk is labeled with the timestamps of the frames it packs, joined by " and "
        ("Frame" for the first frame in the tubelet, "frame" for the rest). The tokenizer has no
        real `<video>` token, so the chunks use the image token; the model tells image from video
        by which `pixel_values_*` argument was passed.
        """
        num_frames = kwargs["num_frames"]
        video_metadata = kwargs.get("video_metadata")
        tokens_per_tubelet = int(video_inputs["num_tokens"][video_idx])
        each_group = self.image_start_token + self.image_token * tokens_per_tubelet + self.image_end_token

        temporal_patch_dim = self.video_temporal_patch_dim
        n_groups = (num_frames + temporal_patch_dim - 1) // temporal_patch_dim

        source_fps = video_metadata.fps if (video_metadata is not None and video_metadata.fps) else None
        frames_indices = video_metadata.frames_indices if video_metadata is not None else None
        if source_fps is not None:
            frame_duration_ms = int(1000.0 / source_fps)

        frame_labels = []
        for group in range(n_groups):
            parts = []
            for offset in range(temporal_patch_dim):
                frame_index = group * temporal_patch_dim + offset
                if frame_index >= num_frames:
                    break  # last group may be short
                prefix = "Frame" if offset == 0 else "frame"
                if source_fps is not None and frames_indices is not None and frame_index < len(frames_indices):
                    ts = int(frames_indices[frame_index]) * frame_duration_ms / 1000.0
                    parts.append(f"{prefix} {frame_index + 1} sampled at {ts:.2f} seconds")
                elif source_fps is not None:
                    ts = frame_index / source_fps
                    parts.append(f"{prefix} {frame_index + 1} sampled at {ts:.2f} seconds")
                else:
                    parts.append(f"{prefix} {frame_index + 1}")
            frame_labels.append(" and ".join(parts) + ": ")

        return "\n".join(label + each_group for label in frame_labels)

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_tokens = audio_inputs["num_tokens"]
        n_tokens = int(num_tokens[audio_idx]) if audio_idx < len(num_tokens) else 1
        return self.audio_start_token + self.audio_token * n_tokens + self.audio_end_token

    def _process_audio(self, audio: AudioInput, audio_kwargs: dict) -> tuple:
        """Extract mel features from the clip(s) and estimate the number of audio embedding tokens."""
        sampling_rate = audio_kwargs.get("sampling_rate", self.audio_sampling_rate)

        if not isinstance(audio, list):
            audio = [audio]

        audio_clips = []
        num_tokens = []
        for audio_item in audio:
            if isinstance(audio_item, str):
                waveform = self._load_audio(audio_item, sampling_rate)
            elif is_torch_available() and isinstance(audio_item, torch.Tensor):
                waveform = audio_item.numpy() if audio_item.dim() == 1 else audio_item.squeeze().numpy()
            elif isinstance(audio_item, np.ndarray):
                waveform = audio_item.squeeze() if audio_item.ndim > 1 else audio_item
            else:
                raise ValueError(f"Unsupported audio type: {type(audio_item)}")

            audio_clips.append(waveform)
            n_tokens = self._estimate_audio_num_embeddings(len(waveform))
            num_tokens.append(max(1, n_tokens))

        features = self.feature_extractor(audio_clips, sampling_rate=sampling_rate, return_tensors="pt")
        return features.input_features, features.get("attention_mask", None), num_tokens

    def _estimate_audio_num_embeddings(self, audio_length_samples: int) -> int:
        """Predict the number of `<so_embedding>` tokens the sound encoder emits for a raw clip.

        Mirrors `ParakeetFeatureExtractor` (center-padded STFT -> ``1 + L // hop`` mel frames)
        followed by the encoder's conv-subsampling (``log2(subsampling_factor)`` stride-2 stages).
        """
        n_frames = 1 + audio_length_samples // self.audio_hop_length
        kernel_size = getattr(self, "audio_subsampling_conv_kernel_size", 3)
        stride = getattr(self, "audio_subsampling_conv_stride", 2)
        num_layers = int(math.log2(self.audio_subsampling_factor))
        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        length = n_frames
        for _ in range(num_layers):
            length = (length + add_pad) // stride + 1
        return length

    def _load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        """Load (and resample) audio from a file path, using librosa or soundfile if available."""
        try:
            import librosa

            waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
            return waveform
        except ImportError:
            pass

        try:
            import soundfile as sf

            waveform, sr = sf.read(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != target_sr:
                import scipy.signal

                num_samples = int(len(waveform) * target_sr / sr)
                waveform = scipy.signal.resample(waveform, num_samples)
            return waveform.astype(np.float32)
        except ImportError:
            pass

        raise ImportError(
            "Audio loading requires either librosa or soundfile. Install with: pip install librosa soundfile"
        )

    def batch_decode(self, *args, **kwargs):
        """Forward to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to the tokenizer's [`~PreTrainedTokenizer.decode`]."""
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """Decode the model's generated token ids into text."""
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["NemotronH_Omni_Reasoning_V3Processor"]
