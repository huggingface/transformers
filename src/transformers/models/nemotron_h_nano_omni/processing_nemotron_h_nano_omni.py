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
"""Processor for the NemotronH Nano Omni model (image + video + audio + text)."""

import math
from typing import Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available
from ...video_utils import VideoInput


if is_torch_available():
    import torch


# Audio input type - file paths, numpy arrays, or torch tensors (or lists thereof).
AudioInput = Union[str, "np.ndarray", "torch.Tensor", list]


class NemotronH_Nano_Omni_Reasoning_V3ImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None


class NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: NemotronH_Nano_Omni_Reasoning_V3ImagesKwargs
    videos_kwargs: VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class NemotronH_Nano_Omni_Reasoning_V3Processor(ProcessorMixin):
    r"""
    Constructs a NemotronH Nano Omni processor which wraps an image processor and a tokenizer into a
    single processor.

    [`NemotronH_Nano_Omni_Reasoning_V3Processor`] offers all the functionalities of the image
    processor and tokenizer. See [`~NemotronH_Nano_Omni_Reasoning_V3Processor.__call__`] and
    [`~NemotronH_Nano_Omni_Reasoning_V3Processor.decode`] for more information.

    Args:
        image_processor ([`NemotronH_Nano_Omni_Reasoning_V3ImageProcessor`], *optional*):
            The image processor.
        tokenizer ([`AutoTokenizer`], *optional*):
            The tokenizer.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a
            tokenizable string.
        audio_sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate for audio processing.
        audio_subsampling_factor (`int`, *optional*, defaults to 8):
            Subsampling factor for the audio encoder.
        audio_hop_length (`int`, *optional*, defaults to 160):
            Hop length in samples for feature extraction.
        video_temporal_patch_dim (`int`, *optional*, defaults to 2):
            Number of frames collapsed into a single temporal patch by the model's video embedder.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        audio_sampling_rate: int = 16000,
        audio_subsampling_factor: int = 8,
        audio_hop_length: int = 160,
        video_temporal_patch_dim: int = 2,
        **kwargs,
    ):
        # Number of frames collapsed into a single temporal patch by the model's video embedder.
        self.video_temporal_patch_dim = video_temporal_patch_dim
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<video>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.audio_token = "<so_embedding>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        self.audio_start_token = "<so_start>"
        self.audio_end_token = "<so_end>"
        self.image_start_token = "<img>" if not hasattr(tokenizer, "image_start_token") else tokenizer.image_start_token
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

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare multimodal inputs (text, images, videos, audio) for the model.

        Text `<image>` / `<video>` / `<audio>` tokens are expanded into placeholder sequences sized
        by the corresponding media, images/videos are run through the image processor, and audio is
        turned into raw waveforms with an estimated token count. Returns a [`BatchFeature`] with (as
        applicable) `input_ids`, `attention_mask`, `pixel_values`, `num_patches`,
        `pixel_values_videos`, and `sound_clips`.
        """
        output_kwargs = self._merge_kwargs(
            NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs, videos_inputs, audio_inputs = {}, {}, {}

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_tokens = image_inputs["num_tokens"]

        if videos is not None:
            # One tile per frame, but sized with the video rule (aspect ratio preserved). We flip a
            # flag on the image processor around the call rather than routing a new kwarg through
            # the strict `ImagesKwargs` dataclass.
            self.image_processor._is_video_mode = True
            try:
                videos_inputs = self.image_processor(images=videos, **output_kwargs["images_kwargs"])
            finally:
                self.image_processor._is_video_mode = False
            video_num_patches = [sum(videos_inputs["num_patches"])]
            videos_inputs["pixel_values_videos"] = videos_inputs["pixel_values"]
            del videos_inputs["pixel_values"]

        audio_num_tokens = []
        if audio is not None:
            audio_clips, audio_num_tokens = self._process_audio(audio, output_kwargs.get("audio_kwargs", {}))
            audio_inputs["sound_clips"] = audio_clips

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n_tokens = image_num_tokens[index]
                    text[i] = text[i].replace(
                        self.image_token,
                        self.image_start_token + "<|placeholder|>" * n_tokens + self.image_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            assert len(text) == 1, "Video is not supported for batch size > 1"
            video_metadata = output_kwargs.get("videos_kwargs", {}).get("video_metadata", None)
            i = 0
            index = 0
            if self.video_token in text[i]:
                # One `<img>...</img>` chunk per temporal patch (tubelet), labeled with the per-frame
                # timestamps joined by " and " ("Frame" for the first frame in the tubelet, "frame"
                # for the rest).
                tokens_per_tubelet = videos_inputs["num_tokens"][0]
                each_group = self.image_start_token + "<|placeholder|>" * tokens_per_tubelet + self.image_end_token
                T = self.video_temporal_patch_dim
                n_frames = video_num_patches[index]
                n_groups = (n_frames + T - 1) // T

                source_fps = video_metadata.fps if (video_metadata is not None and video_metadata.fps) else None
                frames_indices = video_metadata.frames_indices if video_metadata is not None else None
                if source_fps is not None:
                    frame_duration_ms = int(1000.0 / source_fps)

                frame_labels = []
                for g in range(n_groups):
                    parts = []
                    for j in range(T):
                        fi = g * T + j
                        if fi >= n_frames:
                            break  # last group may be short
                        prefix = "Frame" if j == 0 else "frame"
                        if source_fps is not None and frames_indices is not None and fi < len(frames_indices):
                            ts = int(frames_indices[fi]) * frame_duration_ms / 1000.0
                            parts.append(f"{prefix} {fi + 1} sampled at {ts:.2f} seconds")
                        elif source_fps is not None:
                            ts = fi / source_fps
                            parts.append(f"{prefix} {fi + 1} sampled at {ts:.2f} seconds")
                        else:
                            parts.append(f"{prefix} {fi + 1}")
                    frame_labels.append(" and ".join(parts) + ": ")

                video_prompt = ""
                for g, label in enumerate(frame_labels):
                    if g > 0:
                        video_prompt += "\n"
                    video_prompt += label + each_group

                text[i] = text[i].replace(self.video_token, video_prompt, 1)
            # The tokenizer has no real `<video>` token, so we reuse `<image>` as the placeholder.
            # The model distinguishes image vs. video by which `pixel_values_*` arg was passed.
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if audio is not None:
            index = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    num_tokens = audio_num_tokens[index] if index < len(audio_num_tokens) else 1
                    text[i] = text[i].replace(
                        self.audio_token,
                        self.audio_start_token + "<|audio_placeholder|>" * num_tokens + self.audio_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        output_data = {**text_inputs, **image_inputs, **videos_inputs}
        result = BatchFeature(data=output_data, tensor_type=return_tensors)

        # Audio clips stay as raw waveforms (list of numpy arrays), not tensors.
        if audio_inputs:
            result["sound_clips"] = audio_inputs["sound_clips"]

        return result

    def _process_audio(self, audio: AudioInput, audio_kwargs: dict) -> tuple:
        """Load/normalize audio to waveforms and estimate the number of audio embedding tokens."""
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

        return audio_clips, num_tokens

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


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3Processor"]
