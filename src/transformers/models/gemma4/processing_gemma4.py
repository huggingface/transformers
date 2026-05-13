# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import re

import numpy as np

from ...audio_utils import AudioInput
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available, logging
from ...utils.import_utils import requires
from ...video_utils import VideoInput


if is_vision_available():
    from .image_processing_pil_gemma4 import Gemma4ImageProcessorKwargs, get_aspect_ratio_preserving_size


logger = logging.get_logger(__name__)


class Gemma4ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma4ImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
        "audio_kwargs": {},
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
@requires(backends=("vision",))
class Gemma4Processor(ProcessorMixin):
    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        **kwargs,
    ):
        r"""
        image_seq_length (`int`, *optional*, defaults to 280):
            The number of soft tokens per image used for placeholder expansion.
        audio_seq_length (`int`, *optional*, defaults to 750):
            The maximum number of audio soft tokens per audio segment. Serves as an
            upper-bound cap when dynamic audio token counts are computed.
        audio_ms_per_token (`int`, *optional*, defaults to 40):
            Milliseconds of audio per output soft token. Used to dynamically compute
            the number of audio placeholder tokens as ``ceil(duration_ms / audio_ms_per_token)``.
            The default of 40 comes from the SSCP convolution's 4× time reduction on 10ms frames.
        """
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token

        # FIXME: add the token to config and ask Ryan to re-upload
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
        self.video_token = "<|video|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        # Audio token handling, mirroring the vision pattern.
        # audio_seq_length serves as the maximum cap on the number of audio soft tokens
        # any single audio segment can produce. With dynamic audio tokens, the actual
        # number of placeholders inserted per audio is computed from the audio duration.
        self.audio_seq_length = audio_seq_length
        # Milliseconds of audio per output soft token. The default of 40 comes from the
        # SSCP convolution's 4× time reduction applied to 10ms mel spectrogram frames.
        self.audio_ms_per_token = audio_ms_per_token
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        self.audio_token = getattr(tokenizer, "audio_token", None)
        self.boa_token = getattr(tokenizer, "boa_token", None)
        self.eoa_token = getattr(tokenizer, "eoa_token", None)

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio: AudioInput | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Gemma4ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None and audio is None and videos is None:
            raise ValueError("Provide at least one of `text`, `images`, `audio`, or `videos`.")

        output_kwargs = self._merge_kwargs(
            Gemma4ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            num_soft_tokens = image_inputs.pop("num_soft_tokens_per_image")

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.image_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            replacements = [f"{self.boi_token}{self.image_token * n}{self.eoi_token}" for n in num_soft_tokens]
            replacements_iter = iter(replacements)

            # Expand image_token placeholders to per-image soft token sequences.
            # re.sub never re-scans replaced text, so it is safe
            pattern = re.escape(self.image_token)
            text = [re.sub(pattern, lambda _: next(replacements_iter), prompt) for prompt in text]

        # Process video inputs in same way
        video_inputs = {}
        if videos is not None:
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            num_video_tokens = video_inputs.pop("num_soft_tokens_per_video")

            # If user has not requested video metadata, pop it so it isn't returned
            if not kwargs.get("return_metadata"):
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs["video_metadata"]

            video_replacements = []
            for metadata, n_tokens in zip(video_metadata, num_video_tokens):
                if metadata.fps is None:
                    logger.warning_once(
                        "Gemma 4 requires frame timestamps to construct prompts, but the `fps` of the input video "
                        "could not be inferred. Probably `video_metadata` was missing from inputs and you passed "
                        "pre-sampled frames. Defaulting to `fps=24`. Please provide `video_metadata` for more "
                        "accurate results."
                    )
                metadata.fps = 24 if metadata.fps is None else metadata.fps
                # mm:ss format for timestamps
                timestamp_str = [
                    f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps
                ]
                video_replacements.append(
                    " ".join(
                        [f"{t} {self.boi_token}{self.video_token * n_tokens}{self.eoi_token}" for t in timestamp_str]
                    )
                )

            video_replacements = iter(video_replacements)
            pattern = re.escape(self.video_token)
            text = [re.sub(pattern, lambda _: next(video_replacements), prompt) for prompt in text]

        # Process audio inputs
        audio_inputs = {}
        if audio is not None:
            if self.audio_token is None or self.boa_token is None or self.eoa_token is None:
                raise ValueError(
                    "Audio inputs were provided, but the tokenizer does not have an `audio_token` defined."
                )

            # Normalize audio input to list of waveforms
            if isinstance(audio, np.ndarray) and audio.ndim == 1:
                audio = [audio]

            # TODO: Add tests for audio-only processor inputs.
            if not text:
                text = [self.audio_token] * len(audio)

            # Dynamic audio token expansion wihtout padding:
            #   * Extract audio features with feature extractor;
            #   * Compute precise per-audio token counts from the waveform duration;
            #   * Generate full audio token sequence for each computed audio length;
            #   * Expand text prompts with full audio token sequences.
            audio_kwargs = output_kwargs.get("audio_kwargs", {})
            audio_inputs = self.feature_extractor(audio, **audio_kwargs)
            sampling_rate = self.feature_extractor.sampling_rate
            num_audio_tokens = [self._compute_audio_num_tokens(a, sampling_rate) for a in audio]
            replacements = [f"{self.boa_token}{self.audio_token * n}{self.eoa_token}" for n in num_audio_tokens]
            replacements_iter = iter(replacements)
            audio_pattern = re.escape(self.audio_token)
            text = [re.sub(audio_pattern, lambda _: next(replacements_iter), prompt) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        # Check special tokens for all active modalities
        active_modalities = []
        if images is not None:
            active_modalities.append("image")
        if videos is not None:
            active_modalities.append("video")
        if audio is not None:
            active_modalities.append("audio")
        if active_modalities:
            self._check_special_mm_tokens(text, text_inputs, modalities=active_modalities)

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **audio_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate: int) -> int:
        """Compute the number of audio soft tokens for a single waveform.

        Replicates the exact sequence-length arithmetic of the audio encoder
        so that the processor inserts the correct number of placeholder tokens.
        The computation mirrors:

        1. Mel framing via ``_unfold`` in ``Gemma4AudioFeatureExtractor``
        2. Two ``Conv2d`` subsampling layers in ``Gemma4AudioSubSampleConvProjection``
           (each: kernel=3, stride=2, semicausal padding top=1, bottom=1)

        The result is capped at ``self.audio_seq_length`` (the configured maximum).

        Args:
            audio_waveform: A 1-D numpy array or list containing the raw audio samples.
            sampling_rate: The sampling rate of the audio waveform in Hz.

        Returns:
            The number of audio soft tokens to insert as placeholders.
        """
        num_samples = len(audio_waveform)

        # Step 1: Mel frames (matches feature_extraction_gemma4.py _unfold)
        frame_length = int(round(sampling_rate * 20.0 / 1000.0))  # 320 @ 16kHz
        hop_length = int(round(sampling_rate * 10.0 / 1000.0))  # 160 @ 16kHz
        frame_size_for_unfold = frame_length + 1  # 321

        # The feature extractor prepends (frame_length // 2) zero samples as
        # semicausal time-padding before the unfold.  We must include this to
        # match the actual number of mel frames it produces.
        pad_left = frame_length // 2  # 160 @ 16kHz
        padded_samples = num_samples + pad_left
        num_mel_frames = (padded_samples - frame_size_for_unfold) // hop_length + 1

        if num_mel_frames <= 0:
            return 0

        # Step 2: Two SSCP conv layers (kernel=3, stride=2, semicausal pad top=1, bottom=1)
        # Each layer: T_out = (T_in + pad_top + pad_bottom - kernel) // stride + 1
        t = num_mel_frames
        for _ in range(2):
            t_padded = t + 2  # pad_top=1, pad_bottom=1
            t = (t_padded - 3) // 2 + 1

        # Cap at the configured maximum
        return min(t, self.audio_seq_length)

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            audio_lengths (`list[int]`, *optional*):
                The lengths of audio inputs in number of samples. Used to dynamically
                compute per-audio token counts.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        images_kwargs = Gemma4ProcessorKwargs._defaults.get("images_kwargs", {})
        images_kwargs.update(kwargs)
        patch_size = images_kwargs.get("patch_size", None) or self.image_processor.patch_size
        pooling_kernel_size = (
            images_kwargs.get("pooling_kernel_size", None) or self.image_processor.pooling_kernel_size
        )
        max_soft_tokens = images_kwargs.get("max_soft_tokens", None) or self.image_processor.max_soft_tokens

        max_patches = max_soft_tokens * pooling_kernel_size**2

        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = []
            for image_size in image_sizes:
                target_h, target_w = get_aspect_ratio_preserving_size(
                    height=image_size[0],
                    width=image_size[1],
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                )
                patch_height = target_h // patch_size
                patch_width = target_w // patch_size
                num_image_tokens.append(patch_height * patch_width // pooling_kernel_size**2)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if audio_lengths is not None:
            # Dynamically compute per-audio token counts from sample lengths.
            # audio_lengths are in number of samples; assume default sampling rate.
            sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16_000)
            num_audio_tokens = [
                self._compute_audio_num_tokens(np.zeros(length), sampling_rate) for length in audio_lengths
            ]
            vision_data.update({"num_audio_tokens": num_audio_tokens})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        model_input_names = super().model_input_names
        model_input_names = [
            name
            for name in model_input_names
            if name not in ["num_soft_tokens_per_image", "num_soft_tokens_per_video"]
        ]

        # Include audio feature extractor input names if available
        if self.feature_extractor is not None:
            feature_extractor_input_names = self.feature_extractor.model_input_names
            model_input_names.extend([name for name in feature_extractor_input_names if name not in model_input_names])

        return model_input_names + ["mm_token_type_ids"]


__all__ = ["Gemma4Processor"]
