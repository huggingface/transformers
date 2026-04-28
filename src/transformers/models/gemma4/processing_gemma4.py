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


import numpy as np

from ...audio_utils import AudioInput
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
    valid_processor_kwargs = Gemma4ProcessorKwargs

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

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )

        # Model requires nested struct
        if images is not None:
            images = make_nested_list_of_images(images)

        # Create empty text to be replaced with placeholders
        if images and not text:
            text = [" ".join([self.boi_token] * len(image_list)) for image_list in images]
        if audio and not text:
            text = [self.audio_token] * len(audio)

        return images, text, videos, audio

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)

        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        if audio is not None and self.audio_token is None or self.boa_token is None or self.eoa_token is None:
            raise ValueError("Audio inputs were provided, but the tokenizer does not have an `audio_token` defined.")

        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                if len(images) != len(text):
                    raise ValueError(
                        f"Received inconsistently sized batches of images ({len(images)}) and text ({len(text)})."
                    )

                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                        f" Found {n_images_in_text} {self.image_token} tokens and {n_images_in_images} images per sample."
                    )
            elif images is None and any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
        return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        num_soft_tokens = video_inputs["num_soft_tokens_per_video"][video_idx]
        metadata = video_inputs["video_metadata"][video_idx]

        if metadata.fps is None:
            logger.warning_once(
                "Gemma4 requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
            )
        metadata.fps = 24 if metadata.fps is None else metadata.fps

        # mm:ss format for timestamps
        timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
        video_replacement = " ".join(
            [f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}" for t in timestamp_str]
        )
        return video_replacement

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        # TODO: Add tests for audio-only processor inputs.
        mask = audio_inputs["input_features_mask"][audio_idx]

        # Simulate two stride-2 conv blocks on the mask
        t = len(mask)
        for _ in range(2):
            t_out = (t + 2 - 3) // 2 + 1
            mask = mask[::2][:t_out]
            t = len(mask)

        return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"

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
        return super().model_input_names + ["mm_token_type_ids"]

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_soft_tokens_per_image", "num_soft_tokens_per_video"]


__all__ = ["Gemma4Processor"]
