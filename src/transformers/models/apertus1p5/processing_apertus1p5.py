# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Apertus 1.5."""

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from .image_processing_apertus1p5 import Apertus1p5ImageProcessorKwargs


class Apertus1p5ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Apertus1p5ImageProcessorKwargs
    # TypedDict subclasses do not inherit class attributes; `_merge_kwargs` requires this to exist
    _defaults = {}


def _is_nested_media(media) -> bool:
    # one sub-list per text sample; a bare list of numbers is a single audio clip, not a nesting
    return (
        isinstance(media, (list, tuple))
        and len(media) > 0
        and all(
            isinstance(el, (list, tuple)) and (len(el) == 0 or not isinstance(el[0], (int, float))) for el in media
        )
    )


def _is_all_empty(media) -> bool:
    # an empty flat list or nested lists with only empty sub-lists carry no media items
    return isinstance(media, (list, tuple)) and all(isinstance(el, (list, tuple)) and len(el) == 0 for el in media)


@auto_docstring(
    custom_intro="""
    Constructs an Apertus 1.5 processor which wraps an image processor, an audio feature extractor and the
    tokenizer into a single processor: each `<|image|>` / `<|audio|>` placeholder in the text is expanded into
    the model's structured token run, and the media are prepared into the model's tensor inputs.

    Media handling:

    - Media items may be loaded objects or URL / local path strings, which are fetched automatically
      (fetched audio is decoded and resampled to 24 kHz).
    - Flat lists are consumed in batch-sample order, then left-to-right placeholder order, and validate only
      the total count, so items must already follow that order. Nested lists (one sub-list per batch sample,
      empty sub-lists allowed) give explicit ownership and validate counts per sample.
    - Images and audio are tracked independently and may be interleaved arbitrarily; image sizes and
      per-sample media counts may vary freely.

    Input expectations:

    - Images: expected UNSCALED (PIL images or uint8-range pixel values; per the standard `do_rescale`
      convention, float images already in `[0, 1]` would be rescaled again). The image processor converts to
      RGB, resizes, and normalizes to `[-1, 1]`.
    - Audio: bare waveform arrays are assumed to be 24 kHz mono; their absolute scale is irrelevant because
      every clip is peak-normalized to -3 dBFS before feature extraction. Stereo or empty clips are rejected,
      as is a declared `sampling_rate` other than 24000.
    """
)
class Apertus1p5Processor(ProcessorMixin):
    valid_processor_kwargs = Apertus1p5ProcessorKwargs

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image|>"
        self.audio_token = getattr(tokenizer, "audio_token", None) or "<|audio|>"
        self.boi_token = getattr(tokenizer, "boi_token", None) or "<|img_start|>"
        self.eoi_token = getattr(tokenizer, "eoi_token", None) or "<|img_end|>"
        self.image_wrapper_token = getattr(tokenizer, "image_wrapper_token", None) or "<|img_token_start|>"
        self.eol_token = getattr(tokenizer, "eol_token", None) or "<|img_end_of_row|>"
        self.boa_token = getattr(tokenizer, "boa_token", None) or "<|audio_start|>"
        self.eoa_token = getattr(tokenizer, "eoa_token", None) or "<|audio_end|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token, token_id in ((self.image_token, self.image_token_id), (self.audio_token, self.audio_token_id)):
            if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
                raise ValueError(
                    f"The tokenizer does not contain the media placeholder token '{token}'. Apertus 1.5 requires "
                    "a tokenizer with the media special tokens (see the model's `extra_special_tokens`)."
                )
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        audio: AudioInput | None = None,
        **kwargs,
    ):
        # collections without a single media item (e.g. `[[], []]` from a uniform collator) mean "no media"
        if images is not None and _is_all_empty(images):
            images = None
        if audio is not None and _is_all_empty(audio):
            audio = None
        # unlike the base layout, audio may be nested (one sub-list per sample, empty sub-lists allowed) so that
        # per-sample ownership can be validated; clips are flattened again in `_process_audio`
        if audio is not None and _is_nested_media(audio):
            sampling_rate = kwargs.get("sampling_rate", self.feature_extractor.sampling_rate)
            audio = [
                make_list_of_audio(self.feature_extractor.fetch_audio(sublist, sampling_rate=sampling_rate))
                if len(sublist) > 0
                else []
                for sublist in audio
            ]
            images, text, videos, _ = super().prepare_inputs_layout(
                images=images, text=text, videos=videos, audio=None, **kwargs
            )
            return images, text, videos, audio
        return super().prepare_inputs_layout(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        audio: AudioInput | None = None,
        **kwargs,
    ):
        """Run the base validation, then match placeholder and media counts for images and audio."""
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is None:
            if images is not None or audio is not None:
                raise ValueError("`text` with media placeholder tokens must be provided when passing images or audio.")
            return

        self._validate_media_counts(text, images, _is_nested_media(images), self.image_token, "image")
        self._validate_media_counts(text, audio, _is_nested_media(audio), self.audio_token, "audio")

    def _validate_media_counts(self, text, media, is_nested, token, modality):
        """Strict in both directions: every placeholder needs a media item and vice versa."""
        placeholder_counts = [sample.count(token) for sample in text]
        if media is None:
            if any(placeholder_counts):
                raise ValueError(
                    f"Found {sum(placeholder_counts)} '{token}' placeholders in the text but no {modality} "
                    "inputs were passed."
                )
            return
        if is_nested:
            if len(media) != len(text):
                raise ValueError(
                    f"Received {len(media)} {modality} sub-lists for {len(text)} text samples; nested "
                    f"{modality} inputs must provide one sub-list per sample."
                )
            media_counts = [len(sublist) for sublist in media]
            if media_counts != placeholder_counts:
                raise ValueError(
                    f"Per-sample '{token}' placeholder counts {placeholder_counts} do not match the numbers "
                    f"of {modality} inputs {media_counts}."
                )
        else:
            num_media = len(make_flat_list_of_images(media)) if modality == "image" else len(media)
            if sum(placeholder_counts) != num_media:
                raise ValueError(
                    f"The text contains {sum(placeholder_counts)} '{token}' placeholders in total but "
                    f"{num_media} {modality} inputs were passed."
                )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # the grid comes from the image processor's output, so per-call `spatial_factor` overrides stay consistent
        grid_height, grid_width = (int(side) for side in image_inputs["image_grids"][image_idx])
        rows = self.eol_token.join([self.image_token * grid_width] * grid_height)
        return f"{self.boi_token}{grid_height}*{grid_width}{self.image_wrapper_token}{rows}{self.eoi_token}"

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_codes = int(audio_inputs["num_audio_codes"][audio_idx])
        return f"{self.boa_token}{self.audio_token * num_codes}{self.eoa_token}"

    def _process_audio(self, audio: AudioInput, **kwargs):
        """Peak-normalize the clips, extract features under the model's input names, and build the replacements."""
        if _is_nested_media(audio):
            audio = [clip for sublist in audio for clip in sublist]
        # peak-normalize each clip to -3 dBFS in float32, as in the reference Apertus 1.5 pipeline
        # (the codec feature extractor itself deliberately performs no normalization)
        target_peak = 10.0 ** (-3.0 / 20.0)
        clips = []
        for clip in audio:
            clip = np.asarray(clip, dtype=np.float32)
            if clip.size > 0:
                clip = clip * (target_peak / max(float(np.abs(clip).max()), 1e-10))
            clips.append(clip)  # empty clips fall through to the feature extractor's clear error

        # bare arrays are assumed to already be 24 kHz mono, like the reference pipeline
        kwargs.setdefault("sampling_rate", self.feature_extractor.sampling_rate)
        audio_inputs = self.feature_extractor(clips, **kwargs)
        if "padding_mask" not in audio_inputs:
            raise ValueError(
                "The audio feature extractor returned no `padding_mask`; audio must be processed with "
                "`padding=True` (the default)."
            )
        audio_inputs["input_features"] = audio_inputs.pop("input_values")
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("padding_mask")
        # counts come from the feature-extractor OUTPUT so that truncation/max_length can never desync the
        # placeholder count from the features the model will actually encode
        audio_inputs["num_audio_codes"] = [
            self.feature_extractor.get_num_audio_codes(int(np.asarray(mask).sum()))
            for mask in audio_inputs["feature_attention_mask"]
        ]

        audio_replacements = []
        for idx in range(len(clips)):
            audio_replacements.append(self.replace_audio_token(audio_inputs, audio_idx=idx))
        return audio_inputs, audio_replacements

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Counts cover only the `<|image|>` / `<|audio|>` placeholder tokens (one per discrete code), not the
        surrounding structure tokens or the textual `H*W` size header. Audio counts assume untruncated clips;
        explicit `truncation`/`max_length` audio kwargs passed to `__call__` reduce the actual counts.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            audio_lengths (`list[int]`, *optional*):
                The number of audio samples (at 24 kHz) per each clip.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        data = {}
        if image_sizes is not None:
            images_kwargs = dict(kwargs)
            data["num_image_tokens"] = [
                self.image_processor.get_number_of_image_patches(int(height), int(width), images_kwargs)
                for height, width in image_sizes
            ]
            data["num_image_patches"] = [1] * len(image_sizes)
        if audio_lengths is not None:
            data["num_audio_tokens"] = [
                self.feature_extractor.get_num_audio_codes(int(length)) for length in audio_lengths
            ]
        return MultiModalData(**data)

    @property
    def model_input_names(self):
        # the audio names are hardcoded because `_process_audio` renames the feature extractor's
        # `input_values`/`padding_mask` outputs to the model's `input_features`/`feature_attention_mask`
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_input_names = ["input_features", "feature_attention_mask"]
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_input_names))

    @property
    def unused_input_names(self) -> list[str]:
        """Input names returned always by subprocessors but not used in the model's `forward`"""
        return ["image_grids", "num_audio_codes"]


__all__ = ["Apertus1p5Processor"]
