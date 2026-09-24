# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for YuE"""

import re

import numpy as np

from ...audio_utils import make_list_of_audio
from ...processing_utils import AudioKwargs, BatchFeature, ProcessingKwargs, ProcessorMixin
from ...utils import is_torch_available


if is_torch_available():
    import torch


class YuEAudioKwargs(AudioKwargs, total=False):
    eoa_token_id: int
    soa_token_id: int
    xcodec_marker_token_id: int
    start_of_reference_token_id: int
    end_of_reference_token_id: int
    prompt_start_time: float
    prompt_end_time: float
    codebook_size: int
    num_codebooks: int
    global_offset: int
    fps: int
    sample_rate: int


class YuEProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: YuEAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "truncation": False,
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "eoa_token_id": 32002,
            "soa_token_id": 32001,
            "xcodec_marker_token_id": 32016,
            "start_of_reference_token_id": [518, 2962, 29918, 974, 29918, 5679, 29962],
            "end_of_reference_token_id": [518, 2962, 29918, 974, 29918, 5679, 29962],
            "prompt_start_time": 0.0,
            "prompt_end_time": 5.0,
            "codebook_size": 1024,
            "num_codebooks": 12,
            "global_offset": 45334,
            "fps": 50,
            "sample_rate": 16000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class YuEProcessor(ProcessorMixin):
    """
    Constructs a YuE processor which wraps a YuE tokenizer and a finetuned XCodec audio tokenizer into a single processor.

    [`YuEProcessor`] offers all the functionalities of [`YuETokenizer`] and [`XCodecModel`]. See the
    [`~YuEProcessor.__call__`] and [`~YuEProcessor.decode`] for more information.

    Args:
        tokenizer ([`YuETokenizer`]):
            The tokenizer is a required input.
        audio_tokenizer ([`XCodecModel`]):
            The audio tokenizer is a required input.
    """

    tokenizer_class = "YuETokenizer"
    audio_tokenizer_class = "XCodecModel"
    attributes = ["tokenizer", "audio_tokenizer"]

    def __init__(self, tokenizer, audio_tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.feature_extractor = feature_extractor

    def __call__(
        self,
        text=None,
        lyrics_segments=None,
        genre_tags=None,
        audio=None,
        return_tensors=None,
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(YuEProcessorKwargs, **kwargs)
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]

        batch_lyrics_segments, batch_genre_tags = self._normalize_inputs(text, lyrics_segments, genre_tags)
        batch_main_prompts = [
            self._build_main_prompt(segments, genres)
            for segments, genres in zip(batch_lyrics_segments, batch_genre_tags)
        ]

        text_kwargs.pop("return_tensors", None)

        # tokenize main prompt with genre and full lyrics (this is head_ids)
        tokenizer_output = self.tokenizer(batch_main_prompts, **text_kwargs)
        head_prompt_ids = tokenizer_output["input_ids"]
        head_attention_mask = tokenizer_output["attention_mask"]

        if audio is not None and self.audio_tokenizer is not None:
            print("first audio: ", [au.shape for au in audio])
            audio = make_list_of_audio(audio)

            print("after make_list_of_audio: ", [au.shape for au in audio])

            input_audios = self.feature_extractor(audio, sampling_rate=audio_kwargs.get("sample_rate"))

            print("YuEProcessor FE: input_values shape =", input_audios["input_values"].shape)
            print("YuEProcessor FE: padding_mask shape =", input_audios["padding_mask"].shape)

            with torch.no_grad():
                encoded = self.audio_tokenizer.encode(
                    input_values=input_audios["input_values"],  # (B, 1, T)
                    bandwidth=0.5,
                )
                audio_codes = encoded.audio_codes  # (B, num_codebooks, T_frames)

            print("YuEProcessor: audio_codes shape =", audio_codes.shape)

            # update heads with audio prompt tokens, batched
            head_prompt_ids, head_attention_mask = self._process_audio_prompt(
                head_prompt_ids, head_attention_mask, audio_codes, audio_kwargs, self.tokenizer.pad_token_id
            )

        # batching segments so that lyrics_segments_ids shape is (batch_size, max_num_segments, max_segment_length)
        # so that the stage 1 generation loop can iterate over lyrics_segments_ids[:, segment_idx, :]
        # to support batched generation seamlessly

        # max_num_segments is the max number of segments in the batch.
        batch_size = len(batch_lyrics_segments)
        max_num_segments = max(len(segment) for segment in batch_lyrics_segments)
        segment_token_ids = []
        max_segment_length = 0

        for segment_position in range(max_num_segments):
            # build the list of segment texts at this position for each sample
            segments_at_position = []
            for i in range(batch_size):
                if segment_position < len(batch_lyrics_segments[i]):
                    segments_at_position.append(batch_lyrics_segments[i][segment_position])
                else:
                    segments_at_position.append("")

            tokenized = self.tokenizer(segments_at_position, **text_kwargs)
            ids_per_batch = tokenized["input_ids"]

            # making sure missing segments are represented as empty lists
            for sample_idx, segment_text in enumerate(segments_at_position):
                if not segment_text:
                    ids_per_batch[sample_idx] = []

            max_len_seg = max(len(ids) for ids in ids_per_batch) if ids_per_batch else 0
            max_segment_length = max(max_segment_length, max_len_seg)
            segment_token_ids.append(ids_per_batch)

        # pad to (batch_size, max_num_segments, max_segment_length) with missing segments have all pad
        lyrics_segments_ids = []
        lyrics_attention_mask = []

        for batch_idx in range(batch_size):
            sample_segment_ids = []
            sample_segment_mask = []

            for segment_idx in range(max_num_segments):
                ids = segment_token_ids[segment_idx][batch_idx]
                if not ids:
                    # filling missing segments with padding values
                    sample_segment_ids.append([self.tokenizer.pad_token_id] * max_segment_length)
                    sample_segment_mask.append([0] * max_segment_length)
                else:
                    pad_len = max_segment_length - len(ids)
                    sample_segment_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                    sample_segment_mask.append([1] * len(ids) + [0] * pad_len)

            lyrics_segments_ids.append(sample_segment_ids)
            lyrics_attention_mask.append(sample_segment_mask)

        data = {
            "head_prompt_ids": torch.tensor(head_prompt_ids),
            "head_attention_mask": torch.tensor(head_attention_mask),
            "lyrics_segments_ids": torch.tensor(lyrics_segments_ids),
            "lyrics_attention_mask": torch.tensor(lyrics_attention_mask),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    @staticmethod
    def _split_lyrics_into_segments(lyrics):
        """Split lyrics into segments based on structure tags like [verse], [chorus], etc"""
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

    @staticmethod
    def _build_main_prompt(segments, genres):
        genres = ", ".join(genres) if genres else ""
        full_lyrics = "\n".join(segments)
        return f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"

    def _normalize_inputs(self, text, lyrics_segments, genre_tags):
        if text is None and lyrics_segments is None:
            raise ValueError("Either `lyrics_segments` or `text` must be provided.")

        if text is not None:
            if isinstance(text, str):
                lyrics_segments = [self._split_lyrics_into_segments(text)]
            elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
                lyrics_segments = [self._split_lyrics_into_segments(t) for t in text]
            else:
                raise ValueError("Invalid input `text`. Please provide a string or a list of strings")

        if lyrics_segments is not None:
            if isinstance(lyrics_segments, list):
                if isinstance(lyrics_segments[0], str):
                    lyrics_segments = [lyrics_segments]

                elif all(isinstance(segment_list, list) for segment_list in lyrics_segments):
                    lyrics_segments = [list(segment_list) for segment_list in lyrics_segments]
            else:
                raise ValueError(
                    "Invalid input lyrics_segments. Please provide a list of strings or a list of list of strings as batch"
                )

        if genre_tags is not None:
            if isinstance(genre_tags, str):
                genre_tags = [[genre_tags]]
            elif isinstance(genre_tags, (list, tuple)) and all(isinstance(tag, str) for tag in genre_tags):
                genre_tags = [list(genre_tags)]
            elif isinstance(genre_tags, (list, tuple)) and all(
                isinstance(tags, (list, tuple)) and all(isinstance(tag, str) for tag in tags) for tags in genre_tags
            ):
                genre_tags = [list(tags) for tags in genre_tags]
        else:
            raise ValueError(
                "Please provide `genre_tags`, it must be str, a list of strings or a list of list of strings as batch"
            )

        return lyrics_segments, genre_tags

    def _process_audio_prompt(self, head_prompt_ids, head_attention_mask, audio_codes, audio_kwargs, pad_token_id):
        fps = audio_kwargs.get("fps", 50)
        prompt_start_time = audio_kwargs.get("prompt_start_time", 0.0)
        prompt_end_time = audio_kwargs.get("prompt_end_time", None)

        eoa_token_id = audio_kwargs.get("eoa_token_id")
        soa_token_id = audio_kwargs.get("soa_token_id")
        xcodec_marker_token_id = audio_kwargs.get("xcodec_marker_token_id")

        batch_size = len(head_prompt_ids)
        print("YuEProcessor: _process_audio_prompt batch_size =", batch_size)
        print("YuEProcessor: _process_audio_prompt audio_codes shape =", audio_codes.shape)

        audio_augmented_heads = []

        for i in range(batch_size):
            head_ids = [token for token in head_prompt_ids[i] if token != pad_token_id]
            print(f" sample {i}: original head len =", len(head_ids))

            codes_i = audio_codes[i : i + 1, 0, :].cpu().numpy()
            print(f" sample {i}: codes_i shape =", codes_i.shape)

            audio_ids_full = self._offset_and_flatten_tokens(codes_i, audio_kwargs)
            print(f" sample {i}: audio_ids_full len =", len(audio_ids_full))

            start = int(prompt_start_time * fps)
            end = int(prompt_end_time * fps)
            audio_ids = audio_ids_full[start:end]
            print(f" sample {i}: slicing frames [{start}:{end}] -> len =", len(audio_ids))

            # [SOA] + <xcodec> + codes + [EOA]
            audio_ids = [soa_token_id] + [xcodec_marker_token_id] + audio_ids + [eoa_token_id]

            start_of_reference = self.tokenizer("[start_of_reference]", add_special_tokens=False)["input_ids"]
            end_of_reference = self.tokenizer("[end_of_reference]", add_special_tokens=False)["input_ids"]
            audio_ids = start_of_reference + audio_ids + end_of_reference
            print(f" sample {i}: audio prompt tokens len =", len(audio_ids))

            full_ids = head_ids + audio_ids
            print(f" sample {i}: new head len {len(full_ids)}")

            audio_augmented_heads.append(full_ids)

        encoded = {"input_ids": audio_augmented_heads}
        padded = self.tokenizer.pad(encoded, padding=True, return_attention_mask=True, return_tensors=None)

        padded_heads = padded["input_ids"]
        padded_masks = padded["attention_mask"]

        return padded_heads, padded_masks

    def _offset_and_flatten_tokens(self, audio_codes, audio_kwargs):
        print("audio_codes.shape :", audio_codes.shape)
        if audio_codes.ndim != 2 or audio_codes.shape[0] != 1:
            raise ValueError(f"Audio codes shape should be (1, T), got {audio_codes.shape}")

        # TODO handle this as well
        codebook_size = audio_kwargs.get("codebook_size", 1024)
        global_offset = audio_kwargs.get("global_offset", 45334)

        if audio_codes.max() >= codebook_size:
            raise ValueError(f"max(audio_codes)={audio_codes.max()}, codebook_size={codebook_size}")
        if audio_codes.min() < 0:
            raise ValueError(f"min(audio_codes)={audio_codes.min()}, must be >= 0")

        # apply offset to audio codes then flatten like original yue implementation
        # does  offset = global_offset + k * codebook_size for each quantizer k
        # for one quantizer k=0 so only global_offset is added
        # see https://github.com/multimodal-art-projection/YuE/blob/main/inference/codecmanipulator.py#L90

        offset_codes = audio_codes.copy().astype(np.uint32)
        offset_codes[0] += global_offset
        flattened_tokens = offset_codes.flatten()

        return flattened_tokens.tolist()

    @staticmethod
    def _build_main_prompt(segments: list[str], genres: list[str]) -> str:
        genres = ", ".join(genres) if genres else ""
        full_lyrics = "\n".join(segments)
        return f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"

    def _normalize_inputs(self, text, lyrics_segments, genre_tags):
        if text is None and lyrics_segments is None:
            raise ValueError("Either `lyrics_segments` or `text` must be provided.")

        if text is not None:
            if isinstance(text, str):
                lyrics_segments = [self._split_lyrics_into_segments(text)]
            elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
                lyrics_segments = [self._split_lyrics_into_segments(t) for t in text]
            else:
                raise ValueError("Invalid input `text`. Please provide a string or a list of strings")

        if lyrics_segments is not None:
            if isinstance(lyrics_segments, list):
                if isinstance(lyrics_segments[0], str):
                    lyrics_segments = [lyrics_segments]
                elif all(isinstance(segment_list, list) for segment_list in lyrics_segments):
                    lyrics_segments = [list(segment_list) for segment_list in lyrics_segments]
            else:
                raise ValueError(
                    "Invalid input lyrics_segments. Please provide a list of strings or a list of list of strings as batch"
                )

        if genre_tags is not None:
            if isinstance(genre_tags, str):
                genre_tags = [[genre_tags]]
            elif isinstance(genre_tags, (list, tuple)) and all(isinstance(tag, str) for tag in genre_tags):
                genre_tags = [list(genre_tags)]
            elif isinstance(genre_tags, (list, tuple)) and all(
                isinstance(tags, (list, tuple)) and all(isinstance(tag, str) for tag in tags) for tags in genre_tags
            ):
                genre_tags = [list(tags) for tags in genre_tags]
        else:
            raise ValueError(
                "Please provide `genre_tags`, it must be str, a list of strings or a list of list of strings as batch"
            )

        return lyrics_segments, genre_tags
