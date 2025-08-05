# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers import WhisperTokenizer, WhisperTokenizerFast, is_speech_available
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchaudio

from .test_feature_extraction_whisper import floats_list


if is_speech_available():
    from transformers import WhisperFeatureExtractor, WhisperProcessor


TRANSCRIBE = 50358
NOTIMESTAMPS = 50362


@require_torch
@require_torchaudio
@require_sentencepiece
class WhisperProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "openai/whisper-small.en"
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        return WhisperTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return WhisperFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = WhisperProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, WhisperTokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = WhisperProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, WhisperTokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )

    def test_get_decoder_prompt_ids(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe", no_timestamps=True)

        self.assertIsInstance(forced_decoder_ids, list)
        for ids in forced_decoder_ids:
            self.assertIsInstance(ids, (list, tuple))

        expected_ids = [TRANSCRIBE, NOTIMESTAMPS]
        self.assertListEqual([ids[-1] for ids in forced_decoder_ids], expected_ids)

    def test_get_prompt_ids(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        prompt_ids = processor.get_prompt_ids("Mr. Quilter")
        decoded_prompt = processor.tokenizer.decode(prompt_ids)

        self.assertListEqual(prompt_ids.tolist(), [50360, 1770, 13, 2264, 346, 353])
        self.assertEqual(decoded_prompt, "<|startofprev|> Mr. Quilter")

    def test_empty_get_prompt_ids(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        prompt_ids = processor.get_prompt_ids("")
        decoded_prompt = processor.tokenizer.decode(prompt_ids)

        self.assertListEqual(prompt_ids.tolist(), [50360, 220])
        self.assertEqual(decoded_prompt, "<|startofprev|> ")

    def test_get_prompt_ids_with_special_tokens(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())

        def _test_prompt_error_raised_helper(prompt, special_token):
            with pytest.raises(ValueError) as excinfo:
                processor.get_prompt_ids(prompt)
            expected = f"Encountered text in the prompt corresponding to disallowed special token: {special_token}."
            self.assertEqual(expected, str(excinfo.value))

        _test_prompt_error_raised_helper("<|startofprev|> test", "<|startofprev|>")
        _test_prompt_error_raised_helper("test <|notimestamps|>", "<|notimestamps|>")
        _test_prompt_error_raised_helper("test <|zh|> test <|transcribe|>", "<|zh|>")

    def test_find_longest_common_subsequence_old(self):
        """Test using the old processing functions used in the ASR pipeline, but that serves as a BC reference."""
        max_source_positions = 1500
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

        previous_sequence = [[51492, 406, 3163, 1953, 466, 13, 51612, 51612]]
        self.assertEqual(
            processor.decode(previous_sequence[0], output_offsets=True),
            {
                "text": " not worth thinking about.",
                "offsets": [{"text": " not worth thinking about.", "timestamp": (22.56, 24.96)}],
            },
        )

        # Merge when the previous sequence is a suffix of the next sequence
        # fmt: off
        next_sequences_1 = [
            [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 50614, 50614, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]
        ]
        # fmt: on
        self.assertEqual(
            processor.decode(next_sequences_1[0], output_offsets=True),
            {
                "text": (
                    " of spectators, retrievality is not worth thinking about. His instant panic was followed by a"
                    " small, sharp blow high on his chest.<|endoftext|>"
                ),
                "offsets": [
                    {"text": " of spectators, retrievality is not worth thinking about.", "timestamp": (0.0, 5.0)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (5.0, 9.4),
                    },
                ],
            },
        )
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_1, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )

        # fmt: off
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51739, 51739, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51959],
        )
        # fmt: on
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 27.5)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (27.5, 31.900000000000002),
                    },
                ],
            },
        )

        # Merge when the sequence is in the middle of the 1st next sequence
        # fmt: off
        next_sequences_2 = [
            [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]
        ]
        # fmt: on
        # {'text': ' of spectators, retrievality is not worth thinking about. His instant panic was followed by a small, sharp blow high on his chest.','timestamp': (0.0, 9.4)}
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_2, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        # fmt: off
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51959],
        )
        # fmt: on
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {
                        "text": (
                            " not worth thinking about. His instant panic was followed by a small, sharp blow high on"
                            " his chest."
                        ),
                        "timestamp": (22.56, 31.900000000000002),
                    },
                ],
            },
        )

        # Merge when the previous sequence is not included in the current sequence
        next_sequences_3 = [[50364, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50584, 50257]]  # fmt: skip
        # {'text': ' His instant panic was followed by a small, sharp blow high on his chest.','timestamp': (0.0, 9.4)}
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_3, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51612, 51612, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51832],
        )  # fmt: skip
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 24.96)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (24.96, 29.36),
                    },
                ],
            },
        )
        # last case is when the sequence is not in the first next predicted start and end of timestamp
        next_sequences_3 = [
            [50364, 2812, 9836, 14783, 390, 406, 3163, 1953, 466, 13, 50634, 50634, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50934]
        ]  # fmt: skip
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_3, (480_000, 167_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51612, 51612, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51912]
        )  # fmt: skip
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 24.96)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (24.96, 30.96),
                    },
                ],
            },
        )


def _fast_find_longest_common_sequence(sequence_left, sequence_right):
    """Old processing function used in the ASR pipeline."""
    seq_len_left = len(sequence_left)
    seq_len_right = len(sequence_right)
    counter = [[0] * (seq_len_right + 1) for _ in range(seq_len_left + 1)]
    longest = 0
    for i in range(seq_len_left):
        for j in range(seq_len_right):
            if sequence_left[i] == sequence_right[j]:
                previous_counter = counter[i][j] + 1
                counter[i + 1][j + 1] = previous_counter
                if previous_counter > longest:
                    longest = previous_counter

    counter = np.array(counter)
    # we return the idx of the first element of the longest common sequence in the left sequence
    index_left = np.argwhere(counter == longest)[-1][0] - longest if longest != 0 else -1
    index_right = np.argwhere(counter == longest)[-1][1] - longest if longest != 0 else -1
    return index_left, index_right, longest


def _find_timestamp_sequence(sequences, tokenizer, feature_extractor, max_source_positions):
    """
    Old processing function used in the ASR pipeline.

    Computes the final sequences by merging the end of the nth sequence with the beginning of the n+1th sequence. Since
    `WhisperForConditionalGeneration` produces the timestamps pairwise, we filter the consecutive timestamps and only
    iterate over them. We keep track of the `time` which indicates the actual starting time of the chunk that is
    processed. We need to make sure to offset the timestamps tokens by the `time` in order for the tokenizer to
    properly compute the final `offset`.
    """
    # index of the first timestamp token
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    items = []
    # approximation of the token to time ratio : ~0.2seconds
    time_precision = feature_extractor.chunk_length / max_source_positions
    time = 0
    for seq_idx, item in enumerate(sequences):
        sequence, stride = item
        if isinstance(sequence, list):
            sequence = np.array(sequence)
        chunk_len, stride_left, stride_right = stride
        sequence = sequence.squeeze(0)
        # get rid of the `forced_decoder_idx` that are use to parametrize the generation
        begin_idx = np.where(sequence == timestamp_begin)[0][0] if timestamp_begin in sequence else 0
        sequence = sequence[begin_idx:]

        timestamp_tokens = sequence >= timestamp_begin
        if seq_idx != 0 and sum(timestamp_tokens) > 0:
            consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
            last_timestamp = np.where(timestamp_tokens)[0][-1]
            consecutive = np.append(consecutive, last_timestamp) if last_timestamp not in consecutive else consecutive
            time -= stride_left + stride_right
            offset = int((time / feature_extractor.sampling_rate) / time_precision)
            overlap_time = int((stride_left / feature_extractor.sampling_rate) / time_precision)
            # relevant timestamps are in the overlapping part
            relevant_timestamp = np.where(sequence[consecutive] >= timestamp_begin + overlap_time)[0]
            if relevant_timestamp.shape[0] > 0:
                relevant_timestamp = (
                    consecutive[relevant_timestamp[0] - 1] if relevant_timestamp[0] > 0 else consecutive[0]
                )
                # if a big stride is used, we need to check some of the previous items for the best overlap
                best_match = 0
                sliced_sequence = []
                for idx, previous_sequence in enumerate(reversed(items)):
                    previous_tokens = previous_sequence[1:-1]
                    if previous_sequence[0] < (timestamp_begin + offset - overlap_time) and idx != 0:
                        break  # the previous sequence is too far in the past
                    if len(previous_tokens) > 0:
                        # find the longest common sequence between the overlapping parts
                        index_left, index_right, match_length = _fast_find_longest_common_sequence(
                            sequence[1:relevant_timestamp], previous_tokens
                        )
                        # don't do anything if only 1 token was matched
                        if match_length > 1 and match_length > best_match:
                            best_match = match_length
                            best_idx = idx
                            end_of_curr_sequence_idx = (
                                np.where(sequence[index_left + 1 :] >= timestamp_begin)[0][0] + 1
                            )
                            end_of_curr_sequence_idx = end_of_curr_sequence_idx + 1 + index_left
                            # if all the tokens are matched, suffix
                            if index_left == 0 and match_length == len(previous_tokens):
                                sliced_sequence = np.insert(
                                    sequence[index_left + 1 : end_of_curr_sequence_idx], 0, previous_sequence[0]
                                )
                                sliced_sequence[-1] = previous_sequence[-1]
                            # if part of the previous sequence is not taken
                            elif index_left >= 0:
                                sliced_sequence = sequence[index_left + 1 : end_of_curr_sequence_idx]
                                # let's insert the missing part of the previous sequence
                                previous_slice = (
                                    previous_sequence[: index_right + 1] if index_right > 0 else [previous_sequence[0]]
                                )
                                sliced_sequence = np.insert(sliced_sequence, 0, previous_slice)
                                sliced_sequence[-1] += offset

                if len(sliced_sequence) > 0:
                    items[len(items) - best_idx - 1] = sliced_sequence
                    items = items[: len(items) - best_idx]
                    sequence = sequence[end_of_curr_sequence_idx:]

        # sequence might have changed
        timestamp_tokens = sequence >= timestamp_begin
        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if sum(timestamp_tokens) > 0:
            last_timestamp = np.where(timestamp_tokens)[0][-1]
            consecutive = (
                np.append(consecutive, last_timestamp + 1) if last_timestamp not in consecutive else consecutive
            )

        if len(consecutive) > 0:
            last_slice = 0
            for current_slice in consecutive:
                actual_offset = items[-1][-1] if seq_idx != 0 or last_slice != 0 else sequence[0]
                sliced_tokens = sequence[last_slice:current_slice]
                duration = sliced_tokens[-1] - sliced_tokens[0]
                sliced_tokens[0] = actual_offset
                sliced_tokens[-1] = actual_offset + duration
                items.append(sliced_tokens)
                last_slice = current_slice

        time += chunk_len
    result = []
    for i in range(len(items)):
        result += items[i].tolist()
    return result
