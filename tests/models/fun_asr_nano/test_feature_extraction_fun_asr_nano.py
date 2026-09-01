# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

import itertools
import unittest

import numpy as np

from transformers import FunAsrNanoFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


class FunAsrNanoFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        num_mel_bins=8,
        num_frames_lfr=7,
        stride_lfr=6,
        padding_value=0.0,
        sampling_rate=16_000,
        return_attention_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.num_mel_bins = num_mel_bins
        self.num_frames_lfr = num_frames_lfr
        self.stride_lfr = stride_lfr
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        # The padded model input is the LFR-stacked frame, i.e. `num_frames_lfr` mel frames concatenated. The
        # common padding tests assert against this width, while the constructor's `feature_size` is the mel-bin count.
        self.feature_size = num_mel_bins * num_frames_lfr

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.num_mel_bins,
            "num_frames_lfr": self.num_frames_lfr,
            "stride_lfr": self.stride_lfr,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


@require_torch
@require_torchaudio
class FunAsrNanoFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = FunAsrNanoFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = FunAsrNanoFeatureExtractionTester(self)

    def test_lfr_stacking_widens_features_by_num_frames_lfr(self):
        feature_extractor = FunAsrNanoFeatureExtractor()
        audio = np.zeros(16000, dtype=np.float32)

        outputs = feature_extractor(audio, sampling_rate=16000, return_tensors="np")

        self.assertEqual(
            outputs["input_features"].shape[-1],
            feature_extractor.feature_size * feature_extractor.num_frames_lfr,
        )

    def test_integration(self):
        """Pin the Kaldi fbank + LFR front-end so a change to either is caught.

        Reference values produced on this branch with:
        >>> audio = np.sin(2 * np.pi * 440 * np.arange(16000, dtype=np.float32) / 16000)
        >>> FunAsrNanoFeatureExtractor()(audio, sampling_rate=16000, return_tensors="np")["input_features"]
        """
        # fmt: off
        EXPECTED_FIRST_FRAME_HEAD = np.array(
            [-7.515916, -6.506824, -6.296969, -6.987014, -7.677636, -7.169872, -5.736084, -4.967820]
        )
        # fmt: on
        audio = np.sin(2 * np.pi * 440 * np.arange(16000, dtype=np.float32) / 16000)

        outputs = FunAsrNanoFeatureExtractor()(audio, sampling_rate=16000, return_tensors="np")
        input_features = outputs["input_features"]

        self.assertEqual(input_features.shape, (1, 17, 560))
        np.testing.assert_allclose(input_features[0, 0, :8], EXPECTED_FIRST_FRAME_HEAD, atol=1e-4)
        self.assertEqual(int(outputs["input_features_mask"].sum()), 17)

    def test_input_features_mask_marks_the_unpadded_frames(self):
        feature_extractor = FunAsrNanoFeatureExtractor()
        audio = [np.ones(16000, dtype=np.float32), np.ones(8000, dtype=np.float32)]

        outputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

        self.assertEqual(set(outputs), {"input_features", "input_features_mask"})
        self.assertEqual(outputs["input_features_mask"].shape, outputs["input_features"].shape[:2])
        # The shorter clip is half as long, so it must have strictly fewer valid frames than the longer one.
        lengths = outputs["input_features_mask"].sum(-1)
        self.assertGreater(int(lengths[0]), int(lengths[1]))
        self.assertEqual(int(lengths[0]), outputs["input_features"].shape[1])

    def test_batched_valid_frames_match_individual_extraction(self):
        feature_extractor = FunAsrNanoFeatureExtractor()
        short_audio = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
        long_audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)

        single = feature_extractor(short_audio, sampling_rate=16000, return_tensors="pt")
        batch = feature_extractor([short_audio, long_audio], sampling_rate=16000, return_tensors="pt")
        valid_length = int(single["input_features_mask"].sum(-1)[0])

        torch.testing.assert_close(
            batch["input_features"][0, :valid_length],
            single["input_features"][0, :valid_length],
        )
        self.assertTrue(torch.all(batch["input_features_mask"][0, :valid_length] == 1))
        self.assertTrue(torch.all(batch["input_features_mask"][0, valid_length:] == 0))


if __name__ == "__main__":
    unittest.main()
