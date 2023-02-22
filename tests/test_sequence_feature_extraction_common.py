# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

from transformers import BatchFeature
from transformers.testing_utils import require_tf, require_torch

from .test_feature_extraction_common import FeatureExtractionSavingTestMixin


class SequenceFeatureExtractionTestMixin(FeatureExtractionSavingTestMixin):
    # to overwrite at feature extractactor specific tests
    feat_extract_tester = None
    feature_extraction_class = None

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_common_properties(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feat_extract, "feature_size"))
        self.assertTrue(hasattr(feat_extract, "sampling_rate"))
        self.assertTrue(hasattr(feat_extract, "padding_value"))

    def test_batch_feature(self):
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        self.assertTrue(all(len(x) == len(y) for x, y in zip(speech_inputs, processed_features[input_name])))

        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common(equal_length=True)
        processed_features = BatchFeature({input_name: speech_inputs}, tensor_type="np")

        batch_features_input = processed_features[input_name]

        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]

        self.assertTrue(
            batch_features_input.shape
            == (self.feat_extract_tester.batch_size, len(speech_inputs[0]), self.feat_extract_tester.feature_size)
        )

    @require_torch
    def test_batch_feature_pt(self):
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common(equal_length=True)
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs}, tensor_type="pt")

        batch_features_input = processed_features[input_name]

        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]

        self.assertTrue(
            batch_features_input.shape
            == (self.feat_extract_tester.batch_size, len(speech_inputs[0]), self.feat_extract_tester.feature_size)
        )

    @require_tf
    def test_batch_feature_tf(self):
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common(equal_length=True)
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs}, tensor_type="tf")

        batch_features_input = processed_features[input_name]

        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]

        self.assertTrue(
            batch_features_input.shape
            == (self.feat_extract_tester.batch_size, len(speech_inputs[0]), self.feat_extract_tester.feature_size)
        )

    def _check_padding(self, numpify=False):
        def _inputs_have_equal_length(input):
            length = len(input[0])
            for input_slice in input[1:]:
                if len(input_slice) != length:
                    return False
            return True

        def _inputs_are_equal(input_1, input_2):
            if len(input_1) != len(input_2):
                return False

            for input_slice_1, input_slice_2 in zip(input_1, input_2):
                if not np.allclose(np.asarray(input_slice_1), np.asarray(input_slice_2), atol=1e-3):
                    return False
            return True

        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common(numpify=numpify)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        pad_diff = self.feat_extract_tester.seq_length_diff
        pad_max_length = self.feat_extract_tester.max_seq_length + pad_diff
        pad_min_length = self.feat_extract_tester.min_seq_length
        batch_size = self.feat_extract_tester.batch_size
        feature_size = self.feat_extract_tester.feature_size

        # test padding for List[int] + numpy
        input_1 = feat_extract.pad(processed_features, padding=False)
        input_1 = input_1[input_name]

        input_2 = feat_extract.pad(processed_features, padding="longest")
        input_2 = input_2[input_name]

        input_3 = feat_extract.pad(processed_features, padding="max_length", max_length=len(speech_inputs[-1]))
        input_3 = input_3[input_name]

        input_4 = feat_extract.pad(processed_features, padding="longest", return_tensors="np")
        input_4 = input_4[input_name]

        # max_length parameter has to be provided when setting `padding="max_length"`
        with self.assertRaises(ValueError):
            feat_extract.pad(processed_features, padding="max_length")[input_name]

        input_5 = feat_extract.pad(
            processed_features, padding="max_length", max_length=pad_max_length, return_tensors="np"
        )
        input_5 = input_5[input_name]

        self.assertFalse(_inputs_have_equal_length(input_1))
        self.assertTrue(_inputs_have_equal_length(input_2))
        self.assertTrue(_inputs_have_equal_length(input_3))
        self.assertTrue(_inputs_are_equal(input_2, input_3))
        self.assertTrue(len(input_1[0]) == pad_min_length)
        self.assertTrue(len(input_1[1]) == pad_min_length + pad_diff)
        self.assertTrue(input_4.shape[:2] == (batch_size, len(input_3[0])))
        self.assertTrue(input_5.shape[:2] == (batch_size, pad_max_length))

        if feature_size > 1:
            self.assertTrue(input_4.shape[2] == input_5.shape[2] == feature_size)

        # test padding for `pad_to_multiple_of` for List[int] + numpy
        input_6 = feat_extract.pad(processed_features, pad_to_multiple_of=10)
        input_6 = input_6[input_name]

        input_7 = feat_extract.pad(processed_features, padding="longest", pad_to_multiple_of=10)
        input_7 = input_7[input_name]

        input_8 = feat_extract.pad(
            processed_features, padding="max_length", pad_to_multiple_of=10, max_length=pad_max_length
        )
        input_8 = input_8[input_name]

        input_9 = feat_extract.pad(
            processed_features,
            padding="max_length",
            pad_to_multiple_of=10,
            max_length=pad_max_length,
            return_tensors="np",
        )
        input_9 = input_9[input_name]

        self.assertTrue(all(len(x) % 10 == 0 for x in input_6))
        self.assertTrue(_inputs_are_equal(input_6, input_7))

        expected_mult_pad_length = pad_max_length if pad_max_length % 10 == 0 else (pad_max_length // 10 + 1) * 10
        self.assertTrue(all(len(x) == expected_mult_pad_length for x in input_8))
        self.assertEqual(input_9.shape[:2], (batch_size, expected_mult_pad_length))

        if feature_size > 1:
            self.assertTrue(input_9.shape[2] == feature_size)

        # Check padding value is correct
        padding_vector_sum = (np.ones(self.feat_extract_tester.feature_size) * feat_extract.padding_value).sum()
        self.assertTrue(
            abs(np.asarray(input_2[0])[pad_min_length:].sum() - padding_vector_sum * (pad_max_length - pad_min_length))
            < 1e-3
        )
        self.assertTrue(
            abs(
                np.asarray(input_2[1])[pad_min_length + pad_diff :].sum()
                - padding_vector_sum * (pad_max_length - pad_min_length - pad_diff)
            )
            < 1e-3
        )
        self.assertTrue(
            abs(
                np.asarray(input_2[2])[pad_min_length + 2 * pad_diff :].sum()
                - padding_vector_sum * (pad_max_length - pad_min_length - 2 * pad_diff)
            )
            < 1e-3
        )
        self.assertTrue(
            abs(input_5[0, pad_min_length:].sum() - padding_vector_sum * (pad_max_length - pad_min_length)) < 1e-3
        )
        self.assertTrue(
            abs(input_9[0, pad_min_length:].sum() - padding_vector_sum * (expected_mult_pad_length - pad_min_length))
            < 1e-3
        )

    def _check_truncation(self, numpify=False):
        def _inputs_have_equal_length(input):
            length = len(input[0])
            for input_slice in input[1:]:
                if len(input_slice) != length:
                    return False
            return True

        def _inputs_are_equal(input_1, input_2):
            if len(input_1) != len(input_2):
                return False

            for input_slice_1, input_slice_2 in zip(input_1, input_2):
                if not np.allclose(np.asarray(input_slice_1), np.asarray(input_slice_2), atol=1e-3):
                    return False
            return True

        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common(numpify=numpify)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        # truncate to smallest
        input_1 = feat_extract.pad(
            processed_features, padding="max_length", max_length=len(speech_inputs[0]), truncation=True
        )
        input_1 = input_1[input_name]

        input_2 = feat_extract.pad(processed_features, padding="max_length", max_length=len(speech_inputs[0]))
        input_2 = input_2[input_name]

        self.assertTrue(_inputs_have_equal_length(input_1))
        self.assertFalse(_inputs_have_equal_length(input_2))

        # truncate to smallest with np
        input_3 = feat_extract.pad(
            processed_features,
            padding="max_length",
            max_length=len(speech_inputs[0]),
            return_tensors="np",
            truncation=True,
        )
        input_3 = input_3[input_name]

        input_4 = feat_extract.pad(
            processed_features, padding="max_length", max_length=len(speech_inputs[0]), return_tensors="np"
        )
        input_4 = input_4[input_name]

        self.assertTrue(_inputs_have_equal_length(input_3))
        self.assertTrue(input_3.shape[1] == len(speech_inputs[0]))

        # since truncation forces padding to be smaller than longest input
        # function can't return `np.ndarray`, but has to return list
        self.assertFalse(_inputs_have_equal_length(input_4))

        # truncate to middle
        input_5 = feat_extract.pad(
            processed_features,
            padding="max_length",
            max_length=len(speech_inputs[1]),
            truncation=True,
            return_tensors="np",
        )
        input_5 = input_5[input_name]

        input_6 = feat_extract.pad(
            processed_features, padding="max_length", max_length=len(speech_inputs[1]), truncation=True
        )
        input_6 = input_6[input_name]

        input_7 = feat_extract.pad(
            processed_features, padding="max_length", max_length=len(speech_inputs[1]), return_tensors="np"
        )
        input_7 = input_7[input_name]

        self.assertTrue(input_5.shape[1] == len(speech_inputs[1]))
        self.assertTrue(_inputs_have_equal_length(input_5))
        self.assertTrue(_inputs_have_equal_length(input_6))
        self.assertTrue(_inputs_are_equal(input_5, input_6))

        # since truncation forces padding to be smaller than longest input
        # function can't return `np.ndarray`, but has to return list
        self.assertFalse(_inputs_have_equal_length(input_7))
        self.assertTrue(len(input_7[-1]) == len(speech_inputs[-1]))

        # padding has to be max_length when setting `truncation=True`
        with self.assertRaises(ValueError):
            feat_extract.pad(processed_features, truncation=True)[input_name]

        # padding has to be max_length when setting `truncation=True`
        with self.assertRaises(ValueError):
            feat_extract.pad(processed_features, padding="longest", truncation=True)[input_name]

        # padding has to be max_length when setting `truncation=True`
        with self.assertRaises(ValueError):
            feat_extract.pad(processed_features, padding="longest", truncation=True)[input_name]

        # max_length parameter has to be provided when setting `truncation=True` and padding="max_length"
        with self.assertRaises(ValueError):
            feat_extract.pad(processed_features, padding="max_length", truncation=True)[input_name]

        # test truncation for `pad_to_multiple_of` for List[int] + numpy
        pad_to_multiple_of = 12
        input_8 = feat_extract.pad(
            processed_features,
            padding="max_length",
            max_length=len(speech_inputs[0]),
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=True,
        )
        input_8 = input_8[input_name]

        input_9 = feat_extract.pad(
            processed_features,
            padding="max_length",
            max_length=len(speech_inputs[0]),
            pad_to_multiple_of=pad_to_multiple_of,
        )
        input_9 = input_9[input_name]

        # retrieve expected_length as multiple of pad_to_multiple_of
        expected_length = len(speech_inputs[0])
        if expected_length % pad_to_multiple_of != 0:
            expected_length = ((len(speech_inputs[0]) // pad_to_multiple_of) + 1) * pad_to_multiple_of

        self.assertTrue(len(input_8[0]) == expected_length)
        self.assertTrue(_inputs_have_equal_length(input_8))
        self.assertFalse(_inputs_have_equal_length(input_9))

    def test_padding_from_list(self):
        self._check_padding(numpify=False)

    def test_padding_from_array(self):
        self._check_padding(numpify=True)

    def test_truncation_from_list(self):
        self._check_truncation(numpify=False)

    def test_truncation_from_array(self):
        self._check_truncation(numpify=True)

    @require_torch
    def test_padding_accepts_tensors_pt(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        input_np = feat_extract.pad(processed_features, padding="longest", return_tensors="np")[input_name]
        input_pt = feat_extract.pad(processed_features, padding="longest", return_tensors="pt")[input_name]

        self.assertTrue(abs(input_np.astype(np.float32).sum() - input_pt.numpy().astype(np.float32).sum()) < 1e-2)

    @require_tf
    def test_padding_accepts_tensors_tf(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        input_np = feat_extract.pad(processed_features, padding="longest", return_tensors="np")[input_name]
        input_tf = feat_extract.pad(processed_features, padding="longest", return_tensors="tf")[input_name]

        self.assertTrue(abs(input_np.astype(np.float32).sum() - input_tf.numpy().astype(np.float32).sum()) < 1e-2)

    def test_attention_mask(self):
        feat_dict = self.feat_extract_dict
        feat_dict["return_attention_mask"] = True
        feat_extract = self.feature_extraction_class(**feat_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        input_lenghts = [len(x) for x in speech_inputs]
        input_name = feat_extract.model_input_names[0]

        processed = BatchFeature({input_name: speech_inputs})

        processed = feat_extract.pad(processed, padding="longest", return_tensors="np")
        self.assertIn("attention_mask", processed)
        self.assertListEqual(list(processed.attention_mask.shape), list(processed[input_name].shape[:2]))
        self.assertListEqual(processed.attention_mask.sum(-1).tolist(), input_lenghts)

    def test_attention_mask_with_truncation(self):
        feat_dict = self.feat_extract_dict
        feat_dict["return_attention_mask"] = True
        feat_extract = self.feature_extraction_class(**feat_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        input_lenghts = [len(x) for x in speech_inputs]
        input_name = feat_extract.model_input_names[0]

        processed = BatchFeature({input_name: speech_inputs})
        max_length = min(input_lenghts)

        processed_pad = feat_extract.pad(
            processed, padding="max_length", max_length=max_length, truncation=True, return_tensors="np"
        )
        self.assertIn("attention_mask", processed_pad)
        self.assertListEqual(
            list(processed_pad.attention_mask.shape), [processed_pad[input_name].shape[0], max_length]
        )
        self.assertListEqual(
            processed_pad.attention_mask[:, :max_length].sum(-1).tolist(), [max_length for x in speech_inputs]
        )
