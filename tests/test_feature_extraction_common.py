# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import json
import os
import tempfile

import numpy as np

from transformers import BatchFeature


class FeatureExtractionMixin:

    # to overwrite at feature extractactor specific tests
    feat_extract_tester = None
    feat_extract_class = None

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_common_properties(self):
        feat_extract = self.feat_extract_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feat_extract, "feature_dim"))
        self.assertTrue(hasattr(feat_extract, "sampling_rate"))
        self.assertTrue(hasattr(feat_extract, "padding_value"))

    def test_feat_extract_to_json_string(self):
        feat_extract = self.feat_extract_class(**self.feat_extract_dict)
        obj = json.loads(feat_extract.to_json_string())
        for key, value in self.feat_extract_dict.items():
            self.assertEqual(obj[key], value)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feat_extract_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feat_extract_class.from_json_file(json_file_path)

        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feat_extract_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            feat_extract_first.save_pretrained(tmpdirname)
            feat_extract_second = self.feat_extract_class.from_pretrained(tmpdirname)

        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_init_without_params(self):
        feat_extract = self.feat_extract_class()
        self.assertIsNotNone(feat_extract)

    def test_padding(self):
        def _input_have_equal_length(input):
            length = len(input[0])
            for input_slice in input[1:]:
                if len(input_slice) != length:
                    return False
            return True

        def _input_are_equal(input_1, input_2):
            if len(input_1) != len(input_2):
                return False

            for input_slice_1, input_slice_2 in zip(input_1, input_2):
                if not np.allclose(np.asarray(input_slice_1), np.asarray(input_slice_2), atol=1e-3):
                    return False
            return True

        feat_extract = self.feat_extract_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: speech_inputs})

        input_1 = feat_extract.pad(processed_features)[input_name]
        input_2 = feat_extract.pad(processed_features, padding="longest")[input_name]
        input_3 = feat_extract.pad(processed_features, padding="max_length", max_length=self.feat_extract_tester.max_seq_length)

        self.assertFalse(_input_have_equal_length(input_1))
        self.assertTrue(_input_have_equal_length(input_2))
        self.assertTrue(_input_have_equal_length(input_3))
        self.assertTrue(_input_are_equal(input_2, input_3))
        self.assertTrue(len(input_1[0]) == self.feat_extract_tester.min_seq_length)
        self.assertTrue(len(input_1[1]) == self.feat_extract_tester.min_seq_length + self.feat_extract_tester.seq_length_diff)

        input_4 = feat_extract.pad(processed_features, padding="max_length")[input_name]
        input_5 = feat_extract.pad(processed_features, padding="max_length", max_length=self.feat_extract_tester.max_seq_length + self.feat_extract_tester.seq_length_diff)[input_name]

        self.assertTrue(_input_are_equal(input_1, input_4))
        self.assertTrue(input_5.shape, (self.feat_extract_tester.batch_size, self.feat_extract_tester.max_seq_length + self.feat_extract_tester.seq_length_diff))

        input_6 = feat_extract.pad(processed_features, pad_to_multiple_of=10)[input_name]
        input_7 = feat_extract.pad(processed_features, padding="longest", pad_to_multiple_of=10)[input_name]
        input_8 = feat_extract.pad(
            processed_features, padding="max_length", pad_to_multiple_of=10, max_length=self.feat_extract_tester.max_seq_length + self.feat_extract_tester.seq_length_diff
        )[input_name]

        self.assertTrue(_input_are_equal(input_1, input_6))
        self.assertTrue(input_7.shape, (3, 1500))
        self.assertTrue(input_8.shape, (3, 2500))

        return

        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_2[0])[self.feat_extract_tester.min_seq_length:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_2[1])[self.feat_extract_tester.min_seq_length + self.feat_extract_tester.seq_length_diff:])) < 1e-3)
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_5[0])[self.feat_extract_tester.min_seq_length:])) < 1e-3)
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_7[0])[800:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_7[1])[1000:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_7[2])[1200:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_8[0])[800:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_8[1])[1000:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_8[2])[1200:])) < 1e-3)

#    def test_attention_mask(self):
#        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
#
        # default case -> no attention_mask is returned
#        feat_extract = self.get_feat_extract()
#        processed = feat_extract(speech_inputs)
#        self.assertNotIn("attention_mask", processed)
#
        # wav2vec2-lv60 -> return attention_mask
#        feat_extract = self.get_feat_extract(return_attention_mask=True)
#        processed = feat_extract(speech_inputs, padding="longest")
#
#        self.assertIn("attention_mask", processed)
#        self.assertListEqual(list(processed.attention_mask.shape), list(processed[input_name].shape))
#        self.assertListEqual(processed.attention_mask.sum(-1).tolist(), [800, 1000, 1200])
