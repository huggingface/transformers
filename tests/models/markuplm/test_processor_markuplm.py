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

import json
import os
import shutil
import tempfile
import unittest
from typing import List

# TODO add dependency check
from transformers import (
    MarkupLMFeatureExtractor,
    MarkupLMProcessor,
    MarkupLMTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    is_tokenizers_available,
)
from transformers.models.markuplm.tokenization_markuplm import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, require_torch, slow
from transformers.utils import FEATURE_EXTRACTOR_NAME, cached_property


if is_tokenizers_available():
    from transformers import MarkupLMTokenizerFast


@require_tokenizers
class MarkupLMProcessorTest(unittest.TestCase):
    tokenizer_class = MarkupLMTokenizer
    rust_tokenizer_class = MarkupLMTokenizerFast

    def setUp(self):
        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        self.tmpdirname = tempfile.mkdtemp()
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

        feature_extractor_map = {
            "do_resize": True,
            "size": 224,
            "apply_ocr": True,
        }

        self.feature_extraction_file = os.path.join(self.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(self.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(feature_extractor_map) + "\n")

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs) -> PreTrainedTokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_tokenizers(self, **kwargs) -> List[PreTrainedTokenizerBase]:
        return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]

    def get_feature_extractor(self, **kwargs):
        return MarkupLMFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        feature_extractor = self.get_feature_extractor()
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            processor.save_pretrained(self.tmpdirname)
            processor = MarkupLMProcessor.from_pretrained(self.tmpdirname)

            self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
            self.assertIsInstance(processor.tokenizer, (MarkupLMTokenizer, MarkupLMTokenizerFast))

            self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
            self.assertIsInstance(processor.feature_extractor, MarkupLMFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = MarkupLMProcessor(feature_extractor=self.get_feature_extractor(), tokenizer=self.get_tokenizer())
        processor.save_pretrained(self.tmpdirname)

        # slow tokenizer
        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_resize=False, size=30)

        processor = MarkupLMProcessor.from_pretrained(
            self.tmpdirname, use_fast=False, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, MarkupLMTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, MarkupLMFeatureExtractor)

        # fast tokenizer
        tokenizer_add_kwargs = self.get_rust_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_resize=False, size=30)

        processor = MarkupLMProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, MarkupLMTokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, MarkupLMFeatureExtractor)


# different use cases tests
@require_torch
class MarkupLMProcessorIntegrationTests(unittest.TestCase):
    @cached_property
    def get_html_strings(self):
        html_string_1 = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Hello world</title>
        </head>
        <body>

        <h1>Welcome</h1>
        <p>Here is my website.</p>

        </body>
        </html>"""

        html_string_2 = """
        <!DOCTYPE html>
        <html>
        <body>

        <h2>HTML Images</h2>
        <p>HTML images are defined with the img tag:</p>

        <img src="w3schools.jpg" alt="W3Schools.com" width="104" height="142">

        </body>
        </html>
        """

        return [html_string_1, html_string_2]

    @cached_property
    def get_tokenizers(self):
        slow_tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")
        fast_tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")
        return [slow_tokenizer, fast_tokenizer]

    @slow
    def test_processor_case_1(self):
        # case 1: web page classification (training, inference) + token classification (inference)

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers
        html_strings = self.get_html_strings

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # not batched
            input_feat_extract = feature_extractor(html_strings[0])
            input_processor = processor(html_strings[0], return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected = [0, 31414, 232, 25194, 11773, 16, 127, 998, 4, 2]
            self.assertSequenceEqual(input_processor.input_ids.squeeze().tolist(), expected)

            # batched
            input_feat_extract = feature_extractor(html_strings)
            input_processor = processor(html_strings, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected = [0, 48085, 2209, 48085, 3156, 32, 6533, 19, 5, 48599, 6694, 35, 2]
            self.assertSequenceEqual(input_processor.input_ids[1].tolist(), expected)

    @slow
    def test_processor_case_2(self):
        # case 2: token classification (training)

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers
        html_strings = self.get_html_strings

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # not batched
            nodes = ["weirdly", "world"]
            xpaths = [[1, 2, 3, 4], [5, 6, 7, 8]]
            node_labels = [1, 2]
            input_processor = processor(
                html_strings[0], words, boxes=boxes, word_labels=word_labels, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "labels", "pixel_values"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s> weirdly world</s>"
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify labels
            expected_labels = [-100, 1, -100, 2, -100]
            self.assertListEqual(input_processor.labels.squeeze().tolist(), expected_labels)

            # batched
            words = [["hello", "world"], ["my", "name", "is", "niels"]]
            boxes = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[3, 2, 5, 1], [6, 7, 4, 2], [3, 9, 2, 4], [1, 1, 2, 3]]]
            word_labels = [[1, 2], [6, 3, 10, 2]]
            input_processor = processor(
                html_strings, words, boxes=boxes, word_labels=word_labels, padding=True, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "labels", "pixel_values"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s> my name is niels</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [
                [0, 0, 0, 0],
                [3, 2, 5, 1],
                [6, 7, 4, 2],
                [3, 9, 2, 4],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [0, 0, 0, 0],
            ]
            self.assertListEqual(input_processor.bbox[1].tolist(), expected_bbox)

            # verify labels
            expected_labels = [-100, 6, 3, 10, 2, -100, -100]
            self.assertListEqual(input_processor.labels[1].tolist(), expected_labels)

    @slow
    def test_processor_case_3(self):
        # case 3: visual question answering (inference)

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers
        html_strings = self.get_html_strings

        for tokenizer in tokenizers:
            print("Tokenizer:", tokenizer)
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            input_processor = processor(html_strings[0], question, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # # verify input_ids
            # # fmt: off
            # expected_decoding = "<s>What's his name?</s>sample documentGoogThis is one headerThis is a another HeaderTravel fromSFO to JFKon May 2, 2015 at 2:00 pm. For details go to confirm.comTravelernameisJohn Doe</s>"  # noqa: E231
            # # fmt: on
            # decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            # print("Actual decoding:", decoding)
            # self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            input_processor = processor(
                html_strings, questions, padding="max_length", max_length=20, truncation=True, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(list(input_processor.keys()))
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = (
                "<s>what's the time</s>My First HeadingMy first paragraph.</s><pad><pad><pad><pad><pad>"
            )
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            print("Actual decoding:", decoding)
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify xpath_subs_seq
            # fmt: off
            expected_xpath_subs_seq = [[1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 98, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 98, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 98, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 98, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 148, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 148, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 148, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [109, 25, 148, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001]]  # noqa: E231
            # fmt: on
            self.assertListEqual(input_processor.xpath_subs_seq[1].tolist(), expected_xpath_subs_seq)
