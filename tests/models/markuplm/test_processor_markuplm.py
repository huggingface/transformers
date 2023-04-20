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

from transformers import (
    MarkupLMProcessor,
    MarkupLMTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.models.markuplm.tokenization_markuplm import VOCAB_FILES_NAMES
from transformers.testing_utils import require_bs4, require_tokenizers, require_torch, slow
from transformers.utils import FEATURE_EXTRACTOR_NAME, cached_property, is_bs4_available, is_tokenizers_available


if is_bs4_available():
    from transformers import MarkupLMFeatureExtractor

if is_tokenizers_available():
    from transformers import MarkupLMTokenizerFast


@require_bs4
@require_tokenizers
class MarkupLMProcessorTest(unittest.TestCase):
    tokenizer_class = MarkupLMTokenizer
    rust_tokenizer_class = MarkupLMTokenizerFast

    def setUp(self):
        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        # fmt: off
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "\u0120", "\u0120l", "\u0120n", "\u0120lo", "\u0120low", "er", "\u0120lowest", "\u0120newer", "\u0120wider", "\u0120hello", "\u0120world", "<unk>",]  # noqa
        # fmt: on
        self.tmpdirname = tempfile.mkdtemp()
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.tags_dict = {"a": 0, "abbr": 1, "acronym": 2, "address": 3}
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        self.tokenizer_config_file = os.path.join(self.tmpdirname, "tokenizer_config.json")

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))
        with open(self.tokenizer_config_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps({"tags_dict": self.tags_dict}))

        feature_extractor_map = {"feature_extractor_type": "MarkupLMFeatureExtractor"}
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

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MarkupLMProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            tokenizer.model_input_names,
            msg="`processor` and `tokenizer` model input names do not match",
        )


# different use cases tests
@require_bs4
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
        fast_tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base", from_slow=True)
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
            inputs = processor(html_strings[0], return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected = [0, 31414, 232, 25194, 11773, 16, 127, 998, 4, 2]
            self.assertSequenceEqual(inputs.input_ids.squeeze().tolist(), expected)

            # batched
            inputs = processor(html_strings, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected = [0, 48085, 2209, 48085, 3156, 32, 6533, 19, 5, 48599, 6694, 35, 2]
            self.assertSequenceEqual(inputs.input_ids[1].tolist(), expected)

    @slow
    def test_processor_case_2(self):
        # case 2: web page classification (training, inference) + token classification (inference), parse_html=False

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.parse_html = False

            # not batched
            nodes = ["hello", "world", "how", "are"]
            xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
            inputs = processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = list(inputs.keys())
            for key in expected_keys:
                self.assertIn(key, actual_keys)

            # verify input_ids
            expected_decoding = "<s>helloworldhoware</s>"
            decoding = processor.decode(inputs.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            nodes = [["hello", "world"], ["my", "name", "is"]]
            xpaths = [
                ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
                ["html/body", "html/body/div", "html/body"],
            ]
            inputs = processor(nodes=nodes, xpaths=xpaths, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s>helloworld</s><pad>"
            decoding = processor.decode(inputs.input_ids[0].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

    @slow
    def test_processor_case_3(self):
        # case 3: token classification (training), parse_html=False

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.parse_html = False

            # not batched
            nodes = ["hello", "world", "how", "are"]
            xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
            node_labels = [1, 2, 2, 1]
            inputs = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

            # verify keys
            expected_keys = [
                "attention_mask",
                "input_ids",
                "labels",
                "token_type_ids",
                "xpath_subs_seq",
                "xpath_tags_seq",
            ]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_ids = [0, 42891, 8331, 9178, 1322, 2]
            self.assertSequenceEqual(inputs.input_ids[0].tolist(), expected_ids)

            # verify labels
            expected_labels = [-100, 1, 2, 2, 1, -100]
            self.assertListEqual(inputs.labels.squeeze().tolist(), expected_labels)

            # batched
            nodes = [["hello", "world"], ["my", "name", "is"]]
            xpaths = [
                ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
                ["html/body", "html/body/div", "html/body"],
            ]
            node_labels = [[1, 2], [6, 3, 10]]
            inputs = processor(
                nodes=nodes,
                xpaths=xpaths,
                node_labels=node_labels,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt",
            )

            # verify keys
            expected_keys = [
                "attention_mask",
                "input_ids",
                "labels",
                "token_type_ids",
                "xpath_subs_seq",
                "xpath_tags_seq",
            ]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_ids = [0, 4783, 13650, 354, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            self.assertSequenceEqual(inputs.input_ids[1].tolist(), expected_ids)

            # verify xpath_tags_seq
            # fmt: off
            expected_xpaths_tags_seq = [[216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [109, 25, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [109, 25, 50, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [109, 25, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216]]  # noqa: 
            # fmt: on
            self.assertSequenceEqual(inputs.xpath_tags_seq[1].tolist(), expected_xpaths_tags_seq)

            # verify labels
            # fmt: off
            expected_labels = [-100, 6, 3, 10, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
            # fmt: on
            self.assertListEqual(inputs.labels[1].tolist(), expected_labels)

    @slow
    def test_processor_case_4(self):
        # case 4: question answering (inference), parse_html=True

        feature_extractor = MarkupLMFeatureExtractor()
        tokenizers = self.get_tokenizers
        html_strings = self.get_html_strings

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            inputs = processor(html_strings[0], questions=question, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            # fmt: off
            expected_decoding = "<s>What's his name?</s>Hello worldWelcomeHere is my website.</s>"  # noqa: E231
            # fmt: on
            decoding = processor.decode(inputs.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            inputs = processor(
                html_strings,
                questions=questions,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt",
            )

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = (
                "<s>what's the time</s>HTML ImagesHTML images are defined with the img tag:</s><pad><pad>"
            )
            decoding = processor.decode(inputs.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify xpath_subs_seq
            # fmt: off
            expected_xpath_subs_seq = [[1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001]]
            # fmt: on
            self.assertListEqual(inputs.xpath_subs_seq[1].tolist(), expected_xpath_subs_seq)

    @slow
    def test_processor_case_5(self):
        # case 5: question answering (inference), parse_html=False

        feature_extractor = MarkupLMFeatureExtractor(parse_html=False)
        tokenizers = self.get_tokenizers

        for tokenizer in tokenizers:
            processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.parse_html = False

            # not batched
            question = "What's his name?"
            nodes = ["hello", "world", "how", "are"]
            xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
            inputs = processor(nodes=nodes, xpaths=xpaths, questions=question, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s>What's his name?</s>helloworldhoware</s>"
            decoding = processor.decode(inputs.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            nodes = [["hello", "world"], ["my", "name", "is"]]
            xpaths = [
                ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
                ["html/body", "html/body/div", "html/body"],
            ]
            inputs = processor(nodes=nodes, xpaths=xpaths, questions=questions, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "input_ids", "token_type_ids", "xpath_subs_seq", "xpath_tags_seq"]
            actual_keys = sorted(inputs.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s>How old is he?</s>helloworld</s>"
            decoding = processor.decode(inputs.input_ids[0].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            expected_decoding = "<s>what's the time</s>mynameis</s>"
            decoding = processor.decode(inputs.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify xpath_subs_seq
            # fmt: off
            expected_xpath_subs_seq = [[1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [0, 0, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001], [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001]]
            # fmt: on
            self.assertListEqual(inputs.xpath_subs_seq[1].tolist()[-5:], expected_xpath_subs_seq)
