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

import numpy as np

from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.models.layoutlmv3 import LayoutLMv3Tokenizer, LayoutLMv3TokenizerFast
from transformers.models.layoutlmv3.tokenization_layoutlmv3 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_pytesseract, require_tokenizers, require_torch, slow
from transformers.utils import FEATURE_EXTRACTOR_NAME, cached_property, is_pytesseract_available


if is_pytesseract_available():
    from PIL import Image

    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Processor


@require_pytesseract
@require_tokenizers
class LayoutLMv3ProcessorTest(unittest.TestCase):
    tokenizer_class = LayoutLMv3Tokenizer
    rust_tokenizer_class = LayoutLMv3TokenizerFast

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

        image_processor_map = {
            "do_resize": True,
            "size": 224,
            "apply_ocr": True,
        }

        self.feature_extraction_file = os.path.join(self.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(self.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(image_processor_map) + "\n")

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs) -> PreTrainedTokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_tokenizers(self, **kwargs) -> List[PreTrainedTokenizerBase]:
        return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]

    def get_image_processor(self, **kwargs):
        return LayoutLMv3ImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_default(self):
        image_processor = self.get_image_processor()
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            processor.save_pretrained(self.tmpdirname)
            processor = LayoutLMv3Processor.from_pretrained(self.tmpdirname)

            self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
            self.assertIsInstance(processor.tokenizer, (LayoutLMv3Tokenizer, LayoutLMv3TokenizerFast))

            self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
            self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        processor = LayoutLMv3Processor(image_processor=self.get_image_processor(), tokenizer=self.get_tokenizer())
        processor.save_pretrained(self.tmpdirname)

        # slow tokenizer
        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_resize=False, size=30)

        processor = LayoutLMv3Processor.from_pretrained(
            self.tmpdirname, use_fast=False, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, LayoutLMv3Tokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

        # fast tokenizer
        tokenizer_add_kwargs = self.get_rust_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_resize=False, size=30)

        processor = LayoutLMv3Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, LayoutLMv3TokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # add extra args
        inputs = processor(text=input_str, images=image_input, return_codebook_pixels=False, return_image_mask=False)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)


# different use cases tests
@require_torch
@require_pytesseract
class LayoutLMv3ProcessorIntegrationTests(unittest.TestCase):
    @cached_property
    def get_images(self):
        # we verify our implementation on 2 document images from the DocVQA dataset
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/fixtures_docvqa", split="test")

        image_1 = Image.open(ds[0]["file"]).convert("RGB")
        image_2 = Image.open(ds[1]["file"]).convert("RGB")

        return image_1, image_2

    @cached_property
    def get_tokenizers(self):
        slow_tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base", add_visual_labels=False)
        fast_tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base", add_visual_labels=False)
        return [slow_tokenizer, fast_tokenizer]

    @slow
    def test_processor_case_1(self):
        # case 1: document image classification (training, inference) + token classification (inference), apply_ocr = True

        image_processor = LayoutLMv3ImageProcessor()
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            input_image_proc = image_processor(images[0], return_tensors="pt")
            input_processor = processor(images[0], return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify image
            self.assertAlmostEqual(
                input_image_proc["pixel_values"].sum(), input_processor["pixel_values"].sum(), delta=1e-2
            )

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            expected_decoding = "<s> 11:14 to 11:39 a.m 11:39 to 11:44 a.m. 11:44 a.m. to 12:25 p.m. 12:25 to 12:58 p.m. 12:58 to 4:00 p.m. 2:00 to 5:00 p.m. Coffee Break Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. (Exhibits Open) TRRF GENERAL SESSION (PART |) Presiding: Lee A. Waller TRRF Vice President “Introductory Remarks” Lee A. Waller, TRRF Vice Presi- dent Individual Interviews with TRRF Public Board Members and Sci- entific Advisory Council Mem- bers Conducted by TRRF Treasurer Philip G. Kuehn to get answers which the public refrigerated warehousing industry is looking for. Plus questions from the floor. Dr. Emil M. Mrak, University of Cal- ifornia, Chairman, TRRF Board; Sam R. Cecil, University of Georgia College of Agriculture; Dr. Stanley Charm, Tufts University School of Medicine; Dr. Robert H. Cotton, ITT Continental Baking Company; Dr. Owen Fennema, University of Wis- consin; Dr. Robert E. Hardenburg, USDA. Questions and Answers Exhibits Open Capt. Jack Stoney Room TRRF Scientific Advisory Council Meeting Ballroom Foyer</s>"  # fmt: skip
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            input_image_proc = image_processor(images, return_tensors="pt")
            input_processor = processor(images, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify images
            self.assertAlmostEqual(
                input_image_proc["pixel_values"].sum(), input_processor["pixel_values"].sum(), delta=1e-2
            )

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            expected_decoding = "<s> 7 ITC Limited REPORT AND ACCOUNTS 2013 ITC’s Brands: An Asset for the Nation The consumer needs and aspirations they fulfil, the benefit they generate for millions across ITC’s value chains, the future-ready capabilities that support them, and the value that they create for the country, have made ITC’s brands national assets, adding to India’s competitiveness. It is ITC’s aspiration to be the No 1 FMCG player in the country, driven by its new FMCG businesses. A recent Nielsen report has highlighted that ITC's new FMCG businesses are the fastest growing among the top consumer goods companies operating in India. ITC takes justifiable pride that, along with generating economic value, these celebrated Indian brands also drive the creation of larger societal capital through the virtuous cycle of sustainable and inclusive growth. DI WILLS * ; LOVE DELIGHTFULLY SOFT SKIN? aia Ans Source: https://www.industrydocuments.ucsf.edu/docs/snbx0223</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"  # fmt: skip
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

    @slow
    def test_processor_case_2(self):
        # case 2: document image classification (training, inference) + token classification (inference), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            words = ["hello", "world"]
            boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
            input_processor = processor(images[0], words, boxes=boxes, return_tensors="pt")

            # verify keys
            expected_keys = ["input_ids", "bbox", "attention_mask", "pixel_values"]
            actual_keys = list(input_processor.keys())
            for key in expected_keys:
                self.assertIn(key, actual_keys)

            # verify input_ids
            expected_decoding = "<s> hello world</s>"
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            words = [["hello", "world"], ["my", "name", "is", "niels"]]
            boxes = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[3, 2, 5, 1], [6, 7, 4, 2], [3, 9, 2, 4], [1, 1, 2, 3]]]
            input_processor = processor(images, words, boxes=boxes, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s> hello world</s><pad><pad><pad>"
            decoding = processor.decode(input_processor.input_ids[0].tolist())
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

    @slow
    def test_processor_case_3(self):
        # case 3: token classification (training), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            words = ["weirdly", "world"]
            boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
            word_labels = [1, 2]
            input_processor = processor(images[0], words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "labels", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
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
                images, words, boxes=boxes, word_labels=word_labels, padding=True, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "labels", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
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
    def test_processor_case_4(self):
        # case 4: visual question answering (inference), apply_ocr=True

        image_processor = LayoutLMv3ImageProcessor()
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            input_processor = processor(images[0], question, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            expected_decoding = "<s> What's his name?</s></s> 11:14 to 11:39 a.m 11:39 to 11:44 a.m. 11:44 a.m. to 12:25 p.m. 12:25 to 12:58 p.m. 12:58 to 4:00 p.m. 2:00 to 5:00 p.m. Coffee Break Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. (Exhibits Open) TRRF GENERAL SESSION (PART |) Presiding: Lee A. Waller TRRF Vice President “Introductory Remarks” Lee A. Waller, TRRF Vice Presi- dent Individual Interviews with TRRF Public Board Members and Sci- entific Advisory Council Mem- bers Conducted by TRRF Treasurer Philip G. Kuehn to get answers which the public refrigerated warehousing industry is looking for. Plus questions from the floor. Dr. Emil M. Mrak, University of Cal- ifornia, Chairman, TRRF Board; Sam R. Cecil, University of Georgia College of Agriculture; Dr. Stanley Charm, Tufts University School of Medicine; Dr. Robert H. Cotton, ITT Continental Baking Company; Dr. Owen Fennema, University of Wis- consin; Dr. Robert E. Hardenburg, USDA. Questions and Answers Exhibits Open Capt. Jack Stoney Room TRRF Scientific Advisory Council Meeting Ballroom Foyer</s>"  # fmt: skip
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            input_processor = processor(
                images, questions, padding="max_length", max_length=20, truncation=True, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            expected_decoding = "<s> what's the time</s></s> 7 ITC Limited REPORT AND ACCOUNTS 2013 ITC</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 45, 67, 80], [72, 56, 109, 67], [72, 56, 109, 67], [116, 56, 189, 67], [198, 59, 253, 66], [257, 59, 285, 66], [289, 59, 365, 66], [289, 59, 365, 66], [289, 59, 365, 66], [372, 59, 407, 66], [74, 136, 161, 158], [74, 136, 161, 158], [0, 0, 0, 0]]  # fmt: skip
            self.assertListEqual(input_processor.bbox[1].tolist(), expected_bbox)

    @slow
    def test_processor_case_5(self):
        # case 5: visual question answering (inference), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            words = ["hello", "world"]
            boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
            input_processor = processor(images[0], question, words, boxes, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s> What's his name?</s></s> hello world</s>"
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            words = [["hello", "world"], ["my", "name", "is", "niels"]]
            boxes = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[3, 2, 5, 1], [6, 7, 4, 2], [3, 9, 2, 4], [1, 1, 2, 3]]]
            input_processor = processor(images, questions, words, boxes, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "<s> How old is he?</s></s> hello world</s><pad><pad>"
            decoding = processor.decode(input_processor.input_ids[0].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            expected_decoding = "<s> what's the time</s></s> my name is niels</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [[6, 7, 4, 2], [3, 9, 2, 4], [1, 1, 2, 3], [1, 1, 2, 3], [0, 0, 0, 0]]
            self.assertListEqual(input_processor.bbox[1].tolist()[-5:], expected_bbox)
