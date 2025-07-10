# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    UdopProcessor,
    UdopTokenizer,
    UdopTokenizerFast,
)
from transformers.testing_utils import (
    require_pytesseract,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)
from transformers.utils import cached_property, is_pytesseract_available, is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


if is_pytesseract_available():
    from transformers import LayoutLMv3ImageProcessor


@require_pytesseract
@require_sentencepiece
@require_tokenizers
class UdopProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    tokenizer_class = UdopTokenizer
    rust_tokenizer_class = UdopTokenizerFast
    processor_class = UdopProcessor
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = LayoutLMv3ImageProcessor(
            do_resize=True,
            size=224,
            apply_ocr=True,
        )
        tokenizer = UdopTokenizer.from_pretrained("microsoft/udop-large")
        processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(cls.tmpdirname)

        cls.tokenizer_pretrained_name = "microsoft/udop-large"

        image_processor = cls.get_image_processor()
        tokenizer = cls.get_tokenizers()[0]
        processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, **kwargs) -> PreTrainedTokenizer:
        return cls.tokenizer_class.from_pretrained(cls.tokenizer_pretrained_name, **kwargs)

    @classmethod
    def get_image_processor(cls, **kwargs):
        return LayoutLMv3ImageProcessor.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def get_rust_tokenizer(cls, **kwargs) -> PreTrainedTokenizerFast:
        return cls.rust_tokenizer_class.from_pretrained(cls.tokenizer_pretrained_name, **kwargs)

    @classmethod
    def get_tokenizers(cls, **kwargs) -> list[PreTrainedTokenizerBase]:
        return [cls.get_tokenizer(**kwargs), cls.get_rust_tokenizer(**kwargs)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        image_processor = self.get_image_processor()
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as tmpdir:
                processor.save_pretrained(tmpdir)
                processor = UdopProcessor.from_pretrained(tmpdir)

            self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
            self.assertIsInstance(processor.tokenizer, (UdopTokenizer, UdopTokenizerFast))

            self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
            self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UdopProcessor(image_processor=self.get_image_processor(), tokenizer=self.get_tokenizer())
            processor.save_pretrained(tmpdir)

            # slow tokenizer
            tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
            image_processor_add_kwargs = self.get_image_processor(do_resize=False, size=30)

            processor = UdopProcessor.from_pretrained(
                tmpdir,
                use_fast=False,
                bos_token="(BOS)",
                eos_token="(EOS)",
                do_resize=False,
                size=30,
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, UdopTokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

        # fast tokenizer
        tokenizer_add_kwargs = self.get_rust_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_resize=False, size=30)

        processor = UdopProcessor.from_pretrained(
            self.tmpdirname, use_xlm=True, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, UdopTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, LayoutLMv3ImageProcessor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = UdopProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, text=input_str)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    def test_text_target(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = UdopProcessor(tokenizer=tokenizer, image_processor=image_processor)

        text = "hello world"
        expected_decoding = "hello world</s>"

        encoding_processor = processor(text_target=text)
        encoding_tokenizer = tokenizer(text_target=text)

        self.assertListEqual(encoding_processor["input_ids"], [21820, 296, 1])
        self.assertListEqual(encoding_processor["attention_mask"], [1, 1, 1])
        self.assertDictEqual(dict(encoding_processor), dict(encoding_tokenizer))
        self.assertEqual(tokenizer.decode(encoding_processor["input_ids"]), expected_decoding)

    @slow
    def test_overflowing_tokens(self):
        # In the case of overflowing tokens, test that we still have 1-to-1 mapping between the images and input_ids (sequences that are too long are broken down into multiple sequences).

        from datasets import load_dataset

        # set up
        datasets = load_dataset("nielsr/funsd")
        processor = UdopProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)

        def preprocess_data(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            words = examples["words"]
            boxes = examples["bboxes"]
            word_labels = examples["ner_tags"]
            encoded_inputs = processor(
                images,
                words,
                boxes=boxes,
                word_labels=word_labels,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
                stride=50,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            return encoded_inputs

        train_data = preprocess_data(datasets["train"])

        self.assertEqual(len(train_data["pixel_values"]), len(train_data["input_ids"]))


# different use cases tests
@require_sentencepiece
@require_torch
@require_pytesseract
class UdopProcessorIntegrationTests(unittest.TestCase):
    @cached_property
    def get_images(self):
        # we verify our implementation on 2 document images from the DocVQA dataset
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/fixtures_docvqa", split="test")
        return ds[0]["image"].convert("RGB"), ds[1]["image"].convert("RGB")

    @cached_property
    def get_tokenizers(self):
        slow_tokenizer = UdopTokenizer.from_pretrained("microsoft/udop-large")
        fast_tokenizer = UdopTokenizerFast.from_pretrained("microsoft/udop-large")
        return [slow_tokenizer, fast_tokenizer]

    @slow
    def test_processor_case_1(self):
        # case 1: document image classification (training, inference) + token classification (inference), apply_ocr = True

        image_processor = LayoutLMv3ImageProcessor()
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            input_image_processor = image_processor(images[0], return_tensors="pt")
            input_processor = processor(images[0], return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify pixel_values
            self.assertTrue(
                torch.allclose(input_image_processor["pixel_values"], input_processor["pixel_values"], atol=1e-2)
            )

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            # fmt: off
            expected_decoding = "11:14 to 11:39 a.m 11:39 to 11:44 a.m. 11:44 a.m. to 12:25 p.m. 12:25 to 12:58 p.m. 12:58 to 4:00 p.m. 2:00 to 5:00 p.m. Coffee Break Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. (Exhibits Open) TRRF GENERAL SESSION (PART |) Presiding: Lee A. Waller TRRF Vice President “Introductory Remarks” Lee A. Waller, TRRF Vice Presi- dent Individual Interviews with TRRF Public Board Members and Sci- entific Advisory Council Mem- bers Conducted by TRRF Treasurer Philip G. Kuehn to get answers which the public refrigerated warehousing industry is looking for. Plus questions from the floor. Dr. Emil M. Mrak, University of Cal- ifornia, Chairman, TRRF Board; Sam R. Cecil, University of Georgia College of Agriculture; Dr. Stanley Charm, Tufts University School of Medicine; Dr. Robert H. Cotton, ITT Continental Baking Company; Dr. Owen Fennema, University of Wis- consin; Dr. Robert E. Hardenburg, USDA. Questions and Answers Exhibits Open Capt. Jack Stoney Room TRRF Scientific Advisory Council Meeting Ballroom Foyer</s>"  # noqa: E231
            # fmt: on
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            input_image_processor = image_processor(images, return_tensors="pt")
            input_processor = processor(images, padding=True, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify pixel_values
            self.assertTrue(
                torch.allclose(input_image_processor["pixel_values"], input_processor["pixel_values"], atol=1e-2)
            )

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            # fmt: off
            expected_decoding = "7 ITC Limited REPORT AND ACCOUNTS 2013 ITC’s Brands: An Asset for the Nation The consumer needs and aspirations they fulfil, the benefit they generate for millions across ITC’s value chains, the future-ready capabilities that support them, and the value that they create for the country, have made ITC’s brands national assets, adding to India’s competitiveness. It is ITC’s aspiration to be the No 1 FMCG player in the country, driven by its new FMCG businesses. A recent Nielsen report has highlighted that ITC's new FMCG businesses are the fastest growing among the top consumer goods companies operating in India. ITC takes justifiable pride that, along with generating economic value, these celebrated Indian brands also drive the creation of larger societal capital through the virtuous cycle of sustainable and inclusive growth. DI WILLS * ; LOVE DELIGHTFULLY SOFT SKIN? aia Ans Source: https://www.industrydocuments.ucsf.edu/docs/snbx0223</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"  # noqa: E231
            # fmt: on
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

    @slow
    def test_processor_case_2(self):
        # case 2: document image classification (training, inference) + token classification (inference), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            words = ["hello", "world"]
            boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
            input_processor = processor(images[0], words, boxes=boxes, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = list(input_processor.keys())
            for key in expected_keys:
                self.assertIn(key, actual_keys)

            # verify input_ids
            expected_decoding = "hello world</s>"
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
            expected_decoding = "hello world</s><pad><pad><pad><pad>"
            decoding = processor.decode(input_processor.input_ids[0].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [
                [3, 2, 5, 1],
                [6, 7, 4, 2],
                [3, 9, 2, 4],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [1000, 1000, 1000, 1000],
            ]
            self.assertListEqual(input_processor.bbox[1].tolist(), expected_bbox)

    @slow
    def test_processor_case_3(self):
        # case 3: token classification (training), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

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
            expected_decoding = "weirdly world</s>"
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify labels
            expected_labels = [1, -100, 2, -100]
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
            expected_decoding = "my name is niels</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [
                [3, 2, 5, 1],
                [6, 7, 4, 2],
                [3, 9, 2, 4],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [1000, 1000, 1000, 1000],
            ]
            self.assertListEqual(input_processor.bbox[1].tolist(), expected_bbox)

            # verify labels
            expected_labels = [6, 3, 10, 2, -100, -100, -100]
            self.assertListEqual(input_processor.labels[1].tolist(), expected_labels)

    @slow
    def test_processor_case_4(self):
        # case 4: visual question answering (inference), apply_ocr=True

        image_processor = LayoutLMv3ImageProcessor()
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            input_processor = processor(images[0], question, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            # this was obtained with Tesseract 4.1.1
            # fmt: off
            expected_decoding = "What's his name?</s> 11:14 to 11:39 a.m 11:39 to 11:44 a.m. 11:44 a.m. to 12:25 p.m. 12:25 to 12:58 p.m. 12:58 to 4:00 p.m. 2:00 to 5:00 p.m. Coffee Break Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. (Exhibits Open) TRRF GENERAL SESSION (PART |) Presiding: Lee A. Waller TRRF Vice President “Introductory Remarks” Lee A. Waller, TRRF Vice Presi- dent Individual Interviews with TRRF Public Board Members and Sci- entific Advisory Council Mem- bers Conducted by TRRF Treasurer Philip G. Kuehn to get answers which the public refrigerated warehousing industry is looking for. Plus questions from the floor. Dr. Emil M. Mrak, University of Cal- ifornia, Chairman, TRRF Board; Sam R. Cecil, University of Georgia College of Agriculture; Dr. Stanley Charm, Tufts University School of Medicine; Dr. Robert H. Cotton, ITT Continental Baking Company; Dr. Owen Fennema, University of Wis- consin; Dr. Robert E. Hardenburg, USDA. Questions and Answers Exhibits Open Capt. Jack Stoney Room TRRF Scientific Advisory Council Meeting Ballroom Foyer</s>"  # noqa: E231
            # fmt: on
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
            expected_decoding = "what's the time</s> 7 ITC Limited REPORT AND ACCOUNTS 2013 I</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            # fmt: off
            expected_bbox = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [0, 45, 67, 80], [72, 56, 109, 67], [72, 56, 109, 67], [116, 56, 189, 67], [198, 59, 253, 66], [198, 59, 253, 66], [257, 59, 285, 66], [289, 59, 365, 66], [289, 59, 365, 66], [289, 59, 365, 66], [289, 59, 365, 66], [372, 59, 407, 66], [74, 136, 161, 158], [1000, 1000, 1000, 1000]]  # noqa: E231
            # fmt: on
            self.assertListEqual(input_processor.bbox[1].tolist(), expected_bbox)

    @slow
    def test_processor_case_5(self):
        # case 5: visual question answering (inference), apply_ocr=False

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizers = self.get_tokenizers
        images = self.get_images

        for tokenizer in tokenizers:
            processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

            # not batched
            question = "What's his name?"
            words = ["hello", "world"]
            boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
            input_processor = processor(images[0], question, text_pair=words, boxes=boxes, return_tensors="pt")

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "What's his name?</s> hello world</s>"
            decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # batched
            questions = ["How old is he?", "what's the time"]
            words = [["hello", "world"], ["my", "name", "is", "niels"]]
            boxes = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[3, 2, 5, 1], [6, 7, 4, 2], [3, 9, 2, 4], [1, 1, 2, 3]]]
            input_processor = processor(
                images, questions, text_pair=words, boxes=boxes, padding=True, return_tensors="pt"
            )

            # verify keys
            expected_keys = ["attention_mask", "bbox", "input_ids", "pixel_values"]
            actual_keys = sorted(input_processor.keys())
            self.assertListEqual(actual_keys, expected_keys)

            # verify input_ids
            expected_decoding = "How old is he?</s> hello world</s><pad><pad><pad>"
            decoding = processor.decode(input_processor.input_ids[0].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            expected_decoding = "what's the time</s> my name is niels</s>"
            decoding = processor.decode(input_processor.input_ids[1].tolist())
            self.assertSequenceEqual(decoding, expected_decoding)

            # verify bbox
            expected_bbox = [[3, 9, 2, 4], [1, 1, 2, 3], [1, 1, 2, 3], [1, 1, 2, 3], [1000, 1000, 1000, 1000]]
            self.assertListEqual(input_processor.bbox[1].tolist()[-5:], expected_bbox)
