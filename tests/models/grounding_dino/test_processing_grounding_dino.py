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

import json
import os
import shutil
import tempfile
import unittest
from typing import Optional

import pytest

from transformers import BertTokenizer, BertTokenizerFast, GroundingDinoProcessor
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import IMAGE_PROCESSOR_NAME, is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput

if is_vision_available():
    from transformers import GroundingDinoImageProcessor


@require_torch
@require_vision
class GroundingDinoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "IDEA-Research/grounding-dino-base"
    processor_class = GroundingDinoProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]","want","##want","##ed","wa","un","runn","##ing",",","low","lowest"]  # fmt: skip
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        image_processor_map = {
            "do_resize": True,
            "size": None,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_pad": True,
        }
        cls.image_processor_file = os.path.join(cls.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(cls.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

        image_processor = GroundingDinoImageProcessor()
        tokenizer = BertTokenizer.from_pretrained(cls.from_pretrained_id)

        processor = GroundingDinoProcessor(image_processor, tokenizer)

        processor.save_pretrained(cls.tmpdirname)

        cls.batch_size = 7
        cls.num_queries = 5
        cls.embed_dim = 5
        cls.seq_length = 5

    def prepare_text_inputs(self, batch_size: Optional[int] = None, modality: Optional[str] = None):
        labels = ["a cat", "remote control"]
        labels_longer = ["a person", "a car", "a dog", "a cat"]

        if batch_size is None:
            return labels

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return [labels]
        return [labels, labels_longer] + [labels] * (batch_size - 2)

    @classmethod
    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.get_tokenizer with CLIP->Bert
    def get_tokenizer(cls, **kwargs):
        return BertTokenizer.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.get_rust_tokenizer with CLIP->Bert
    def get_rust_tokenizer(cls, **kwargs):
        return BertTokenizerFast.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.get_image_processor with CLIP->GroundingDino
    def get_image_processor(cls, **kwargs):
        return GroundingDinoImageProcessor.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_fake_grounding_dino_output(self):
        torch.manual_seed(42)
        return GroundingDinoObjectDetectionOutput(
            pred_boxes=torch.rand(self.batch_size, self.num_queries, 4),
            logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
            input_ids=self.get_fake_grounding_dino_input_ids(),
        )

    def get_fake_grounding_dino_input_ids(self):
        input_ids = torch.tensor([101, 1037, 4937, 1012, 102])
        return torch.stack([input_ids] * self.batch_size, dim=0)

    def test_post_process_grounded_object_detection(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        grounding_dino_output = self.get_fake_grounding_dino_output()

        post_processed = processor.post_process_grounded_object_detection(grounding_dino_output)

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["scores", "boxes", "text_labels", "labels"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))

        expected_scores = torch.tensor([0.7050, 0.7222, 0.7222, 0.6829, 0.7220])
        torch.testing.assert_close(post_processed[0]["scores"], expected_scores, rtol=1e-4, atol=1e-4)

        expected_box_slice = torch.tensor([0.6908, 0.4354, 1.0737, 1.3947])
        torch.testing.assert_close(post_processed[0]["boxes"][0], expected_box_slice, rtol=1e-4, atol=1e-4)

    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.test_save_load_pretrained_default with CLIP->GroundingDino,GroundingDinoTokenizer->BertTokenizer
    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        with tempfile.TemporaryDirectory() as tmpdir:
            processor_slow = GroundingDinoProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
            processor_slow.save_pretrained(tmpdir)
            processor_slow = GroundingDinoProcessor.from_pretrained(tmpdir, use_fast=False)

            processor_fast = GroundingDinoProcessor(tokenizer=tokenizer_fast, image_processor=image_processor)
            processor_fast.save_pretrained(tmpdir)
            processor_fast = GroundingDinoProcessor.from_pretrained(tmpdir)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, BertTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, BertTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, GroundingDinoImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, GroundingDinoImageProcessor)

    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.test_save_load_pretrained_additional_features with CLIP->GroundingDino,GroundingDinoTokenizer->BertTokenizer
    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = GroundingDinoProcessor(
                tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor()
            )
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = BertTokenizer.from_pretrained(tmpdir, bos_token="(BOS)", eos_token="(EOS)")
            image_processor_add_kwargs = GroundingDinoImageProcessor.from_pretrained(
                tmpdir, do_normalize=False, padding_value=1.0
            )

            processor = GroundingDinoProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, BertTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, GroundingDinoImageProcessor)

    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.test_image_processor with CLIP->GroundingDino
    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.test_tokenizer with CLIP->GroundingDino
    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()), ["input_ids", "token_type_ids", "attention_mask", "pixel_values", "pixel_mask"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    # Copied from tests.models.clip.test_processing_clip.CLIPProcessorTest.test_tokenizer_decode with CLIP->GroundingDino
    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_text_preprocessing_equivalence(self):
        processor = GroundingDinoProcessor.from_pretrained(self.tmpdirname)

        # check for single input
        formatted_labels = "a cat. a remote control."
        labels = ["a cat", "a remote control"]
        inputs1 = processor(text=formatted_labels, return_tensors="pt")
        inputs2 = processor(text=labels, return_tensors="pt")
        self.assertTrue(
            torch.allclose(inputs1["input_ids"], inputs2["input_ids"]),
            f"Input ids are not equal for single input: {inputs1['input_ids']} != {inputs2['input_ids']}",
        )

        # check for batched input
        formatted_labels = ["a cat. a remote control.", "a car. a person."]
        labels = [["a cat", "a remote control"], ["a car", "a person"]]
        inputs1 = processor(text=formatted_labels, return_tensors="pt", padding=True)
        inputs2 = processor(text=labels, return_tensors="pt", padding=True)
        self.assertTrue(
            torch.allclose(inputs1["input_ids"], inputs2["input_ids"]),
            f"Input ids are not equal for batched input: {inputs1['input_ids']} != {inputs2['input_ids']}",
        )
