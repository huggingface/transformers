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

import pytest

from transformers import AutoProcessor, CLIPTokenizerFast, OmDetTurboProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]

if is_torch_available():
    import torch

    from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboObjectDetectionOutput

if is_vision_available():
    from transformers import DetrImageProcessor


@require_torch
@require_vision
class OmDetTurboProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmDetTurboProcessor
    text_input_name = "classes_input_ids"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = DetrImageProcessor()
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

        processor = OmDetTurboProcessor(image_processor, tokenizer)
        processor.save_pretrained(cls.tmpdirname)

        cls.input_keys = [
            "tasks_input_ids",
            "tasks_attention_mask",
            "classes_input_ids",
            "classes_attention_mask",
            "classes_structure",
            "pixel_values",
            "pixel_mask",
        ]

        cls.batch_size = 5
        cls.num_queries = 5
        cls.embed_dim = 3

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_fake_omdet_turbo_output(self):
        classes = self.get_fake_omdet_turbo_classes()
        classes_structure = torch.tensor([len(sublist) for sublist in classes])
        torch.manual_seed(42)
        return OmDetTurboObjectDetectionOutput(
            decoder_coord_logits=torch.rand(self.batch_size, self.num_queries, 4),
            decoder_class_logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
            classes_structure=classes_structure,
        )

    def get_fake_omdet_turbo_classes(self):
        return [[f"class{i}_{j}" for i in range(self.num_queries)] for j in range(self.batch_size)]

    def test_post_process_grounded_object_detection(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        omdet_turbo_output = self.get_fake_omdet_turbo_output()
        omdet_turbo_classes = self.get_fake_omdet_turbo_classes()

        post_processed = processor.post_process_grounded_object_detection(
            omdet_turbo_output, omdet_turbo_classes, target_sizes=[(400, 30) for _ in range(self.batch_size)]
        )

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["boxes", "scores", "labels", "text_labels"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))
        expected_scores = torch.tensor([0.7310, 0.6579, 0.6513, 0.6444, 0.6252])
        torch.testing.assert_close(post_processed[0]["scores"], expected_scores, rtol=1e-4, atol=1e-4)

        expected_box_slice = torch.tensor([14.9657, 141.2052, 30.0000, 312.9670])
        torch.testing.assert_close(post_processed[0]["boxes"][0], expected_box_slice, rtol=1e-4, atol=1e-4)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = OmDetTurboProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
            image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

            processor = OmDetTurboProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, DetrImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor).image_processor

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor).tokenizer

        input_str = "lower newer"

        encoded_processor = processor(text=input_str, padding="max_length", truncation=True, max_length=77)

        encoded_tok = tokenizer(input_str, padding="max_length", truncation=True, max_length=77)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_tasks = "task"
        input_classes = ["class1", "class2"]
        image_input = self.prepare_image_inputs()

        input_processor = processor(images=image_input, text=input_classes, task=input_tasks, return_tensors="pt")

        for key in self.input_keys:
            assert torch.is_tensor(input_processor[key])
        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_tasks = "task"
        input_classes = ["class1", "class2"]
        image_input = self.prepare_image_inputs()
        inputs = processor(images=image_input, text=input_classes, task=input_tasks, return_tensors="pt")

        self.assertListEqual(list(inputs.keys()), self.input_keys)
