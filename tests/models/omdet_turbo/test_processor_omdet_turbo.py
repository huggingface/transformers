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

import numpy as np
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
    from PIL import Image

    from transformers import DetrImageProcessor


@require_torch
@require_vision
class OmDetTurboProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmDetTurboProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = DetrImageProcessor()
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

        processor = OmDetTurboProcessor(image_processor, tokenizer)
        processor.save_pretrained(self.tmpdirname)

        self.input_keys = ["tasks", "classes", "pixel_values", "pixel_mask"]
        self.text_input_keys = ["input_ids", "attention_mask"]

        self.batch_size = 5
        self.num_queries = 5
        self.embed_dim = 3

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def get_fake_omdet_turbo_output(self):
        torch.manual_seed(42)
        return OmDetTurboObjectDetectionOutput(
            decoder_coord_logits=torch.rand(self.batch_size, self.num_queries, 4),
            decoder_class_logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
        )

    def get_fake_omdet_turbo_classes(self):
        input_classes = [f"class{i}" for i in range(self.num_queries)]
        return [input_classes] * self.batch_size

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
        self.assertEqual(list(post_processed[0].keys()), ["boxes", "scores", "classes"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))
        expected_scores = torch.tensor([0.7310, 0.6579, 0.6513, 0.6444, 0.6252])
        self.assertTrue(torch.allclose(post_processed[0]["scores"], expected_scores, atol=1e-4))

        expected_box_slice = torch.tensor([14.9657, 141.2052, 30.0000, 312.9670])
        self.assertTrue(torch.allclose(post_processed[0]["boxes"][0], expected_box_slice, atol=1e-4))

    def test_save_load_pretrained_additional_features(self):
        processor = OmDetTurboProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = OmDetTurboProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
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

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor).tokenizer

        input_str = "lower newer"

        encoded_processor = processor(text=input_str, padding="max_length", truncation=True, max_length=77)

        encoded_tok = tokenizer(input_str, padding="max_length", truncation=True, max_length=77)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_tasks = "task"
        input_classes = ["class1", "class2"]
        image_input = self.prepare_image_inputs()

        input_processor = processor(images=image_input, text=input_tasks, classes=input_classes, return_tensors="pt")

        assert torch.is_tensor(input_processor["pixel_values"])
        for key in self.text_input_keys:
            assert torch.is_tensor(input_processor["tasks"][key])
            assert torch.is_tensor(input_processor["classes"][key])
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

        inputs = processor(images=image_input, text=input_tasks, classes=input_classes, return_tensors="pt")

        self.assertListEqual(list(inputs.keys()), self.input_keys)

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, classes=[input_str], return_tensors="pt")
        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 117)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, classes=[input_str])
        self.assertEqual(len(inputs["pixel_values"][0][0]), 234)

    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, classes=[input_str], return_tensors="pt", max_length=112
        )
        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 112)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, classes=[input_str], size=[224, 224])
        self.assertEqual(len(inputs["pixel_values"][0][0]), 224)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            classes=[input_str],
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)
        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            classes=[input_str],
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 6)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, classes=[input_str], **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, classes=[input_str], **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["tasks"]["input_ids"][0]), 76)
