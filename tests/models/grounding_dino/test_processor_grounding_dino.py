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

import numpy as np
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
    from PIL import Image

    from transformers import GroundingDinoImageProcessor


@require_torch
@require_vision
class GroundingDinoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "IDEA-Research/grounding-dino-base"
    processor_class = GroundingDinoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]","want","##want","##ed","wa","un","runn","##ing",",","low","lowest"]  # fmt: skip
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
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
        self.image_processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

        image_processor = GroundingDinoImageProcessor()
        tokenizer = BertTokenizer.from_pretrained(self.from_pretrained_id)

        processor = GroundingDinoProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

        self.batch_size = 7
        self.num_queries = 5
        self.embed_dim = 5
        self.seq_length = 5

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.get_tokenizer with CLIP->Bert
    def get_tokenizer(self, **kwargs):
        return BertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.get_rust_tokenizer with CLIP->Bert
    def get_rust_tokenizer(self, **kwargs):
        return BertTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.get_image_processor with CLIP->GroundingDino
    def get_image_processor(self, **kwargs):
        return GroundingDinoImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.tearDown
    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.prepare_image_inputs
    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def get_fake_grounding_dino_output(self):
        torch.manual_seed(42)
        return GroundingDinoObjectDetectionOutput(
            pred_boxes=torch.rand(self.batch_size, self.num_queries, 4),
            logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
        )

    def get_fake_grounding_dino_input_ids(self):
        input_ids = torch.tensor([101, 1037, 4937, 1012, 102])
        return torch.stack([input_ids] * self.batch_size, dim=0)

    def test_post_process_grounded_object_detection(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        grounding_dino_output = self.get_fake_grounding_dino_output()
        grounding_dino_input_ids = self.get_fake_grounding_dino_input_ids()

        post_processed = processor.post_process_grounded_object_detection(
            grounding_dino_output, grounding_dino_input_ids
        )

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["scores", "labels", "boxes"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))

        expected_scores = torch.tensor([0.7050, 0.7222, 0.7222, 0.6829, 0.7220])
        self.assertTrue(torch.allclose(post_processed[0]["scores"], expected_scores, atol=1e-4))

        expected_box_slice = torch.tensor([0.6908, 0.4354, 1.0737, 1.3947])
        self.assertTrue(torch.allclose(post_processed[0]["boxes"][0], expected_box_slice, atol=1e-4))

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_save_load_pretrained_default with CLIP->GroundingDino,GroundingDinoTokenizer->BertTokenizer
    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = GroundingDinoProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = GroundingDinoProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        processor_fast = GroundingDinoProcessor(tokenizer=tokenizer_fast, image_processor=image_processor)
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = GroundingDinoProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, BertTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, BertTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, GroundingDinoImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, GroundingDinoImageProcessor)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_save_load_pretrained_additional_features with CLIP->GroundingDino,GroundingDinoTokenizer->BertTokenizer
    def test_save_load_pretrained_additional_features(self):
        processor = GroundingDinoProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = GroundingDinoProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, BertTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, GroundingDinoImageProcessor)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_image_processor with CLIP->GroundingDino
    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_tokenizer with CLIP->GroundingDino
    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
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

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_tokenizer_decode with CLIP->GroundingDino
    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    # Copied from tests.models.clip.test_processor_clip.CLIPProcessorTest.test_model_input_names with CLIP->GroundingDino
    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = GroundingDinoProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size={"height": 234, "width": 234})
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
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
            text=input_str, images=image_input, return_tensors="pt", padding="max_length", max_length=112
        )
        self.assertEqual(len(inputs["input_ids"][0]), 112)

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

        inputs = processor(text=input_str, images=image_input, return_tensors="pt", padding="max_length")
        self.assertEqual(len(inputs["input_ids"][0]), 117)

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

        inputs = processor(text=input_str, images=image_input, size=[224, 224])
        self.assertEqual(len(inputs["pixel_values"][0][0]), 224)

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

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)

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

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)

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
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            crop_size={"height": 214, "width": 214},
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 6)
