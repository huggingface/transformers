# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    pass

if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPVisionModel,
        LlavaProcessor,
        PreTrainedTokenizerFast,
    )


@require_torch
class LlavaProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        image_processor.pad = False

        processor = LlavaProcessor(image_processor, tokenizer, vision_model)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_vision_model(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).vision_model

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs[0]

    def test_save_load_pretrained_additional_features(self):
        processor = LlavaProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            vision_model=self.get_vision_model(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = LlavaProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)
        self.assertIsInstance(processor.vision_model, CLIPVisionModel)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        vision_model = self.get_vision_model()

        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor, vision_model=vision_model)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="pt")
        input_feat_extract = vision_model(input_feat_extract["pixel_values"], output_hidden_states=True)
        image_features = input_feat_extract.hidden_states[-2]
        image_features = image_features[:, 1:]

        input_processor = processor(images=image_input, return_tensors="pt")["pixel_values"]
        print(image_features, input_processor)
        self.assertAlmostEqual(image_features.sum(), input_processor.sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        vision_model = self.get_vision_model()

        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor, vision_model=vision_model)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "pixel_values"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        vision_model = self.get_vision_model()

        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor, vision_model=vision_model)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        vision_model = self.get_vision_model()

        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor, vision_model=vision_model)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "pixel_values"],
        )
