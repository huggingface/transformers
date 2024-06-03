import unittest

import numpy as np

from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import AltCLIPProcessor, CLIPImageProcessor


@require_vision
class AltClipProcessorTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "BAAI/AltCLIP"
        pass

    def get_tokenizer(self, **kwargs):
        return XLMRobertaTokenizer.from_pretrained(self.model_id, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return XLMRobertaTokenizerFast.from_pretrained(self.model_id, **kwargs)

    def get_image_processor(self, **kwargs):
        return CLIPImageProcessor.from_pretrained(self.model_id, **kwargs)

    def tearDown(self):
        pass

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
        return image_inputs

    def test_defaults_preserved(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer(model_max_length=117)
        processor = AltCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, padding="max_length")

        self.assertEqual(len(inputs["input_ids"]), 117)

    @require_torch
    def test_structured_kwargs(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AltCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        common_kwargs = {"return_tensors": "pt"}
        images_kwargs = {"crop_size": {"height": 214, "width": 214}}
        text_kwargs = {"padding": "max_length", "max_length": 76}

        # Combine them into a single dictionary
        all_kwargs = {"images_kwargs": images_kwargs, "text_kwargs": text_kwargs, "common_kwargs": common_kwargs}

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEquals(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)
