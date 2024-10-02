import shutil
import tempfile
import unittest

import torch

from transformers import GemmaTokenizer
from transformers.models.colpali.processing_colpali import ColPaliProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available
from transformers.utils.dummy_vision_objects import SiglipImageProcessor

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        ColPaliProcessor,
        PaliGemmaProcessor,
        SiglipImageProcessor,
    )

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class ColPaliProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ColPaliProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_processor.image_seq_length = 0
        tokenizer = GemmaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    @require_torch
    @require_vision
    def test_process_images(self):
        # Processor configuration
        image_input = self.prepare_image_inputs()
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")
        image_processor.image_seq_length = 14

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the image
        batch_feature = processor.process_images(image_input)

        # Assertions
        self.assertIn("pixel_values", batch_feature)
        self.assertEqual(batch_feature["pixel_values"].shape, torch.Size([1, 3, 448, 448]))

    @require_torch
    @require_vision
    def test_process_queries(self):
        # Inputs
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Processor configuration
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")
        image_processor.image_seq_length = 14

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the image
        batch_feature = processor.process_queries(queries)

        # Assertions
        self.assertIn("input_ids", batch_feature)
        self.assertIsInstance(batch_feature["input_ids"], torch.Tensor)
        self.assertEqual(batch_feature["input_ids"].shape[0], len(queries))
