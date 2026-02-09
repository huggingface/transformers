import unittest
import tempfile
import shutil
import numpy as np
import torch
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import (
    Qwen2TokenizerFast,
    WhisperFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.testing_utils import (
    require_librosa, 
    require_torch, 
    require_torchaudio,
)
from ...test_processing_common import ProcessorTesterMixin

class Qwen3ASRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3ASRProcessor

    @classmethod
    @require_torch
    @require_torchaudio
    def setUpClass(cls):
        cls.checkpoint = "Qwen/Qwen3-ASR-0.6B"
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Qwen3ASRProcessor.from_pretrained(cls.checkpoint) 
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    @require_torchaudio
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    @require_torchaudio
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def test_can_load_various_tokenizers(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    @require_torchaudio
    def test_save_load_pretrained_default(self):    
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen3ASRProcessor.from_pretrained(self.tmpdirname)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = Qwen3ASRProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)
        self.assertIsInstance(reloaded.tokenizer, Qwen2TokenizerFast)

    @require_torch
    @require_torchaudio
    def test_tokenizer_integration(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        prompt = (
            "<|im_start|>user\n"
            "<sound>Transcribe the following audio.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        tokens = tokenizer.tokenize(prompt)

        # Core structural checks
        self.assertIn("<sound>", tokens)
        self.assertIn("<|im_start|>", tokens)
        self.assertIn("<|im_end|>", tokens)

        # Text should be tokenized, not dropped
        self.assertTrue(any("Transcribe" in tok or "transcribe" in tok for tok in tokens))

        # Sanity check: non-empty and stable
        self.assertGreater(len(tokens), 5)
