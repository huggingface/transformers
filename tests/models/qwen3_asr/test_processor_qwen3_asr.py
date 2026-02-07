import unittest
import tempfile
import shutil
import numpy as np
import torch
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

class Qwen3ASRProcessorTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "Qwen/Qwen3-ASR-0.6B"
        cls.tmpdirname = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)
    
    def get_tokenizer(self, **kwargs):
        return Qwen2TokenizerFast.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return WhisperFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def test_save_load_pretrained_default(self):    
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen3ASRProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, Qwen2TokenizerFast)

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        text = "hello world"
        encoded_processor = processor(text=text)
        encoded_tokenizer = tokenizer(text)

        for key in encoded_tokenizer:
            self.assertListEqual(encoded_processor[key][0], encoded_tokenizer[key])

    def test_feature_extractor(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = np.random.randn(16000).astype(np.float32)

        fe_out = feature_extractor(raw_speech, return_tensors="np")
        proc_out = processor.feature_extractor(raw_speech, return_tensors="np")

        for key in fe_out:
            np.testing.assert_allclose(fe_out[key], proc_out[key], rtol=1e-4, atol=1e-4)

    def test_tokenizer_decode(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 2, 3, 4], [5, 6, 7]]
        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tokenizer = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_processor, decoded_tokenizer)