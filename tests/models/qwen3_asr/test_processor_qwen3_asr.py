import unittest
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

class Qwen3ASRProcessorTester(unittest.TestCase):
    processor_class = Qwen3ASRProcessor
    model_id = "Qwen/Qwen3-ASR-0.6B"

    def test_processor_initialization(self):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.model_id)

        processor = Qwen3ASRProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )

        assert hasattr(processor, "feature_extractor")
        assert hasattr(processor, "tokenizer")

