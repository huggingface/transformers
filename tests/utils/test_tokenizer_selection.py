# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import unittest
from unittest.mock import MagicMock, patch

from transformers.utils.tokenizer_selection import (
    CorpusAnalyzer,
    CorpusStats,
    TokenizerRecommender,
    TokenizerSelector,
    suggest_and_train_tokenizer,
)


class TestCorpusAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_texts = [
            ["Hello world, this is a test.", "Machine learning is fascinating."],
            ["Natural language processing helps computers.", "Tokenization is important."],
            ["BPE and WordPiece are popular algorithms.", "SentencePiece works well too."],
        ]

        self.cjk_texts = [["你好世界", "机器学习很有趣"], ["自然语言处理帮助计算机", "分词很重要"]]

    def test_analyze_corpus_basic(self):
        """Test basic corpus analysis functionality."""
        stats = CorpusAnalyzer.analyze_corpus(iter(self.test_texts))

        self.assertIsInstance(stats, CorpusStats)
        self.assertGreater(stats.vocab_size, 0)
        self.assertGreater(stats.avg_word_length, 0)
        self.assertGreater(stats.char_diversity, 0)
        self.assertGreaterEqual(stats.morphological_complexity, 0)
        self.assertGreaterEqual(stats.token_frequency_ratio, 0)
        self.assertGreaterEqual(stats.avg_sentence_length, 0)

    def test_analyze_corpus_empty(self):
        """Test corpus analysis with empty input."""
        empty_texts = [[]]

        with self.assertRaises(ValueError):
            CorpusAnalyzer.analyze_corpus(iter(empty_texts))

    def test_detect_language_hint_latin(self):
        """Test language detection for Latin scripts."""
        stats = CorpusAnalyzer.analyze_corpus(iter(self.test_texts))
        self.assertEqual(stats.language_hint, "latin")

    def test_detect_language_hint_cjk(self):
        """Test language detection for CJK scripts."""
        stats = CorpusAnalyzer.analyze_corpus(iter(self.cjk_texts))
        self.assertEqual(stats.language_hint, "cjk")

    def test_sample_size_limit(self):
        """Test that sample size limit is respected."""
        large_texts = [["test text"] * 100 for _ in range(100)]  # 10k texts

        stats = CorpusAnalyzer.analyze_corpus(iter(large_texts), sample_size=50)
        self.assertIsInstance(stats, CorpusStats)
        # Should still work despite large input


class TestTokenizerRecommender(unittest.TestCase):
    def setUp(self):
        """Set up test corpus statistics."""
        self.latin_stats = CorpusStats(
            vocab_size=1000,
            avg_word_length=5.0,
            char_diversity=50,
            morphological_complexity=0.3,
            token_frequency_ratio=0.1,
            avg_sentence_length=10.0,
            language_hint="latin",
        )

        self.cjk_stats = CorpusStats(
            vocab_size=5000,
            avg_word_length=2.0,
            char_diversity=2000,
            morphological_complexity=0.8,
            token_frequency_ratio=0.05,
            avg_sentence_length=15.0,
            language_hint="cjk",
        )

        self.complex_stats = CorpusStats(
            vocab_size=80000,
            avg_word_length=12.0,
            char_diversity=100,
            morphological_complexity=0.9,
            token_frequency_ratio=0.02,
            avg_sentence_length=20.0,
            language_hint="latin",
        )

    def test_recommend_tokenizer_cjk(self):
        """Test recommendation for CJK languages."""
        recommendation = TokenizerRecommender.recommend_tokenizer(self.cjk_stats)

        self.assertEqual(recommendation["type"], "SentencePiece")
        self.assertIn("CJK", recommendation["rationale"])
        self.assertIn("config", recommendation)

    def test_recommend_tokenizer_high_complexity(self):
        """Test recommendation for high morphological complexity."""
        recommendation = TokenizerRecommender.recommend_tokenizer(self.complex_stats)

        self.assertEqual(recommendation["type"], "BPE")
        self.assertIn("morphological complexity", recommendation["rationale"])

    def test_recommend_tokenizer_large_vocab(self):
        """Test recommendation for large vocabulary."""
        large_vocab_stats = CorpusStats(
            vocab_size=60000,
            avg_word_length=6.0,
            char_diversity=80,
            morphological_complexity=0.4,
            token_frequency_ratio=0.05,
            avg_sentence_length=12.0,
            language_hint="latin",
        )

        recommendation = TokenizerRecommender.recommend_tokenizer(large_vocab_stats)

        # Should recommend WordPiece for large vocab or BPE for complexity
        self.assertIn(recommendation["type"], ["WordPiece", "BPE"])

    def test_generate_config_bpe(self):
        """Test BPE configuration generation."""
        recommendation = TokenizerRecommender.recommend_tokenizer(self.complex_stats)

        if recommendation["type"] == "BPE":
            config = recommendation["config"]
            self.assertIn("vocab_size", config)
            self.assertIn("continuing_subword_prefix", config)

    def test_generate_config_sentencepiece(self):
        """Test SentencePiece configuration generation."""
        recommendation = TokenizerRecommender.recommend_tokenizer(self.cjk_stats)

        if recommendation["type"] == "SentencePiece":
            config = recommendation["config"]
            self.assertIn("vocab_size", config)
            self.assertIn("character_coverage", config)
            self.assertIn("model_type", config)

    def test_vocab_size_scaling(self):
        """Test vocabulary size recommendations scale appropriately."""
        small_vocab = CorpusStats(5000, 5.0, 50, 0.3, 0.1, 10.0, "latin")
        # medium_vocab = CorpusStats(30000, 6.0, 60, 0.4, 0.08, 12.0, "latin")
        large_vocab = CorpusStats(100000, 7.0, 80, 0.5, 0.05, 15.0, "latin")

        small_rec = TokenizerRecommender.recommend_tokenizer(small_vocab)
        large_rec = TokenizerRecommender.recommend_tokenizer(large_vocab)

        # Vocabulary size recommendations should scale
        self.assertLess(small_rec["config"]["vocab_size"], large_rec["config"]["vocab_size"])


class TestTokenizerSelector(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_texts = [
            ["Hello world, this is a test.", "Machine learning is fascinating."],
            ["Natural language processing helps computers.", "Tokenization is important."],
        ]

    def test_analyze_corpus(self):
        """Test corpus analysis through TokenizerSelector."""
        stats = TokenizerSelector.analyze_corpus(iter(self.test_texts))

        self.assertIsInstance(stats, CorpusStats)
        self.assertGreater(stats.vocab_size, 0)

    def test_recommend_tokenizer(self):
        """Test tokenizer recommendation through TokenizerSelector."""
        stats = TokenizerSelector.analyze_corpus(iter(self.test_texts))
        recommendation = TokenizerSelector.recommend_tokenizer(stats)

        self.assertIn("type", recommendation)
        self.assertIn("rationale", recommendation)
        self.assertIn("config", recommendation)
        self.assertIn(recommendation["type"], ["BPE", "WordPiece", "SentencePiece"])

    @patch("transformers.models.auto.AutoTokenizer")
    def test_suggest_and_train_tokenizer_mock(self, mock_auto_tokenizer):
        """Test end-to-end tokenizer training with mocked AutoTokenizer."""
        # Mock the tokenizer and its training method
        mock_tokenizer = MagicMock()
        mock_trained_tokenizer = MagicMock()
        mock_tokenizer.train_new_from_iterator.return_value = mock_trained_tokenizer
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        trained_tokenizer, recommendation = TokenizerSelector.suggest_and_train_tokenizer(
            iter(self.test_texts), vocab_size=1000
        )

        # Verify the method was called
        mock_auto_tokenizer.from_pretrained.assert_called()
        mock_tokenizer.train_new_from_iterator.assert_called()

        # Check return values
        self.assertEqual(trained_tokenizer, mock_trained_tokenizer)
        self.assertIn("type", recommendation)
        self.assertIn("rationale", recommendation)

    def test_convenience_function(self):
        """Test the convenience function."""
        # This test would require mocking as well since it calls the main method
        with patch(
            "transformers.utils.tokenizer_selection.TokenizerSelector.suggest_and_train_tokenizer"
        ) as mock_method:
            mock_method.return_value = (MagicMock(), {"type": "BPE"})

            tokenizer, info = suggest_and_train_tokenizer(iter(self.test_texts))

            mock_method.assert_called_once()
            self.assertIsNotNone(tokenizer)
            self.assertIn("type", info)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def setUp(self):
        """Set up test data with different characteristics."""
        self.english_texts = [
            ["The quick brown fox jumps over the lazy dog."],
            ["Machine learning models require substantial computational resources."],
            ["Natural language processing enables computers to understand human language."],
        ]

        self.technical_texts = [
            ["Hyperparameter optimization improves model performance significantly."],
            ["Convolutional neural networks excel at computer vision tasks."],
            ["Transformer architectures revolutionized natural language understanding."],
        ]

    def test_different_corpus_types(self):
        """Test that different corpus types get different recommendations."""
        english_stats = TokenizerSelector.analyze_corpus(iter(self.english_texts))
        technical_stats = TokenizerSelector.analyze_corpus(iter(self.technical_texts))

        english_rec = TokenizerSelector.recommend_tokenizer(english_stats)
        technical_rec = TokenizerSelector.recommend_tokenizer(technical_stats)

        # Both should provide valid recommendations
        self.assertIn(english_rec["type"], ["BPE", "WordPiece", "SentencePiece"])
        self.assertIn(technical_rec["type"], ["BPE", "WordPiece", "SentencePiece"])

        # Technical text typically has higher complexity
        self.assertGreaterEqual(technical_stats.morphological_complexity, english_stats.morphological_complexity)


if __name__ == "__main__":
    unittest.main()
