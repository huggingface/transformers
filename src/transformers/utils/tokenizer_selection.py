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

"""
Tokenizer selection utilities for corpus-aware tokenizer recommendations.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Optional


logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    """
    Container for corpus analysis statistics.
    """

    vocab_size: int
    avg_word_length: float
    char_diversity: int
    morphological_complexity: float
    token_frequency_ratio: float
    avg_sentence_length: float
    language_hint: Optional[str] = None


class CorpusAnalyzer:
    """
    Analyzes text corpus characteristics to inform tokenizer selection.
    """

    @staticmethod
    def analyze_corpus(text_iterator: Iterator[list[str]], sample_size: int = 10000) -> CorpusStats:
        """
        Analyze corpus characteristics.

        Args:
            text_iterator: Iterator yielding batches of text strings
            sample_size: Maximum number of texts to analyze for efficiency

        Returns:
            CorpusStats: Statistical analysis of the corpus
        """
        word_lengths = []
        char_counter = Counter()
        word_counter = Counter()
        sentence_lengths = []
        all_chars = set()
        processed_count = 0

        for batch in text_iterator:
            for text in batch:
                if processed_count >= sample_size:
                    break

                # Basic text processing
                sentences = text.split(".")
                sentence_lengths.extend([len(s.split()) for s in sentences if s.strip()])

                words = re.findall(r"\b\w+\b", text.lower())
                word_lengths.extend([len(word) for word in words])
                word_counter.update(words)

                chars = [c for c in text if c.isalnum()]
                char_counter.update(chars)
                all_chars.update(chars)

                processed_count += 1

            if processed_count >= sample_size:
                break

        if not word_lengths:
            raise ValueError("No valid text found in corpus")

        # Calculate statistics
        vocab_size = len(word_counter)
        avg_word_length = sum(word_lengths) / len(word_lengths)
        char_diversity = len(all_chars)

        # Morphological complexity (ratio of unique words to total words)
        total_words = sum(word_counter.values())
        morphological_complexity = vocab_size / total_words if total_words > 0 else 0

        # Token frequency distribution (how concentrated the vocabulary is)
        word_frequencies = list(word_counter.values())
        token_frequency_ratio = max(word_frequencies) / sum(word_frequencies) if word_frequencies else 0

        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        # Simple language detection based on character patterns
        language_hint = CorpusAnalyzer._detect_language_hint(char_counter)

        return CorpusStats(
            vocab_size=vocab_size,
            avg_word_length=avg_word_length,
            char_diversity=char_diversity,
            morphological_complexity=morphological_complexity,
            token_frequency_ratio=token_frequency_ratio,
            avg_sentence_length=avg_sentence_length,
            language_hint=language_hint,
        )

    @staticmethod
    def _detect_language_hint(char_counter: Counter) -> Optional[str]:
        """Simple language detection based on character frequency patterns."""
        total_chars = sum(char_counter.values())
        if total_chars == 0:
            return None

        # Check for common patterns
        latin_chars = sum(count for char, count in char_counter.items() if ord(char) < 256)
        asian_chars = sum(count for char, count in char_counter.items() if ord(char) > 4352)  # CJK range approximation

        latin_ratio = latin_chars / total_chars
        asian_ratio = asian_chars / total_chars

        if asian_ratio > 0.3:
            return "cjk"  # Chinese, Japanese, Korean
        elif latin_ratio > 0.8:
            return "latin"
        else:
            return "mixed"


class TokenizerRecommender:
    """
    Recommends tokenizer type and configuration based on corpus statistics.
    """

    @staticmethod
    def recommend_tokenizer(corpus_stats: CorpusStats) -> dict[str, Any]:
        """
        Recommend tokenizer type and configuration based on corpus characteristics.

        Args:
            corpus_stats: Statistics from corpus analysis

        Returns:
            Dict containing recommendation with 'type', 'rationale', and 'config'
        """
        recommendations = []

        # Rule-based recommendation logic
        if corpus_stats.language_hint == "cjk":
            recommendations.append(
                {
                    "type": "SentencePiece",
                    "score": 0.9,
                    "rationale": "SentencePiece handles CJK languages effectively without whitespace dependency",
                }
            )

        if corpus_stats.morphological_complexity > 0.7:
            recommendations.append(
                {
                    "type": "BPE",
                    "score": 0.8,
                    "rationale": "High morphological complexity benefits from BPE's subword handling",
                }
            )

        if corpus_stats.vocab_size > 50000:
            recommendations.append(
                {"type": "WordPiece", "score": 0.7, "rationale": "Large vocabulary size suits WordPiece tokenization"}
            )

        if corpus_stats.avg_word_length > 8.0:
            recommendations.append(
                {
                    "type": "BPE",
                    "score": 0.8,
                    "rationale": "Long average word length benefits from subword tokenization",
                }
            )

        # Default fallback
        if not recommendations:
            recommendations.append(
                {"type": "BPE", "score": 0.6, "rationale": "BPE is a robust default choice for most corpora"}
            )

        # Select highest scoring recommendation
        best_rec = max(recommendations, key=lambda x: x["score"])

        # Generate configuration suggestions
        config = TokenizerRecommender._generate_config(corpus_stats, best_rec["type"])

        return {
            "type": best_rec["type"],
            "rationale": best_rec["rationale"],
            "config": config,
            "corpus_stats": corpus_stats,
        }

    @staticmethod
    def _generate_config(corpus_stats: CorpusStats, tokenizer_type: str) -> dict[str, Any]:
        """Generate tokenizer configuration based on corpus stats and type."""
        config = {}

        # Vocabulary size suggestion
        if corpus_stats.vocab_size < 10000:
            config["vocab_size"] = 16000
        elif corpus_stats.vocab_size < 50000:
            config["vocab_size"] = 32000
        else:
            config["vocab_size"] = 50000

        # Type-specific configurations
        if tokenizer_type == "BPE":
            config.update(
                {
                    "dropout": 0.1 if corpus_stats.morphological_complexity > 0.5 else None,
                    "continuing_subword_prefix": "##",
                }
            )
        elif tokenizer_type == "WordPiece":
            config.update(
                {
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": max(100, int(corpus_stats.avg_word_length * 10)),
                }
            )
        elif tokenizer_type == "SentencePiece":
            config.update(
                {
                    "character_coverage": 0.9995 if corpus_stats.language_hint == "latin" else 0.995,
                    "model_type": "unigram",
                }
            )

        return config


class TokenizerSelector:
    """
    Main utility class for context-aware tokenizer selection and training.
    """

    @staticmethod
    def suggest_and_train_tokenizer(
        text_iterator: Iterator[list[str]],
        vocab_size: Optional[int] = None,
        base_tokenizer: str = "google-bert/bert-base-uncased",
        sample_size: int = 10000,
        **trainer_kwargs,
    ):
        """
        End-to-end utility to analyze corpus, recommend tokenizer, and train it.

        Args:
            text_iterator: Iterator yielding batches of text strings
            vocab_size: Target vocabulary size (auto-selected if None)
            base_tokenizer: Base tokenizer to use as template for training
            sample_size: Number of texts to analyze for recommendations
            **trainer_kwargs: Additional arguments passed to tokenizer trainer

        Returns:
            Tuple of (trained_tokenizer, recommendation_info)
        """
        logger.info("Analyzing corpus characteristics...")

        # Convert iterator to list for reuse (needed for both analysis and training)
        text_batches = list(text_iterator)

        # Analyze corpus
        corpus_stats = CorpusAnalyzer.analyze_corpus(iter(text_batches), sample_size)

        # Get recommendation
        recommendation = TokenizerRecommender.recommend_tokenizer(corpus_stats)

        logger.info(f"Recommended tokenizer type: {recommendation['type']}")
        logger.info(f"Rationale: {recommendation['rationale']}")

        # Use recommended vocab size if not provided
        if vocab_size is None:
            vocab_size = recommendation["config"]["vocab_size"]

        # Load base tokenizer for training (lazy import to avoid circular dependency)
        from ..models.auto import AutoTokenizer

        try:
            base_tok = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)
        except Exception:
            logger.warning(f"Could not load {base_tokenizer}, falling back to bert-base-uncased")
            base_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)

        # Merge trainer configs
        trainer_config = {**recommendation["config"], **trainer_kwargs}
        # Remove vocab_size from trainer_config since it's a separate parameter
        trainer_config.pop("vocab_size", None)

        # Train new tokenizer using existing method
        logger.info(f"Training {recommendation['type']} tokenizer with vocab_size={vocab_size}")

        trained_tokenizer = base_tok.train_new_from_iterator(
            text_iterator=iter(text_batches), vocab_size=vocab_size, **trainer_config
        )

        return trained_tokenizer, recommendation

    @staticmethod
    def analyze_corpus(text_iterator: Iterator[list[str]], sample_size: int = 10000) -> CorpusStats:
        """
        Analyze corpus and return statistics.

        Args:
            text_iterator: Iterator yielding batches of text strings
            sample_size: Number of texts to analyze

        Returns:
            CorpusStats: Analysis results
        """
        return CorpusAnalyzer.analyze_corpus(text_iterator, sample_size)

    @staticmethod
    def recommend_tokenizer(corpus_stats: CorpusStats) -> dict[str, Any]:
        """
        Get tokenizer recommendation based on corpus statistics.

        Args:
            corpus_stats: Analysis results from analyze_corpus

        Returns:
            Dict: Recommendation with type, rationale, and config
        """
        return TokenizerRecommender.recommend_tokenizer(corpus_stats)


# Convenience function for simple usage
def suggest_and_train_tokenizer(text_iterator: Iterator[list[str]], vocab_size: Optional[int] = None, **kwargs):
    """
    Convenience function for end-to-end tokenizer selection and training.

    Args:
        text_iterator: Iterator yielding batches of text strings
        vocab_size: Target vocabulary size (auto-selected if None)
        **kwargs: Additional arguments passed to TokenizerSelector

    Returns:
        Tuple of (trained_tokenizer, recommendation_info)

    Example:
        >>> texts = [["Hello world", "This is a test"], ["More training data"]]
        >>> tokenizer, info = suggest_and_train_tokenizer(iter(texts))
        >>> print(f"Trained {info['type']} tokenizer: {info['rationale']}")
    """
    return TokenizerSelector.suggest_and_train_tokenizer(text_iterator, vocab_size, **kwargs)
