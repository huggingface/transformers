import gc
import threading
import unittest
import weakref
from unittest.mock import MagicMock

import numpy as np
import torch

from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    AssistedCandidateGeneratorDifferentTokenizers,
)
 

class TestAssistedCandidateGeneratorDifferentTokenizers(unittest.TestCase):
    def test_no_intersection(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[4, 5, 6]])
        result = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(prompt, prompt_plus_new_tokens)
        self.assertEqual(result, (None, None, None))

    def test_complete_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_partial_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_no_new_tokens(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))


class TestAssistantToTargetTranslator(unittest.TestCase):
    def setUp(self):
        # Create mock tokenizers with predefined vocabularies
        self.target_tokenizer = MagicMock()
        self.assistant_tokenizer = MagicMock()

        # Define mock vocabularies for the tokenizers
        self.target_vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        self.assistant_vocab = {"hello": 0, "world": 1, "foo": 2, "baz": 4}

        self.target_tokenizer.get_vocab.return_value = self.target_vocab
        self.assistant_tokenizer.get_vocab.return_value = self.assistant_vocab

        # Instantiate the class under test
        self.translator = AssistantToTargetTranslator(
            target_tokenizer=self.target_tokenizer, assistant_tokenizer=self.assistant_tokenizer
        )

    def test_get_assistant_to_target_input_ids(self):
        """Test the mapping from assistant tokens to target tokens."""
        expected_mapping = {0: 0, 1: 1, 2: 2}
        actual_mapping = self.translator._assistant_to_target_input_ids
        self.assertEqual(actual_mapping, expected_mapping)

    def test_get_suppress_input_ids(self):
        """Test the suppression of assistant input IDs not present in the target vocabulary."""
        expected_suppress_ids = [4]
        actual_suppress_ids = self.translator.suppress_input_ids
        self.assertEqual(actual_suppress_ids, expected_suppress_ids)

    def test_get_target_ids(self):
        """Test the translation of assistant candidate IDs to target candidate IDs."""
        assistant_input_ids = torch.LongTensor([[0, 1, 2]])  # 'hello world foo' in assistant tokenizer
        target_input_ids = torch.LongTensor([[0, 1, 2]])  # 'hello world foo' in target tokenizer
        assistant_candidate_ids = torch.LongTensor([[0, 1, 2, 4]])  # 'hello world foo baz' in assistant tokenizer

        expected_target_ids = torch.LongTensor(
            [[0, 1, 2, 4]]
        )  # 'hello world foo baz' in target tokenizer (baz id remains 4)

        actual_target_ids = self.translator.get_target_ids(
            assistant_input_ids, target_input_ids, assistant_candidate_ids
        )
        self.assertTrue(torch.equal(actual_target_ids, expected_target_ids))

    def test_get_target_logits(self):
        """Test the conversion of assistant logits to target logits."""
        # Assistant logits for IDs 0, 1, 2
        assistant_logits = torch.FloatTensor([[[0.1, 0.2, 0.3]]])  # Shape (1, 1, 3)

        # Expected target logits (target_vocab_size = 4)
        expected_target_logits = torch.full((1, 1, 4), -float("inf"))
        expected_target_logits[0, 0, 0] = 0.1  # 'hello'
        expected_target_logits[0, 0, 1] = 0.2  # 'world'
        expected_target_logits[0, 0, 2] = 0.3  # 'foo'
        # The 'bar' token in target vocab remains at -inf

        actual_target_logits = self.translator.get_target_logits(assistant_logits)
        self.assertTrue(torch.equal(actual_target_logits, expected_target_logits))


class MockTokenizer:
    """A simple mock tokenizer class that supports weak references."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {}

    def get_vocab(self):
        return self._vocab


class TestAssistantVocabTranslatorCache(unittest.TestCase):
    def setUp(self):
        # Clear the cache before each test
        AssistantVocabTranslatorCache._cache.clear()
        # Create mock tokenizers with different vocabularies
        self.target_tokenizer = MockTokenizer({"hello": 0, "world": 1})
        self.assistant_tokenizer = MockTokenizer({"hello": 0, "world": 1, "foo": 2})
        self.other_target_tokenizer = MockTokenizer({"foo": 2, "bar": 3})
        self.other_assistant_tokenizer = MockTokenizer({"baz": 4, "qux": 5})

    def test_same_instance_for_same_tokenizers(self):
        """Test that the same translator is returned for the same tokenizers."""
        translator1 = AssistantVocabTranslatorCache.get_translator(self.target_tokenizer, self.assistant_tokenizer)
        translator2 = AssistantVocabTranslatorCache.get_translator(self.target_tokenizer, self.assistant_tokenizer)
        self.assertIs(translator1, translator2, "Translators should be cached and identical")

    def test_different_instances_for_different_tokenizers(self):
        """Test that different tokenizers produce different translators."""
        translator1 = AssistantVocabTranslatorCache.get_translator(self.target_tokenizer, self.assistant_tokenizer)
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.other_target_tokenizer, self.other_assistant_tokenizer
        )
        self.assertIsNot(translator1, translator2, "Translators should differ for different tokenizers")

    def test_cache_with_weakref_key(self):
        """Ensure that the cache uses weak references as keys."""
        initial_cache_size = len(AssistantVocabTranslatorCache._cache)
        target_tokenizer = MockTokenizer({"hello": 0})
        assistant_tokenizer = MockTokenizer({"hello": 0})

        # Store translator in a local variable to avoid it being kept alive
        translator = AssistantVocabTranslatorCache.get_translator(target_tokenizer, assistant_tokenizer)
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

        # Delete all strong references
        del target_tokenizer
        del assistant_tokenizer
        del translator

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The cache size remains increased due to strong references
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

    def test_weakref_cache_cleanup(self):
        """Test that the cache cleans up translators when tokenizers are garbage collected."""

        def create_translator():
            target_tokenizer = MockTokenizer({"hello": 0})
            assistant_tokenizer = MockTokenizer({"hello": 0})
            translator = AssistantVocabTranslatorCache.get_translator(target_tokenizer, assistant_tokenizer)
            # Create weak references before returning
            refs = (weakref.ref(translator), weakref.ref(target_tokenizer), weakref.ref(assistant_tokenizer))
            # Remove strong references inside the function
            del target_tokenizer
            del assistant_tokenizer
            del translator
            return refs

        translator_ref, target_ref, assistant_ref = create_translator()

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The tokenizers and translator are not garbage collected due to strong references
        self.assertIsNotNone(target_ref(), "Target tokenizer should still be alive due to strong references")
        self.assertIsNotNone(assistant_ref(), "Assistant tokenizer should still be alive due to strong references")
        self.assertIsNotNone(translator_ref(), "Translator should still be alive due to strong references")

    def test_thread_safety(self):
        """Test that get_translator is thread-safe."""
        translators = []

        def get_translator():
            translator = AssistantVocabTranslatorCache.get_translator(self.target_tokenizer, self.assistant_tokenizer)
            translators.append(translator)

        threads = [threading.Thread(target=get_translator) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All translators should be the same instance
        for translator in translators:
            self.assertIs(translators[0], translator, "All translators should be identical across threads")
