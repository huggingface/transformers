import unittest
from parameterized import parameterized

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers
from transformers.generation.utils import GenerationConfig


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


class TestGenerateWithDifferentModels(unittest.TestCase):
    """Tests generation with different target and assistant models."""

    @parameterized.expand([
        (False,),
        (True,),
    ])
    def test_generate_with_different_models(self, do_sample):
        # Use smaller test models instead
        target_model_checkpoint = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        assistant_checkpoint = "hf-internal-testing/tiny-random-gpt2"

        prompt = "Alice and Bob"

        # Load models sequentially and handle cleanup
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_checkpoint,
        )
        target_tokenizer = AutoTokenizer.from_pretrained(target_model_checkpoint)

        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_checkpoint,
        )
        assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_checkpoint)

        # Tokenize input
        input_ids = target_tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

        # Create generation configs
        base_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=do_sample,
        )

        # Generate with and without assistant model
        outputs_normal = target_model.generate(
            input_ids,
            generation_config=base_config,
        )

        # Pass the assistant model and tokenizers directly to the generate method
        outputs_assisted = target_model.generate(
            input_ids,
            generation_config=base_config,
            assistant_model=assistant_model,
            tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )

        # Decode outputs
        text_normal = target_tokenizer.batch_decode(outputs_normal, skip_special_tokens=True)[0]
        text_assisted = target_tokenizer.batch_decode(outputs_assisted, skip_special_tokens=True)[0]

        # Basic validation
        self.assertIsInstance(text_normal, str)
        self.assertIsInstance(text_assisted, str)
        self.assertGreater(len(text_normal), len(prompt))
        self.assertGreater(len(text_assisted), len(prompt))
        self.assertTrue(text_normal.startswith(prompt))
        self.assertTrue(text_assisted.startswith(prompt))
        if not do_sample:
            self.assertEqual(text_normal, text_assisted)
