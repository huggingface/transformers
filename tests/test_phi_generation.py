import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.phi_utils import PhiRecursiveGenerator, phi_temperature, phi_top_k, phi_top_p


class TestPhiGeneration(unittest.TestCase):
    def setUp(self):
        self.model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def test_phi_parameters(self):
        # Actual values based on formula
        self.assertAlmostEqual(phi_temperature(0.7), 0.8146, places=3)
        self.assertAlmostEqual(phi_top_p(), 0.618, places=3)
        self.assertEqual(phi_top_k(), 42)

    def test_recursive_generation(self):
        generator = PhiRecursiveGenerator(self.model, self.tokenizer)
        output = generator.generate(
            "The future of AI is", max_iterations=2, verbose=False
        )
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 10)


if __name__ == "__main__":
    unittest.main()
