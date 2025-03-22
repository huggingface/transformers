# test_modeling_arlow.py

import unittest

import torch

from transformers.models.arlow import ArlowConfig, ArlowForCausalLM, ArlowModel
from transformers.testing_utils import require_torch, slow, torch_device


@require_torch
class ArlowModelTester:
    """
    This helper class sets up small configs & sample data to test ArlowModel and ArlowForCausalLM.
    """

    def __init__(
        self,
        batch_size=2,
        seq_length=8,
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        pad_token_id=0,
        use_cache=True,
        tie_word_embeddings=True,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

    def get_config(self):
        return ArlowConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=128,
            pad_token_id=self.pad_token_id,
            use_cache=self.use_cache,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=10000.0,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_length), device=torch_device
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=torch_device)
        # Make half the tokens padded in one row, as an example
        if self.seq_length > 2:
            attention_mask[0, -2:] = 0  # simulate some padding

        labels = input_ids.clone()
        # Replace some tokens with pad_token_id
        labels[0, -1] = config.pad_token_id

        return config, input_ids, attention_mask, labels

    def create_and_check_model(self, config, input_ids, attention_mask, labels):
        """
        Test ArlowModel forward pass. We only verify that it runs and
        outputs the correct shape.
        """
        model = ArlowModel(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        # Check shape
        assert outputs.last_hidden_state.shape == (
            self.batch_size,
            self.seq_length,
            config.hidden_size,
        )

    def create_and_check_causal_lm(self, config, input_ids, attention_mask, labels):
        """
        Test ArlowForCausalLM forward pass with and without labels,
        verifying shape and that we get a loss when labels are provided.
        """
        model = ArlowForCausalLM(config).to(torch_device)
        model.eval()

        # Without labels
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits: (batch_size, seq_length, vocab_size)
        assert outputs.logits.shape == (
            self.batch_size,
            self.seq_length,
            config.vocab_size,
        )
        assert outputs.loss is None

        # With labels
        with torch.no_grad():
            outputs_with_labels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        assert outputs_with_labels.logits.shape == (
            self.batch_size,
            self.seq_length,
            config.vocab_size,
        )
        assert outputs_with_labels.loss is not None
        assert torch.isfinite(outputs_with_labels.loss), "Loss is not finite."

    def create_and_check_generation(self, config, input_ids, attention_mask):
        """
        If you want to test generation, do so here.
        """
        model = ArlowForCausalLM(config).to(torch_device)
        model.eval()

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.seq_length + 5,
            num_beams=1,
            do_sample=False,
        )
        # shape => (batch_size, new_seq_len)
        assert generated.shape[0] == input_ids.shape[0]
        assert generated.shape[1] > self.seq_length


@require_torch
class ArlowModelTest(unittest.TestCase):
    """
    Main test class for ArlowModel and ArlowForCausalLM.
    """

    # Hugging Face test suite expects all_model_classes to be defined:
    all_model_classes = (ArlowModel, ArlowForCausalLM)

    def setUp(self):
        self.model_tester = ArlowModelTester()

    def test_model(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, attention_mask, labels)

    def test_model_for_causal_lm(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm(config, input_ids, attention_mask, labels)

    @slow
    def test_generation(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generation(config, input_ids, attention_mask)


if __name__ == "__main__":
    unittest.main()
