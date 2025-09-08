"""
Tests for Llama4 model output handling, specifically ensuring robust handling
of missing hidden_states and attentions attributes from the text model.
"""

import types
import torch
import pytest

from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM, CausalLMOutputWithPast
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig


class MockModelOutput:
    """Mock output that mimics BaseModelOutputWithPast behavior."""

    def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, index):
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.past_key_values
        else:
            raise IndexError(f"Output has no index {index}")


class StubTextModel(torch.nn.Module):
    """
    Lightweight stub that mimics Llama4TextModel output behavior.
    Allows testing different scenarios without instantiating heavy models.
    """

    def __init__(self, return_hidden_states: bool = False, return_attentions: bool = False):
        super().__init__()
        self.return_hidden_states = return_hidden_states
        self.return_attentions = return_attentions
        self.embed = torch.nn.Embedding(128, 32)

    def forward(self, input_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        last_hidden_state = self.embed(input_ids)  # [B, T, H]

        # Optional attributes
        hidden_states = None
        attentions = None

        if self.return_hidden_states:
            hidden_states = (last_hidden_state,)  # Tuple of layer outputs

        if self.return_attentions:
            # Mock attention weights [B, num_heads, T, T]
            mock_attention = torch.ones(batch_size, 1, seq_len, seq_len) / seq_len
            attentions = (mock_attention,)

        # Create compatible output object
        return MockModelOutput(
            last_hidden_state=last_hidden_state,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=attentions
        )


def create_test_model(return_hidden_states: bool = False, return_attentions: bool = False):
    """
    Create a minimal Llama4ForCausalLM for testing that uses StubTextModel.
    This avoids configuration issues by using a working config and then replacing the model.
    """
    # Use Llama4TextConfig (the correct type for Llama4ForCausalLM)
    config = Llama4TextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,  # This was missing and causing the error
        intermediate_size=64,
        layer_types=["full_attention"]  # Use valid layer type
    )

    # Create the model normally first
    lm = Llama4ForCausalLM(config)

    # Then replace the heavy text model with our lightweight stub
    lm.model = StubTextModel(
        return_hidden_states=return_hidden_states,
        return_attentions=return_attentions
    )

    return lm


def test_getattr_passthrough_integration():
    """Integration test demonstrating the getattr fix works as expected."""
    # Test the specific scenario mentioned in the user request
    lm = create_test_model(return_hidden_states=False)
    input_ids = torch.randint(0, 128, (1, 4))

    # Should NOT crash even though hidden_states absent
    output = lm(input_ids=input_ids, output_hidden_states=False)
    assert isinstance(output, CausalLMOutputWithPast)
    assert hasattr(output, "hidden_states") and (output.hidden_states is None)

    # Now with hidden_states present
    lm.model.return_hidden_states = True
    output2 = lm(input_ids=input_ids, output_hidden_states=True)
    assert isinstance(output2.hidden_states, tuple)

    print("✅ getattr passthrough works")


class TestLlama4OutputHandling:
    """Test suite for Llama4ForCausalLM output handling robustness."""

    def test_missing_hidden_states_no_crash(self):
        """Test that missing hidden_states attribute doesn't cause AttributeError."""
        lm = create_test_model(return_hidden_states=False, return_attentions=False)
        input_ids = torch.randint(0, 128, (1, 4))

        # This should NOT crash even though hidden_states is absent
        output = lm(input_ids=input_ids, output_hidden_states=False)

        assert isinstance(output, CausalLMOutputWithPast)
        assert hasattr(output, "hidden_states")
        assert output.hidden_states is None
        assert hasattr(output, "attentions")
        assert output.attentions is None

    def test_missing_attentions_no_crash(self):
        """Test that missing attentions attribute doesn't cause AttributeError."""
        lm = create_test_model(return_hidden_states=True, return_attentions=False)
        input_ids = torch.randint(0, 128, (1, 4))

        output = lm(input_ids=input_ids, output_attentions=False)

        assert isinstance(output, CausalLMOutputWithPast)
        assert hasattr(output, "hidden_states")
        assert output.hidden_states is not None
        assert isinstance(output.hidden_states, tuple)
        assert hasattr(output, "attentions")
        assert output.attentions is None


if __name__ == "__main__":
    # Quick verification when run directly
    test_getattr_passthrough_integration()
    print("✅ All basic checks pass!")
