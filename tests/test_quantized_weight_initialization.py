import unittest
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class TestQuantizedWeightInitialization(unittest.TestCase):
    """Test that quantized weights are not re-initialized during model loading."""

    def test_int8_weights_skipped(self):
        """Test that int8 weights are skipped during initialization."""

        class TestConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.initializer_range = 0.02

        class TestModel(PreTrainedModel):
            config_class = TestConfig

            def __init__(self, config):
                super().__init__(config)
                self.linear = nn.Linear(10, 10)
                # Simulate quantized weights
                with torch.no_grad():
                    self.linear.weight = nn.Parameter(
                        self.linear.weight.to(torch.int8), requires_grad=False
                    )

        config = TestConfig()
        model = TestModel(config)

        # Store original weight
        original_weight = model.linear.weight.clone()

        # This should not raise an error and should not modify the weight
        model._init_weights(model.linear)

        # Verify weight unchanged and still int8
        self.assertEqual(model.linear.weight.dtype, torch.int8)
        self.assertTrue(torch.equal(model.linear.weight, original_weight))

    def test_float_weights_initialized(self):
        """Test that float weights are still properly initialized."""

        class TestConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.initializer_range = 0.02

        class TestModel(PreTrainedModel):
            config_class = TestConfig

            def __init__(self, config):
                super().__init__(config)
                self.linear = nn.Linear(10, 10)

        config = TestConfig()
        model = TestModel(config)

        # Store original weight
        original_weight = model.linear.weight.clone()

        # Initialize weights
        model._init_weights(model.linear)

        # Verify weight was modified and remains float32
        self.assertEqual(model.linear.weight.dtype, torch.float32)
        self.assertFalse(torch.equal(model.linear.weight, original_weight))
