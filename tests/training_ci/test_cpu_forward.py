# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""CPU-only training CI tests."""
#  pytest tests/training_ci/test_cpu_forward.py -s
import sys
import time
import unittest

import torch

from transformers import Qwen2MoeForCausalLM
from transformers.testing_utils import require_torch
from tests.training_ci.logging import logger, init_logger


@require_torch
class TestCPUForward(unittest.TestCase):
    """Test forward pass on CPU for tiny random models."""

    def setUp(self):
        init_logger()
        logger.info("=" * 60)
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info("=" * 60)

    def tearDown(self):
        logger.info(f"Finished test: {self._testMethodName}")
        logger.info("-" * 60)

    def test_qwen2_moe_forward(self):
        """Test forward pass on tiny-random-Qwen2MoeForCausalLM."""
        model_name = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"

        # Load model
        logger.info(f"Loading model: {model_name}")
        start_time = time.perf_counter()
        model = Qwen2MoeForCausalLM.from_pretrained(model_name)
        load_time = time.perf_counter() - start_time
        logger.info(f"Model loaded successfully in {load_time:.3f}s")

        # Log model configuration
        logger.info("Model configuration:")
        logger.info(f"  - Hidden size: {model.config.hidden_size}")
        logger.info(f"  - Num layers: {model.config.num_hidden_layers}")
        logger.info(f"  - Num attention heads: {model.config.num_attention_heads}")
        logger.info(f"  - Vocab size: {model.config.vocab_size}")
        logger.info(f"  - Num experts: {getattr(model.config, 'num_experts', 'N/A')}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")

        # Set eval mode
        logger.info("Setting model to eval mode")
        model.eval()

        # Create input
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        logger.info(f"Created input tensor:")
        logger.info(f"  - Shape: {input_ids.shape}")
        logger.info(f"  - Dtype: {input_ids.dtype}")
        logger.info(f"  - Device: {input_ids.device}")
        logger.info(f"  - Values: {input_ids.tolist()}")

        # Run forward pass
        logger.info("Running forward pass...")
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(input_ids)
        forward_time = time.perf_counter() - start_time
        logger.info(f"Forward pass completed in {forward_time:.3f}s")

        # Log output details
        logger.info("Output details:")
        logger.info(f"  - Logits shape: {output.logits.shape}")
        logger.info(f"  - Logits dtype: {output.logits.dtype}")
        logger.info(f"  - Logits min: {output.logits.min().item():.4f}")
        logger.info(f"  - Logits max: {output.logits.max().item():.4f}")
        logger.info(f"  - Logits mean: {output.logits.mean().item():.4f}")

        # Verify output shape
        vocab_size = model.config.vocab_size
        expected_shape = torch.Size((1, 6, vocab_size))
        logger.info(f"Verifying output shape:")
        logger.info(f"  - Expected: {expected_shape}")
        logger.info(f"  - Actual: {output.logits.shape}")

        self.assertEqual(output.logits.shape, expected_shape)
        logger.info("✓ Shape verification passed!")


if __name__ == "__main__":
    logger.info("Starting CPU Forward Pass Tests")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    unittest.main(verbosity=2)
