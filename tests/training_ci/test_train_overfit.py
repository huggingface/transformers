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

"""Training CI tests - overfit on a small dataset with detailed logging (CPU only)."""
#  pytest tests/training_ci/train_overfit.py -s
import gc
import time
import unittest

import pytest
import torch

from transformers import AutoTokenizer, Qwen2MoeForCausalLM
from transformers.testing_utils import require_torch
from tests.training_ci.logging import logger, init_logger, Colors
from tests.training_ci.metrics import build_cpu_memory_monitor, CPUMemoryMonitor


# Hardcoded sentence for deterministic overfitting
TRAINING_SENTENCE = "This is a training sample"


def create_fixed_batch(
    tokenizer,
    batch_size: int,
    seq_length: int,
) -> dict[str, torch.Tensor]:
    """Create a single fixed batch by tokenizing and repeating a hardcoded sentence.
    
    The sentence "This is a training sample" is tokenized, then repeated to fill
    the desired seq_length. The resulting sequence is then stacked to fill batch_size.
    
    Args:
        tokenizer: The tokenizer to use for encoding the sentence.
        batch_size: Number of sequences in the batch.
        seq_length: Length of each sequence.
    
    Returns:
        A dict with "input_ids" and "labels" tensors of shape (batch_size, seq_length).
    """
    # Tokenize the sentence (without special tokens for clean repetition)
    tokens = tokenizer.encode(TRAINING_SENTENCE, add_special_tokens=False)
    
    # Repeat tokens to fill seq_length
    num_repeats = (seq_length // len(tokens)) + 1
    repeated_tokens = (tokens * num_repeats)[:seq_length]
    
    # Create batch by stacking the same sequence batch_size times
    input_ids = torch.tensor([repeated_tokens] * batch_size, dtype=torch.long)
    
    return {"input_ids": input_ids, "labels": input_ids.clone()}


@pytest.mark.training_ci
@require_torch
class TestOverfitTraining(unittest.TestCase):
    """Test training loop can overfit on a tiny dataset (CPU only)."""

    memory_monitor: CPUMemoryMonitor

    def setUp(self):
        init_logger()
        self.memory_monitor = build_cpu_memory_monitor()
        logger.info("=" * 70)
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info("=" * 70)

    def tearDown(self):
        logger.info(f"Finished test: {self._testMethodName}")
        logger.info("=" * 70)

    def test_qwen2_moe_overfit(self):
        """Test overfitting on tiny-random-Qwen2MoeForCausalLM."""

        # ============================================================
        # Configuration
        # ============================================================
        model_name = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
        total_steps = 200
        batch_size = 2
        learning_rate = 1e-3
        seq_length = 64
        log_freq = 10

        logger.info("Job Configuration:")
        logger.info(f"  model_name: {model_name}")
        logger.info(f"  total_steps: {total_steps}")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  learning_rate: {learning_rate}")
        logger.info(f"  seq_length: {seq_length}")
        logger.info(f"  log_freq: {log_freq}")
        logger.info(f"  device: cpu")

        # ============================================================
        # Determinism
        # ============================================================
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ============================================================
        # Model & Tokenizer Loading
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building model:{Colors.RESET} {model_name}")
        load_start = time.perf_counter()

        model = Qwen2MoeForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_time = time.perf_counter() - load_start
        logger.info(f"Model and tokenizer loaded in {Colors.GREEN}{load_time:.3f}s{Colors.RESET}")

        # Log model architecture
        logger.info(f"{Colors.BOLD}Model Architecture:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}hidden_size:{Colors.RESET} {model.config.hidden_size}")
        logger.info(f"  {Colors.CYAN}num_hidden_layers:{Colors.RESET} {model.config.num_hidden_layers}")
        logger.info(f"  {Colors.CYAN}num_attention_heads:{Colors.RESET} {model.config.num_attention_heads}")
        logger.info(f"  {Colors.CYAN}num_key_value_heads:{Colors.RESET} {getattr(model.config, 'num_key_value_heads', 'N/A')}")
        logger.info(f"  {Colors.CYAN}intermediate_size:{Colors.RESET} {model.config.intermediate_size}")
        logger.info(f"  {Colors.CYAN}vocab_size:{Colors.RESET} {model.config.vocab_size}")
        logger.info(f"  {Colors.CYAN}num_experts:{Colors.RESET} {getattr(model.config, 'num_experts', 'N/A')}")
        logger.info(f"  {Colors.CYAN}num_experts_per_tok:{Colors.RESET} {getattr(model.config, 'num_experts_per_tok', 'N/A')}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{Colors.CYAN}Model size:{Colors.RESET} {Colors.BRIGHT_GREEN}{total_params:,}{Colors.RESET} total parameters")
        logger.info(f"{Colors.CYAN}Trainable parameters:{Colors.RESET} {Colors.BRIGHT_GREEN}{trainable_params:,}{Colors.RESET}")

        # Memory after model load
        mem_stats = self.memory_monitor.get_stats()
        logger.info(f"{Colors.MAGENTA}Memory after model load:{Colors.RESET} {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)")

        # ============================================================
        # Fixed Batch (same batch every step for deterministic overfitting)
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Creating fixed batch{Colors.RESET}")
        logger.info(f"{Colors.CYAN}Training sentence:{Colors.RESET} \"{TRAINING_SENTENCE}\"")

        fixed_batch = create_fixed_batch(
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
        )

        tokens_per_batch = batch_size * seq_length

        logger.info(f"  {Colors.CYAN}batch_size:{Colors.RESET} {batch_size}")
        logger.info(f"  {Colors.CYAN}seq_length:{Colors.RESET} {seq_length}")
        logger.info(f"  {Colors.CYAN}tokens_per_batch:{Colors.RESET} {tokens_per_batch:,}")
        logger.info(f"{Colors.DIM}Using same fixed batch every step (deterministic overfitting){Colors.RESET}")

        # ============================================================
        # Optimizer
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building optimizer{Colors.RESET}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999))
        logger.info(f"{Colors.CYAN}Optimizer:{Colors.RESET} Adam")
        logger.info(f"  {Colors.CYAN}learning_rate:{Colors.RESET} {learning_rate}")
        logger.info(f"  {Colors.CYAN}weight_decay:{Colors.RESET} 0.0")
        logger.info(f"  {Colors.CYAN}betas:{Colors.RESET} (0.9, 0.999)")

        # ============================================================
        # Training Loop
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"Training starts at step 1")

        model.train()
        global_step = 0
        initial_loss = None
        final_loss = None
        initial_grad_norm = None
        final_grad_norm = None
        training_start = time.perf_counter()
        self.memory_monitor.reset_peak_stats()

        # Use fixed batch directly (same batch every step)
        input_ids = fixed_batch["input_ids"]
        labels = fixed_batch["labels"]
        batch_tokens = input_ids.numel()

        while global_step < total_steps:
            step_start = time.perf_counter()
            global_step += 1

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Track initial and final loss
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Track initial and final grad_norm
            if initial_grad_norm is None:
                initial_grad_norm = grad_norm.item()
            final_grad_norm = grad_norm.item()

            # Optimizer step
            optimizer.step()

            step_time = time.perf_counter() - step_start

            # Log at frequency
            if global_step == 1 or global_step % log_freq == 0 or global_step == total_steps:
                tokens_per_sec = batch_tokens / step_time
                mem_stats = self.memory_monitor.get_stats()
                logger.info(
                    f"{Colors.CYAN}step:{Colors.RESET} {global_step}  "
                    f"{Colors.GREEN}loss:{Colors.RESET} {loss.item():7.4f}  "
                    f"{Colors.YELLOW}grad_norm:{Colors.RESET} {grad_norm.item():6.4f}  "
                    f"{Colors.MAGENTA}memory:{Colors.RESET} {mem_stats.rss_gib:.2f}GiB({mem_stats.rss_pct:.1f}%)  "
                    f"{Colors.BLUE}tok/s:{Colors.RESET} {tokens_per_sec:,.0f}  "
                    f"{Colors.DIM}step_time:{Colors.RESET} {step_time:.3f}s"
                )

        training_time = time.perf_counter() - training_start

        # ============================================================
        # Training Summary
        # ============================================================
        total_tokens = global_step * batch_tokens
        logger.info("-" * 70)
        logger.info("Training completed")
        logger.info(f"Total training time: {training_time:.2f}s")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Total tokens seen: {total_tokens:,}")
        logger.info(f"Average tokens/sec: {total_tokens / training_time:,.0f}")

        # Memory summary
        mem_stats = self.memory_monitor.get_stats()
        logger.info("Memory usage:")
        logger.info(f"  current_rss: {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)")
        logger.info(f"  peak_rss: {mem_stats.peak_rss_gib:.2f} GiB ({mem_stats.peak_rss_pct:.1f}%)")
        logger.info(f"  available: {mem_stats.available_gib:.2f} GiB / {mem_stats.total_gib:.2f} GiB")

        # Loss analysis
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100

        logger.info("Loss metrics:")
        logger.info(f"  initial_loss: {initial_loss:.4f}")
        logger.info(f"  final_loss: {final_loss:.4f}")
        logger.info(f"  loss_reduction: {loss_reduction:.1f}%")

        # Grad norm analysis
        grad_norm_reduction = (initial_grad_norm - final_grad_norm) / initial_grad_norm * 100

        logger.info("Grad norm metrics:")
        logger.info(f"  initial_grad_norm: {initial_grad_norm:.4f}")
        logger.info(f"  final_grad_norm: {final_grad_norm:.4f}")
        logger.info(f"  grad_norm_reduction: {grad_norm_reduction:.1f}%")


        # ============================================================
        # Assertions
        # ============================================================
        logger.info("-" * 70)
        logger.info("Running assertions")

        # Assert loss decreased significantly
        self.assertLess(
            final_loss, initial_loss * 0.01,
            f"Expected loss to decrease by at least 99%, got {loss_reduction:.1f}%"
        )
        logger.info("✓ Loss decreased by more than 99%")

        # Assert grad_norm decreased significantly
        self.assertLess(
            final_grad_norm, initial_grad_norm * 0.02,
            f"Expected grad_norm to decrease by at least 98%, got {grad_norm_reduction:.1f}%"
        )
        logger.info("✓ Grad norm decreased by more than 98%")

if __name__ == "__main__":
    init_logger()
    logger.info("=" * 70)
    logger.info("Overfit Training CI Tests (CPU only)")
    logger.info("=" * 70)
    logger.info(f"PyTorch version: {torch.__version__}")
    unittest.main(verbosity=2)
