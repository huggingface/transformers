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
#  pytest tests/training_ci/test_cpu_forward.py -s
import gc
import time
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import Qwen2MoeForCausalLM
from transformers.testing_utils import require_torch
from tests.training_ci.logging import logger, init_logger, Colors
from tests.training_ci.metrics import build_cpu_memory_monitor, CPUMemoryMonitor


class TinyTextDataset(Dataset):
    """A tiny synthetic dataset for overfitting tests."""

    def __init__(self, vocab_size: int, seq_length: int = 32, num_samples: int = 8):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        # Create fixed sequences to overfit on
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.data[idx]}


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
        total_steps = 50
        batch_size = 4
        learning_rate = 1e-3
        seq_length = 128
        num_samples = 8
        log_freq = 1

        logger.info("Job Configuration:")
        logger.info(f"  model_name: {model_name}")
        logger.info(f"  total_steps: {total_steps}")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  learning_rate: {learning_rate}")
        logger.info(f"  seq_length: {seq_length}")
        logger.info(f"  num_samples: {num_samples}")
        logger.info(f"  log_freq: {log_freq}")
        logger.info(f"  device: cpu")

        # ============================================================
        # Model Loading
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"Building model: {model_name}")
        load_start = time.perf_counter()

        model = Qwen2MoeForCausalLM.from_pretrained(model_name)

        load_time = time.perf_counter() - load_start
        logger.info(f"Model loaded in {load_time:.3f}s")

        # Log model architecture
        logger.info("Model Architecture:")
        logger.info(f"  hidden_size: {model.config.hidden_size}")
        logger.info(f"  num_hidden_layers: {model.config.num_hidden_layers}")
        logger.info(f"  num_attention_heads: {model.config.num_attention_heads}")
        logger.info(f"  num_key_value_heads: {getattr(model.config, 'num_key_value_heads', 'N/A')}")
        logger.info(f"  intermediate_size: {model.config.intermediate_size}")
        logger.info(f"  vocab_size: {model.config.vocab_size}")
        logger.info(f"  num_experts: {getattr(model.config, 'num_experts', 'N/A')}")
        logger.info(f"  num_experts_per_tok: {getattr(model.config, 'num_experts_per_tok', 'N/A')}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model size: {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Memory after model load
        mem_stats = self.memory_monitor.get_stats()
        logger.info(f"Memory after model load: {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)")

        # ============================================================
        # Dataset & DataLoader
        # ============================================================
        logger.info("-" * 70)
        logger.info("Building dataloader")

        dataset = TinyTextDataset(
            vocab_size=model.config.vocab_size,
            seq_length=seq_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tokens_per_batch = batch_size * seq_length
        total_tokens_per_epoch = num_samples * seq_length

        logger.info(f"Dataset size: {num_samples} samples")
        logger.info(f"Sequence length: {seq_length}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Tokens per batch: {tokens_per_batch:,}")
        logger.info(f"Total tokens per epoch: {total_tokens_per_epoch:,}")
        logger.info(f"Batches per epoch: {len(dataloader)}")

        # ============================================================
        # Optimizer
        # ============================================================
        logger.info("-" * 70)
        logger.info("Building optimizer")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        logger.info(f"Optimizer: AdamW")
        logger.info(f"  learning_rate: {learning_rate}")
        logger.info(f"  betas: (0.9, 0.999)")
        logger.info(f"  weight_decay: 0.01")

        # ============================================================
        # Training Loop
        # ============================================================
        logger.info("-" * 70)
        logger.info(f"Training starts at step 1")

        model.train()
        global_step = 0
        ntokens_seen = 0
        initial_loss = None
        final_loss = None
        training_start = time.perf_counter()
        self.memory_monitor.reset_peak_stats()

        # Create infinite data iterator
        def infinite_dataloader():
            while True:
                for batch in dataloader:
                    yield batch

        data_iter = infinite_dataloader()

        while global_step < total_steps:
            step_start = time.perf_counter()
            global_step += 1

            # Get batch
            batch = next(data_iter)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            batch_tokens = input_ids.numel()
            ntokens_seen += batch_tokens

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

            # Optimizer step
            optimizer.step()

            step_time = time.perf_counter() - step_start

            # Log at frequency
            if global_step == 1 or global_step % log_freq == 0 or global_step == total_steps:
                tokens_per_sec = batch_tokens / step_time
                mem_stats = self.memory_monitor.get_stats()
                logger.info(
                    f"step: {global_step}  "
                    f"loss: {loss.item():7.4f}  "
                    f"grad_norm: {grad_norm.item():6.4f}  "
                    f"memory: {mem_stats.rss_gib:.2f}GiB({mem_stats.rss_pct:.1f}%)  "
                    f"tok/s: {tokens_per_sec:,.0f}  "
                    f"step_time: {step_time:.3f}s"
                )

        training_time = time.perf_counter() - training_start

        # ============================================================
        # Training Summary
        # ============================================================
        logger.info("-" * 70)
        logger.info("Training completed")
        logger.info(f"Total training time: {training_time:.2f}s")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Total tokens seen: {ntokens_seen:,}")
        logger.info(f"Average tokens/sec: {ntokens_seen / training_time:,.0f}")

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

        # ============================================================
        # Validation on Training Data (Overfit Check)
        # ============================================================
        logger.info("-" * 70)
        logger.info("Running validation on training data")

        model.eval()
        eval_loss = 0.0
        num_eval_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                outputs = model(input_ids=input_ids, labels=labels)
                eval_loss += outputs.loss.item()
                num_eval_batches += 1

        avg_eval_loss = eval_loss / num_eval_batches
        logger.info(f"Validation loss: {avg_eval_loss:.4f}")

        # ============================================================
        # Assertions
        # ============================================================
        logger.info("-" * 70)
        logger.info("Running assertions")

        # Assert loss decreased significantly
        self.assertLess(
            final_loss, initial_loss * 0.5,
            f"Expected loss to decrease by at least 50%, got {loss_reduction:.1f}%"
        )
        logger.info("✓ Loss decreased by more than 50%")

        # Assert eval loss is low (model overfit)
        self.assertLess(avg_eval_loss, 5.0, "Eval loss should be low after overfitting")
        logger.info("✓ Eval loss is below threshold")

        logger.info("✓ All assertions passed - model successfully overfit!")


if __name__ == "__main__":
    init_logger()
    logger.info("=" * 70)
    logger.info("Overfit Training CI Tests (CPU only)")
    logger.info("=" * 70)
    logger.info(f"PyTorch version: {torch.__version__}")
    unittest.main(verbosity=2)
