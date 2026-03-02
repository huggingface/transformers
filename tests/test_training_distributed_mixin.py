# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Training overfit tester mixin for model tests."""

import logging
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Optional

from transformers import is_torch_available, set_seed
from transformers.testing_utils import (
    Colors,
    build_cpu_memory_monitor,
    init_distributed,
    init_test_logger,
    is_training_distributed_test,
)

from .test_training_mixin import TrainingConfigMixin

if is_torch_available():
    import torch
    import torch.distributed as dist

logger = logging.getLogger("transformers.training_test")


def _create_text_training_batch(batch_size: int, seq_length: int, vocab_size: int) -> dict:
    """Create a simple text batch without needing a tokenizer.
    
    Standalone function for use in distributed spawned processes.
    """
    pattern = list(range(1, min(20, vocab_size)))  # tokens 1-19
    num_repeats = (seq_length // len(pattern)) + 1
    tokens = (pattern * num_repeats)[:seq_length]
    input_ids = torch.tensor([tokens] * batch_size, dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _test_training_distributed_overfit_impl(mesh, config_class, model_class, training_params):
    """Implementation for distributed training overfit test.
    
    Note: `mesh` is automatically created and passed by `global_wrapper` in testing_utils.py.
    
    Args:
        mesh: DeviceMesh created by global_wrapper
        config_class: The config class (e.g., LlamaConfig)
        model_class: The model class (e.g., LlamaForCausalLM)
        training_params: Dict with 'config_dict', 'steps', 'batch_size', 'learning_rate', 'seq_length', 'log_freq'
    """
    init_test_logger()
    is_rank_0 = dist.get_rank() == 0
    tp_size = mesh["tp"].size()

    if is_rank_0:
        logger.info(f"Created DeviceMesh: {mesh}")
        logger.info(f"FSDP mesh: {mesh['fsdp']}")
        logger.info(f"TP mesh: {mesh['tp']}")
        logger.info(f"FSDP mesh local rank: {mesh['fsdp'].get_local_rank()}")
        logger.info(f"TP mesh local rank: {mesh['tp'].get_local_rank()}")
    dist.barrier()

    memory_monitor = build_cpu_memory_monitor(logger)

    if is_rank_0:
        logger.info("=" * 70)
        logger.info("Starting distributed training overfit test")
        logger.info("=" * 70)

        # Configuration
        logger.info(f"{Colors.BOLD}Job Configuration:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}total_steps:{Colors.RESET} {training_params['steps']}")
        logger.info(f"  {Colors.CYAN}batch_size:{Colors.RESET} {training_params['batch_size']}")
        logger.info(f"  {Colors.CYAN}learning_rate:{Colors.RESET} {training_params['learning_rate']}")
        logger.info(f"  {Colors.CYAN}seq_length:{Colors.RESET} {training_params['seq_length']}")
        logger.info(f"  {Colors.CYAN}log_freq:{Colors.RESET} {training_params['log_freq']}")
        logger.info(f"  {Colors.CYAN}device:{Colors.RESET} cpu")
        logger.info(f"  {Colors.CYAN}tp_size:{Colors.RESET} {tp_size}")

    set_seed(42)

    if is_rank_0:
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building model with Tensor Parallelism{Colors.RESET}")

    load_start = time.perf_counter()

    # Reconstruct config from passed config class
    config = config_class.from_dict(training_params['config_dict'])

    # NOTE(3outeille): Need to figure out how to do it natively when calling tp_plan="auto"
    # Create a shared temp directory for model saving/loading
    # Only rank 0 creates and saves the model, all ranks load with TP
    temp_dir = tempfile.mkdtemp()
    
    # Broadcast the temp_dir path to all ranks
    if is_rank_0:
        temp_dir_bytes = temp_dir.encode('utf-8')
        temp_dir_tensor = torch.tensor(list(temp_dir_bytes), dtype=torch.uint8)
        temp_dir_len = torch.tensor([len(temp_dir_bytes)], dtype=torch.long)
    else:
        temp_dir_len = torch.tensor([0], dtype=torch.long)
    
    dist.broadcast(temp_dir_len, src=0)
    
    if not is_rank_0:
        temp_dir_tensor = torch.zeros(temp_dir_len.item(), dtype=torch.uint8)
    
    dist.broadcast(temp_dir_tensor, src=0)
    temp_dir = bytes(temp_dir_tensor.tolist()).decode('utf-8')

    # Rank 0 creates and saves the model
    if is_rank_0:
        logger.info(f"Creating base model and saving to temp directory: {temp_dir}")
        base_model = model_class(config)
        base_model.save_pretrained(temp_dir)
        del base_model  # Free memory
        logger.info("Base model saved successfully")

    dist.barrier()

    # All ranks load with tensor parallelism
    if is_rank_0:
        logger.info(f"Loading model with tp_plan='auto' and device_mesh")
        if hasattr(config, "base_model_tp_plan"):
            logger.info(f"  {Colors.CYAN}base_model_tp_plan:{Colors.RESET} {config.base_model_tp_plan}")
    
    # Load with tensor parallelism using the TP mesh
    model = model_class.from_pretrained(
        temp_dir,
        tp_plan="auto",
        device_mesh=mesh["tp"],
    )
    
    model.train()

    load_time = time.perf_counter() - load_start
    if is_rank_0:
        logger.info(f"Model loaded in {Colors.GREEN}{load_time:.3f}s{Colors.RESET}")

        # Log model architecture
        logger.info(f"{Colors.BOLD}Model Architecture:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}model_class:{Colors.RESET} {model_class.__name__}")
        if hasattr(config, "hidden_size"):
            logger.info(f"  {Colors.CYAN}hidden_size:{Colors.RESET} {config.hidden_size}")
        if hasattr(config, "num_hidden_layers"):
            logger.info(f"  {Colors.CYAN}num_hidden_layers:{Colors.RESET} {config.num_hidden_layers}")
        if hasattr(config, "num_attention_heads"):
            logger.info(f"  {Colors.CYAN}num_attention_heads:{Colors.RESET} {config.num_attention_heads}")
        if hasattr(config, "num_key_value_heads"):
            logger.info(f"  {Colors.CYAN}num_key_value_heads:{Colors.RESET} {config.num_key_value_heads}")
        if hasattr(config, "intermediate_size"):
            logger.info(f"  {Colors.CYAN}intermediate_size:{Colors.RESET} {config.intermediate_size}")
        if hasattr(config, "vocab_size"):
            logger.info(f"  {Colors.CYAN}vocab_size:{Colors.RESET} {config.vocab_size}")
        if hasattr(config, "num_experts"):
            logger.info(f"  {Colors.CYAN}num_experts:{Colors.RESET} {config.num_experts}")
        if hasattr(config, "num_experts_per_tok"):
            logger.info(f"  {Colors.CYAN}num_experts_per_tok:{Colors.RESET} {config.num_experts_per_tok}")
        
        # Log TP status
        logger.info(f"  {Colors.GREEN}tensor_parallel:{Colors.RESET} ENABLED (tp_size={tp_size})")

        # Count parameters (local parameters for this rank)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"{Colors.CYAN}Model size (local):{Colors.RESET} {Colors.BRIGHT_GREEN}{total_params:,}{Colors.RESET} parameters"
        )
        logger.info(
            f"{Colors.CYAN}Trainable parameters (local):{Colors.RESET} {Colors.BRIGHT_GREEN}{trainable_params:,}{Colors.RESET}"
        )

        # Memory after model load
        mem_stats = memory_monitor.get_stats()
        logger.info(
            f"{Colors.MAGENTA}Memory after model load:{Colors.RESET} {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)"
        )

    dist.barrier()

    # Create fixed batch
    if is_rank_0:
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Creating fixed batch{Colors.RESET}")

    batch = _create_text_training_batch(
        batch_size=training_params['batch_size'],
        seq_length=training_params['seq_length'],
        vocab_size=config.vocab_size,
    )
    tokens_per_batch = training_params['batch_size'] * training_params['seq_length']

    if is_rank_0:
        logger.info(f"{Colors.CYAN}Training pattern:{Colors.RESET} Repeating token sequence (1-19)")
        logger.info(f"  {Colors.CYAN}batch_size:{Colors.RESET} {training_params['batch_size']}")
        logger.info(f"  {Colors.CYAN}seq_length:{Colors.RESET} {training_params['seq_length']}")
        logger.info(f"  {Colors.CYAN}tokens_per_batch:{Colors.RESET} {tokens_per_batch:,}")
        logger.info(f"{Colors.DIM}Using same fixed batch every step (deterministic overfitting){Colors.RESET}")

    # Build optimizer
    if is_rank_0:
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building optimizer{Colors.RESET}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_params['learning_rate'], weight_decay=0.0, betas=(0.9, 0.999)
    )

    if is_rank_0:
        logger.info(f"{Colors.CYAN}Optimizer:{Colors.RESET} Adam")
        logger.info(f"  {Colors.CYAN}learning_rate:{Colors.RESET} {training_params['learning_rate']}")
        logger.info(f"  {Colors.CYAN}weight_decay:{Colors.RESET} 0.0")
        logger.info(f"  {Colors.CYAN}betas:{Colors.RESET} (0.9, 0.999)")

    # Training Loop
    if is_rank_0:
        logger.info("-" * 70)
        logger.info("Training starts at step 1")

    initial_loss = None
    final_loss = None
    initial_grad_norm = None
    final_grad_norm = None
    training_start = time.perf_counter()
    memory_monitor.reset_peak_stats()

    steps = training_params['steps']
    log_freq = training_params['log_freq']

    for step in range(1, steps + 1):
        step_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if initial_grad_norm is None:
            initial_grad_norm = grad_norm.item()
        final_grad_norm = grad_norm.item()

        optimizer.step()

        step_time = time.perf_counter() - step_start

        # Log at frequency
        if is_rank_0 and (step == 1 or step % log_freq == 0 or step == steps):
            tokens_per_sec = tokens_per_batch / step_time
            mem_stats = memory_monitor.get_stats()
            logger.info(
                f"{Colors.CYAN}step:{Colors.RESET} {step}  "
                f"{Colors.GREEN}loss:{Colors.RESET} {loss.item():7.4f}  "
                f"{Colors.YELLOW}grad_norm:{Colors.RESET} {grad_norm.item():6.4f}  "
                f"{Colors.MAGENTA}memory:{Colors.RESET} {mem_stats.rss_gib:.2f}GiB({mem_stats.rss_pct:.1f}%)  "
                f"{Colors.BLUE}tok/s:{Colors.RESET} {tokens_per_sec:,.0f}  "
                f"{Colors.DIM}step_time:{Colors.RESET} {step_time:.3f}s"
            )

    training_time = time.perf_counter() - training_start

    # Training Summary
    if is_rank_0:
        total_tokens = steps * tokens_per_batch
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Training completed{Colors.RESET}")
        logger.info(f"Total training time: {training_time:.2f}s")
        logger.info(f"Total steps: {steps}")
        logger.info(f"Total tokens seen: {total_tokens:,}")
        logger.info(f"Average tokens/sec: {total_tokens / training_time:,.0f}")

        # Memory summary
        mem_stats = memory_monitor.get_stats()
        logger.info(f"{Colors.BOLD}Memory usage:{Colors.RESET}")
        logger.info(
            f"  {Colors.CYAN}current_rss:{Colors.RESET} {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)"
        )
        logger.info(
            f"  {Colors.CYAN}peak_rss:{Colors.RESET} {mem_stats.peak_rss_gib:.2f} GiB ({mem_stats.peak_rss_pct:.1f}%)"
        )
        logger.info(
            f"  {Colors.CYAN}available:{Colors.RESET} {mem_stats.available_gib:.2f} GiB / {mem_stats.total_gib:.2f} GiB"
        )

        # Loss analysis
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        logger.info(f"{Colors.BOLD}Loss metrics:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}initial_loss:{Colors.RESET} {initial_loss:.4f}")
        logger.info(f"  {Colors.CYAN}final_loss:{Colors.RESET} {final_loss:.4f}")
        logger.info(f"  {Colors.CYAN}loss_reduction:{Colors.RESET} {loss_reduction:.1f}%")

        # Grad norm analysis
        grad_norm_reduction = (initial_grad_norm - final_grad_norm) / initial_grad_norm * 100
        logger.info(f"{Colors.BOLD}Grad norm metrics:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}initial_grad_norm:{Colors.RESET} {initial_grad_norm:.4f}")
        logger.info(f"  {Colors.CYAN}final_grad_norm:{Colors.RESET} {final_grad_norm:.4f}")
        logger.info(f"  {Colors.CYAN}grad_norm_reduction:{Colors.RESET} {grad_norm_reduction:.1f}%")

    # Assertions (run on all ranks for consistency, but only rank 0 logs)
    dist.barrier()

    # Assert loss decreased significantly
    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
    loss_reduction_threshold = 0.9  # 90% reduction
    assert loss_reduction_ratio > loss_reduction_threshold, (
        f"Expected loss to decrease by at least {loss_reduction_threshold * 100:.0f}%, "
        f"got {loss_reduction_ratio * 100:.1f}%"
    )

    # Assert grad_norm decreased significantly
    grad_norm_reduction_ratio = (initial_grad_norm - final_grad_norm) / initial_grad_norm
    grad_norm_reduction_threshold = 0.9  # 90% reduction
    assert grad_norm_reduction_ratio > grad_norm_reduction_threshold, (
        f"Expected grad_norm to decrease by at least {grad_norm_reduction_threshold * 100:.0f}%, "
        f"got {grad_norm_reduction_ratio * 100:.1f}%"
    )

    if is_rank_0:
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Running assertions{Colors.RESET}")
        logger.info(
            f"{Colors.GREEN}✓ Loss decreased by more than {loss_reduction_threshold * 100:.0f}%{Colors.RESET}"
        )
        logger.info(
            f"{Colors.GREEN}✓ Grad norm decreased by more than {grad_norm_reduction_threshold * 100:.0f}%{Colors.RESET}"
        )
        logger.info("=" * 70)
        logger.info("Finished distributed training overfit test")
        logger.info("=" * 70)

    dist.barrier()

    # Cleanup temp directory
    if is_rank_0:
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


class TrainingDistributedTesterMixin(TrainingConfigMixin, ABC):
    """
    Mixin for distributed training overfit tests with Tensor Parallelism.
    Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - prepare_config_and_inputs_for_common() -> config + input dict
      - causal_lm_class, base_model_class, etc.

    This mixin adds distributed training-specific tests using that infrastructure.
    
    Note: Base training hyperparameters are inherited from TrainingConfigMixin.
    We override some values here for faster distributed tests.
    """

    # Override for faster distributed tests
    training_overfit_steps: int = 5
    training_overfit_log_freq: int = 1

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Modality detection
    # ============================================================
    def _get_model_modality(self) -> str:
        """Detect the modality of the model based on its input signature."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        if "input_ids" in inputs_dict:
            return "text"
        elif "pixel_values" in inputs_dict:
            return "image"
        elif "input_features" in inputs_dict or "input_values" in inputs_dict:
            return "audio"
        else:
            raise ValueError(f"Unknown modality: {inputs_dict}")

    # ============================================================
    # Training data creation for each modality
    # ============================================================
    def _create_text_training_batch(
        self,
        batch_size: int,
        seq_length: int,
        vocab_size: int,
    ) -> dict[str, torch.Tensor]:
        """Create a simple text batch without needing a tokenizer."""
        # Create a deterministic sequence (not random, so model can learn it)
        pattern = list(range(1, min(20, vocab_size)))  # tokens 1-19
        num_repeats = (seq_length // len(pattern)) + 1
        tokens = (pattern * num_repeats)[:seq_length]
        input_ids = torch.tensor([tokens] * batch_size, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def _create_image_training_batch(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Create fixed batch for image models using a deterministic pattern."""
        pass

    def _create_audio_training_batch(
        self,
        batch_size: int,
        audio_length: int,
        feature_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Create fixed batch for audio models using a deterministic waveform."""
        pass

    def _decode_text_tokens(self, tokens: list[int], max_display: int = 40) -> str:
        """Decode tokens to readable string (maps token IDs to letters: 1->a, 2->b, etc.)."""
        decoded = "".join(chr(ord("a") + (t - 1) % 26) for t in tokens)
        if len(decoded) > max_display:
            return f"'{decoded[:max_display]}...'"
        return f"'{decoded}'"

    def _get_trainable_model_class(self):
        """Get the model class to use for training (prefers *ForCausalLM, *ForSequenceClassification, etc.)."""
        # Prefer model classes with a head (for computing loss)
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        if (
            hasattr(self.model_tester, "sequence_classification_class")
            and self.model_tester.sequence_classification_class is not None
        ):
            return self.model_tester.sequence_classification_class
        # Fall back to first model class
        return self.all_model_classes[0]

    # ============================================================
    # Shared distributed training test implementation
    # ============================================================
    def _run_distributed_training_test(self, fsdp_size: int, tp_size: int):
        """Shared implementation for distributed training tests."""
        config = self.model_tester.get_config()
        model_class = self._get_trainable_model_class()
        config_class = type(config)

        training_params = {
            "config_dict": config.to_dict(),
            "steps": self.training_overfit_steps,
            "batch_size": self.training_overfit_batch_size,
            "learning_rate": self.training_overfit_learning_rate,
            "seq_length": self.training_overfit_seq_length,
            "log_freq": self.training_overfit_log_freq,
        }

        init_distributed(fsdp_size=fsdp_size, tp_size=tp_size)(_test_training_distributed_overfit_impl)(
            config_class, model_class, training_params
        )

    # ============================================================
    # Distributed training tests (FSDP x TP configurations)
    # ============================================================
    # @is_training_distributed_test
    # def test_training_fsdp1_tp1(self):
    #     """Test distributed training with FSDP=1, TP=1 (1 total processes)."""
    #     self._run_distributed_training_test(fsdp_size=1, tp_size=1)

    @is_training_distributed_test
    def test_training_fsdp1_tp2(self):
        """Test distributed training with FSDP=1, TP=2 (2 total processes)."""
        self._run_distributed_training_test(fsdp_size=1, tp_size=2)

    # def test_training_fsdp2_tp1(self):
    #     "Test distributed training with FSDP=2, TP=1 (2 total processes)."
    #     self._run_distributed_training_test(fsdp_size=2, tp_size=1)

    # @is_training_distributed_test
    # def test_training_fsdp1_tp4(self):
    #     """Test distributed training with FSDP=1, TP=4 (4 total processes)."""
    #     self._run_distributed_training_test(fsdp_size=1, tp_size=4)
