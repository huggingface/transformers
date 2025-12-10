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
import os
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

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.distributed as dist

logger = logging.getLogger("transformers.training_test")
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

    set_seed(42)

    if is_rank_0:
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building model{Colors.RESET}")

    load_start = time.perf_counter()

    # Reconstruct config and model from passed classes
    config = config_class.from_dict(training_params['config_dict'])
    model = model_class(config)
    model.train()

class TrainingDistributedTesterMixin(ABC):
    """
    Mixin for training overfit tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - prepare_config_and_inputs_for_common() -> config + input dict
      - causal_lm_class, base_model_class, etc.

    This mixin adds training-specific tests using that infrastructure.
    """

    # ============================================================
    # Training hyperparameters
    # ============================================================
    training_overfit_steps: int = 300
    training_overfit_batch_size: int = 2
    training_overfit_learning_rate: float = 1e-3
    training_overfit_seq_length: int = 64
    training_overfit_log_freq: int = 10

    # Loss reduction and grad norm reduction thresholds for passing the test (i.e 95% reduction)
    training_loss_reduction_threshold: float = 0.9
    training_grad_norm_reduction_threshold: float = 0.9

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


    # @is_training_distributed_test
    # def test_training_fsdp1_tp4(self):
    #     """Test distributed training with FSDP=1, TP=4 (4 total processes)."""
    #     self._run_distributed_training_test(fsdp_size=1, tp_size=4)
