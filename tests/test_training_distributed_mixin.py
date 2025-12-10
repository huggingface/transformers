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
    get_torch_dist_unique_port,
    init_test_logger,
    is_training_distributed_test,
)


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


def global_wrapper(rank, func, tp, port, func_args, func_kwargs):
    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def init_distributed(tp: int):
    def _init_distributed(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            port = get_torch_dist_unique_port()
            spawn_args = (func, tp, port, args, kwargs)
            mp.spawn(global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed


logger = logging.getLogger("transformers.training_test")


# Standalone implementation function (outside the class) - this CAN be pickled
def _test_training_distributed_overfit_impl(rank, config_dict, model_class_name, training_params):
    """Implementation for distributed training overfit test."""
    init_test_logger()
    logger.info(f"Starting test on rank {rank}")
    logger.info(f"World size: {dist.get_world_size()}")
    logger.info(f"Rank: {dist.get_rank()}")
    
    # Reconstruct config and model from picklable data
    # ... your training logic here using the passed parameters ...
    
    dist.barrier()


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

    @is_training_distributed_test
    def test_training_distributed_overfit(self):
        """Test that a tiny model can overfit on a fixed batch."""
        # Extract all needed data into picklable objects BEFORE spawning
        config = self.model_tester.get_config()
        model_class = self._get_trainable_model_class()
        
        # Prepare picklable arguments (dicts, strings, primitives - NOT self)
        config_dict = config.to_dict()
        model_class_name = model_class.__name__
        training_params = {
            "steps": self.training_overfit_steps,
            "batch_size": self.training_overfit_batch_size,
            "learning_rate": self.training_overfit_learning_rate,
            "seq_length": self.training_overfit_seq_length,
        }
        
        # Call the standalone function with the decorator
        init_distributed(tp=2)(_test_training_distributed_overfit_impl)(
            config_dict, model_class_name, training_params
        )
