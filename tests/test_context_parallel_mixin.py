# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Context parallel tester mixin for model tests."""

import copy
import os
from abc import ABC, abstractmethod

import pytest
from parameterized import parameterized

from transformers import is_torch_available
from transformers.distributed import DistributedConfig
from transformers.testing_utils import SINGLE_PROCESS_DEVICES, backend_device_count, torch_device

from .distributed_testing_utils import _get_distributed_device_type, run_distributed
from .test_fsdp_mixin import _deterministic_init_model_dir


if is_torch_available():
    import torch
    import torch.distributed as dist

    from transformers.distributed.context_parallel import _check_context_parallelism, cp_implementations

    from .context_parallel.test_context_parallel import _compare_with_unsharded, _MeshStub, _tiny_batch

CP_IMPLEMENTATIONS = list(cp_implementations) if is_torch_available() else []


# Set to None to run the context parallel tests for every causal language model.
CONTEXT_PARALLEL_TEST_MODEL_TYPES = {
    "gemma3_text",
    "glm4_moe",
    "llama",
    "mistral",
    "phi3",
    "qwen3",
    "qwen3_moe",
}

BATCH_SIZE = 2
SEQ_LEN = 8


def _available_workers():
    device_type = _get_distributed_device_type()
    return (os.cpu_count() or 1) if device_type == "cpu" else backend_device_count(device_type)


def _full_attention(config):
    """Turn the sliding-window layers into full attention, whose mask `sdpa` can rebuild for the whole sequence."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if "sliding_attention" not in layer_types or not set(layer_types) <= {"full_attention", "sliding_attention"}:
            return False
        config.layer_types = ["full_attention"] * len(layer_types)
        return True
    if getattr(config, "sliding_window", None) is None:
        return False
    config.sliding_window = None
    return True


def _refusal(config, model_class, implementation, cp_size):
    """
    The error that stops `config` from running under context parallelism, or `None` when the config fits. The model is
    built from the config a checkpoint reloads, which is the one the workers get.
    """
    try:
        with torch.device("meta"):
            model = model_class(type(config).from_dict(config.to_diff_dict()))
        _check_context_parallelism(model)
        cp_implementations[implementation](_MeshStub(cp_size)).check(model)
    except Exception as error:
        return error
    return None


def _context_parallel_config(config, model_class, implementation, cp_size):
    """
    A copy of `config` with the change context parallelism needs, made only where the copy without it is refused, and the
    reason the model is refused when it does not help.
    """
    adapted = copy.deepcopy(config)
    text_config = adapted.get_text_config()
    # The auxiliary load-balancing loss of the MoE models does not sum over the tokens of a sequence.
    if hasattr(text_config, "output_router_logits"):
        text_config.output_router_logits = False

    refusal = _refusal(adapted, model_class, implementation, cp_size)
    if refusal is None:
        return adapted, None
    candidate = copy.deepcopy(adapted)
    try:
        changed = _full_attention(candidate.get_text_config())
    except Exception:  # a config class that validates its fields can refuse the change
        return None, refusal
    if changed and _refusal(candidate, model_class, implementation, cp_size) is None:
        return candidate, None
    return None, refusal


def _test_cp_refused(rank, config, model_class, cp_implementation):
    """Loading with context parallelism raises instead of splitting the sequences of a model it cannot shard."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as init_dir:
        distributed_config = DistributedConfig(cp_size=dist.get_world_size(), cp_implementation=cp_implementation)
        with pytest.raises(ValueError):
            model_class.from_pretrained(init_dir, distributed_config=distributed_config)
        dist.barrier()


def _test_cp_matches_unsharded(rank, config, model_class, cp_implementation, packed):
    """The loss, this rank's logits and every gradient of a context parallel model match the run on whole sequences."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as init_dir:
        input_ids, position_ids = _tiny_batch(config, BATCH_SIZE, SEQ_LEN, packed)
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        distributed_config = DistributedConfig(cp_size=dist.get_world_size(), cp_implementation=cp_implementation)
        _compare_with_unsharded(init_dir, batch, distributed_config, model_class)
        dist.barrier()


class ContextParallelTesterMixin(ABC):
    """Context parallel tests on 2 ranks, added to the causal language model test classes."""

    cp_world_size: int = 2

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _skip_if_context_parallel_not_supported(self):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("The device mesh is built on the default accelerator, which MPS cannot back")
        if torch_device in SINGLE_PROCESS_DEVICES:
            self.skipTest(f"Distributed tests spawn one process per device, which {torch_device!r} doesn't allow")
        available_workers = _available_workers()
        if available_workers < self.cp_world_size:
            self.skipTest(f"Need at least {self.cp_world_size} workers, have {available_workers}")
        config = self.model_tester.get_config()
        if (
            CONTEXT_PARALLEL_TEST_MODEL_TYPES is not None
            and config.model_type not in CONTEXT_PARALLEL_TEST_MODEL_TYPES
        ):
            self.skipTest(
                f"Context parallel tests are not enabled for model_type={config.model_type!r} "
                f"(enabled: {sorted(CONTEXT_PARALLEL_TEST_MODEL_TYPES)}). Set CONTEXT_PARALLEL_TEST_MODEL_TYPES = None "
                "to run all tests."
            )

    def _run_cp_matches_unsharded(self, cp_implementation, packed):
        self._skip_if_context_parallel_not_supported()
        model_class = self.model_tester.causal_lm_class
        config = self.model_tester.get_config()
        adapted, refusal = _context_parallel_config(config, model_class, cp_implementation, self.cp_world_size)
        if adapted is None:
            # Only a `ValueError` comes from the load refusing the model, which the workers can check.
            if isinstance(refusal, ValueError):
                run_distributed(world_size=self.cp_world_size, backend="gloo")(_test_cp_refused)(
                    config, model_class, cp_implementation
                )
            self.skipTest(f"Context parallelism cannot run this model: {refusal}")
        run_distributed(world_size=self.cp_world_size, backend="gloo")(_test_cp_matches_unsharded)(
            adapted, model_class, cp_implementation, packed
        )

    @parameterized.expand(CP_IMPLEMENTATIONS, skip_on_empty=True)
    def test_cp_matches_unsharded(self, cp_implementation):
        """The loss, this rank's logits and every gradient match the run on whole sequences."""
        self._run_cp_matches_unsharded(cp_implementation, packed=False)

    @parameterized.expand(CP_IMPLEMENTATIONS, skip_on_empty=True)
    def test_cp_matches_unsharded_packed(self, cp_implementation):
        """Two documents in every sequence, so `position_ids` restart mid-sequence."""
        self._run_cp_matches_unsharded(cp_implementation, packed=True)
