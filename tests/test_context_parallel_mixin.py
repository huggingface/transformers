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
from abc import ABC, abstractmethod

import pytest

from transformers import is_torch_available
from transformers.distributed import DistributedConfig
from transformers.testing_utils import SINGLE_PROCESS_DEVICES, is_fsdp_test, torch_device
from transformers.utils import is_torch_greater_or_equal

from .distributed_testing_utils import run_distributed
from .test_fsdp_mixin import _deterministic_init_model_dir, _get_available_fsdp_workers


if is_torch_available():
    import torch
    import torch.distributed as dist

    from transformers.distributed.context_parallel import apply_context_parallelism

    from .context_parallel.test_context_parallel import _compare_with_unsharded, _MeshStub, _tiny_batch


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


def _refusal(config, model_class, cp_size):
    """
    The error that stops `model_class` from loading with context parallelism, or `None`. The model is built from the
    config a checkpoint reloads, as the workers do, and its plan is resolved as the load does.
    """
    try:
        with torch.device("meta"):
            model = model_class(type(config).from_dict(config.to_diff_dict()))
        _, _, _, cp_plan = model._resolve_parallel_plans(DistributedConfig(cp_size=cp_size))
        apply_context_parallelism(model, _MeshStub(cp_size), cp_plan)
    except Exception as error:
        return error
    return None


def _test_cp_refused(rank, config, model_class):
    """Loading with context parallelism raises instead of splitting the sequences of a model it cannot shard."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as init_dir:
        with pytest.raises(ValueError):
            model_class.from_pretrained(init_dir, distributed_config=DistributedConfig(cp_size=dist.get_world_size()))
        dist.barrier()


def _test_cp_matches_unsharded(rank, config, model_class, packed):
    """The loss, this rank's logits and every gradient of a context parallel model match the run on whole sequences."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size=2, seq_len=8, packed=packed)
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, DistributedConfig(cp_size=dist.get_world_size()), model_class)
        dist.barrier()


class ContextParallelTesterMixin(ABC):
    """Context parallel tests on 2 ranks, added to the causal language model test classes."""

    cp_world_size: int = 2

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _run_cp_matches_unsharded(self, packed):
        if not is_torch_greater_or_equal("2.10"):
            self.skipTest("Context parallelism requires torch>=2.10")
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("The device mesh is built on the default accelerator, which MPS cannot back")
        if torch_device in SINGLE_PROCESS_DEVICES:
            self.skipTest(f"Distributed tests spawn one process per device, which {torch_device!r} doesn't allow")
        if _get_available_fsdp_workers() < self.cp_world_size:
            self.skipTest(f"Need at least {self.cp_world_size} workers")
        config = copy.deepcopy(self.model_tester.get_config())
        if (
            CONTEXT_PARALLEL_TEST_MODEL_TYPES is not None
            and config.model_type not in CONTEXT_PARALLEL_TEST_MODEL_TYPES
        ):
            self.skipTest(f"Context parallel tests are not enabled for model_type={config.model_type!r}")

        # the auxiliary load-balancing loss of the MoE models does not sum over the tokens of a sequence
        if hasattr(config.get_text_config(), "output_router_logits"):
            config.get_text_config().output_router_logits = False
        model_class = self.model_tester.causal_lm_class
        if (refusal := _refusal(config, model_class, self.cp_world_size)) is not None:
            # a `ValueError` is the load refusing the model, which the workers check
            if isinstance(refusal, ValueError):
                run_distributed(world_size=self.cp_world_size, backend="gloo")(_test_cp_refused)(config, model_class)
            self.skipTest(f"Context parallelism cannot run this model: {refusal}")
        run_distributed(world_size=self.cp_world_size, backend="gloo")(_test_cp_matches_unsharded)(
            config, model_class, packed
        )

    @is_fsdp_test
    def test_cp_matches_unsharded(self):
        """The loss, this rank's logits and every gradient match the run on whole sequences."""
        self._run_cp_matches_unsharded(packed=False)

    @is_fsdp_test
    def test_cp_matches_unsharded_packed(self):
        """Two documents in every sequence, so `position_ids` restart mid-sequence."""
        self._run_cp_matches_unsharded(packed=True)
