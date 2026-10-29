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
"""Expert parallel tests for model test classes; the distributed harness lives in `test_tensor_parallel_mixin.py`."""

import tempfile
from abc import ABC
from itertools import product

from parameterized import parameterized

from transformers import set_seed
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.testing_utils import is_expert_parallel_test, is_torch_available

from .test_tensor_parallel_mixin import _get_expert_weight_names, _init_distributed, _reference_grad_shard


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor


# =============================================================================
# Constants
# =============================================================================

# Set to None to run distributed EP tests for every model with a plan.
# Representative MoE model types covered by distributed EP tests.
EP_DISTRIBUTED_TEST_MODEL_TYPES = {
    "qwen4_exp_text",
    "qwen3_moe",
    "glm_moe_dsa",
    "deepseek_v4",
    "glm4_moe_lite",
    "glm4_moe",
    "olmoe",
    "qwen2_moe",
    "cohere2_moe",
    # VLM
    "glm4v_moe",
}


# =============================================================================
# Distributed helpers (top-level for pickling by mp.spawn)
# =============================================================================


def _load_ep_and_reference_models(model_path, model_class):
    """Load EP model and non-EP reference model for comparison."""
    model_ep = model_class.from_pretrained(
        model_path,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size(), ep_size=dist.get_world_size()),
    )
    dist.barrier()

    device = model_ep.device
    model_ref = model_class.from_pretrained(model_path)
    model_ref = model_ref.to(device)

    return model_ep, model_ref, device


def _test_ep_forward_impl(_rank, model_path, model_class, atol, rtol, experts_implementation):
    """Implementation for comparing EP and non-EP model outputs."""
    set_seed(0)

    model_ep, model_ref, device = _load_ep_and_reference_models(model_path, model_class)

    model_ep.eval()
    model_ref.eval()

    model_ep.set_experts_implementation(experts_implementation)
    model_ref.set_experts_implementation(experts_implementation)

    vocab_size = model_ref.config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)

    with torch.no_grad():
        logits_ref = model_ref(input_ids).logits
        logits_ep = model_ep(input_ids).logits

    diff = (logits_ref - logits_ep).abs()
    assert torch.allclose(logits_ref, logits_ep, atol=atol, rtol=rtol), (
        f"EP and non-EP model outputs differ. Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_ep_backward_impl(rank, model_path, model_class, atol, rtol, experts_implementation):
    """Implementation for comparing EP and non-EP model backward passes."""
    set_seed(0)

    model_ep, model_ref, device = _load_ep_and_reference_models(model_path, model_class)
    model_ep.train()
    model_ref.train()

    model_ep.set_experts_implementation(experts_implementation)
    model_ref.set_experts_implementation(experts_implementation)

    vocab_size = model_ref.config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)

    loss_ref = model_ref(input_ids, labels=labels).loss
    loss_ref.backward()

    loss_ep = model_ep(input_ids, labels=labels).loss
    loss_ep.backward()

    assert torch.allclose(loss_ref, loss_ep, atol=atol, rtol=rtol), (
        f"EP and non-EP model losses differ. "
        f"Non-EP loss: {loss_ref.item()}, EP loss: {loss_ep.item()}, "
        f"Diff: {(loss_ref - loss_ep).abs().item()}"
    )

    world_size = dist.get_world_size()
    expert_weights = set(_get_expert_weight_names(model_ep))
    failed_grads = {}
    for (name, param), (_, param_ep) in zip(model_ref.named_parameters(), model_ep.named_parameters()):
        if param.grad is None and param_ep.grad is None:
            continue
        if param.grad is None or param_ep.grad is None:
            failed_grads[name] = "no gradient on the " + ("EP" if param_ep.grad is None else "non-EP") + " model"
            continue
        local_grad = param_ep.grad.to_local() if isinstance(param_ep.grad, DTensor) else param_ep.grad
        expected = _reference_grad_shard(
            name, param.grad, local_grad, rank, world_size, model_ep._tp_plan, expert_weights
        ).cpu()
        if not torch.allclose(expected, local_grad.cpu(), atol=atol, rtol=rtol):
            failed_grads[name] = f"max diff = {(expected - local_grad.cpu()).abs().max().item()}"

    assert not failed_grads, f"Gradients differ for {len(failed_grads)} parameter(s):\n" + "\n".join(
        f"  {name}: {failure}" for name, failure in failed_grads.items()
    )

    dist.barrier()


class ExpertParallelTesterMixin(ABC):
    """
    Mixin for expert parallel tests. Add to model test classes alongside `ModelTesterMixin` and
    `TensorParallelTesterMixin`, whose helpers (`_skip_if_not_supported`, `_get_tp_config`, `_get_tp_model_class`) and
    `tensor_parallel_*` settings it reuses: EP runs with `tp_size == ep_size == tensor_parallel_size`.
    """

    # ============================================================
    # Helper methods
    # ============================================================
    def _has_ep_plan(self) -> bool:
        """Check if model has an expert parallel plan defined."""
        config = self.model_tester.get_config().get_text_config()
        return hasattr(config, "base_model_ep_plan") and config.base_model_ep_plan is not None

    def _skip_if_ep_distributed_not_enabled(self):
        """Skip expensive distributed EP tests for models outside the allowlist."""
        config = self.model_tester.get_config()
        if EP_DISTRIBUTED_TEST_MODEL_TYPES is not None and config.model_type not in EP_DISTRIBUTED_TEST_MODEL_TYPES:
            self.skipTest(
                f"Expert parallel distributed tests are not enabled for model_type={config.model_type!r} "
                f"(enabled: {sorted(EP_DISTRIBUTED_TEST_MODEL_TYPES)}). "
                "Set EP_DISTRIBUTED_TEST_MODEL_TYPES = None to run all tests."
            )

    def _skip_if_ep_not_supported(self):
        """Check and skip the test if expert parallel is not supported for this model/environment."""
        self._skip_if_distributed_not_supported(parallelism="Expert")

        if not self._has_ep_plan():
            self.skipTest("Model does not have an expert parallel plan (base_model_ep_plan)")

        self._skip_if_ep_distributed_not_enabled()

    @parameterized.expand(
        list(
            product(
                [False, True],  # tie_word_embeddings
                ["eager", "grouped_mm", "batched_mm"],  # experts_implementation
            )
        )
    )
    @is_expert_parallel_test
    def test_ep_forward(self, tie_word_embeddings, experts_implementation):
        self._skip_if_ep_not_supported()

        config = self._get_tp_config(tie_word_embeddings=tie_word_embeddings)
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_ep_forward_impl)(
                tmp_dir, model_class, atol, rtol, experts_implementation
            )

    @parameterized.expand([("eager",), ("grouped_mm",), ("batched_mm",)])
    @is_expert_parallel_test
    def test_ep_backward(self, experts_implementation):
        self._skip_if_ep_not_supported()

        config = self._get_tp_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_ep_backward_impl)(
                tmp_dir, model_class, atol, rtol, experts_implementation
            )
