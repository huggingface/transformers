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
import os
import unittest

from transformers.testing_utils import get_torch_dist_unique_port, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.device_mesh import init_device_mesh

    from transformers.distributed.tensor_parallel import EpDispatchExpertsParallel

    class _RecordingExperts(torch.nn.Module):
        """Stand-in experts: the identity, recording the type of the rows it is handed."""

        def __init__(self, num_experts):
            super().__init__()
            self.num_experts = num_experts
            self.received = None

        def forward(self, hidden_states, top_k_index, top_k_weights):
            self.received = type(hidden_states)
            return hidden_states.clone()


def _dispatch_hands_the_experts_the_received_rows(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ep_mesh = init_device_mesh("cpu", (world_size,))
        hidden_states = torch.arange(4 * 8, dtype=torch.float32).view(4, 8) + 100 * rank
        top_k_index = torch.tensor([[0, 3], [1, 2], [2, 1], [3, 0]])
        top_k_weights = torch.full((4, 2), 0.5)
        experts = _RecordingExperts(num_experts=2)
        EpDispatchExpertsParallel().install_forward(experts, ep_mesh)
        output = experts(hidden_states, top_k_index, top_k_weights)
        assert experts.received is torch.Tensor, experts.received
        torch.testing.assert_close(output, hidden_states)
    finally:
        dist.destroy_process_group()


@require_torch
class EpDispatchTest(unittest.TestCase):
    def test_dispatch_hands_the_experts_the_received_rows(self):
        """The experts get the all-to-all's received rows as a plain tensor. Its pending result
        (`AsyncCollectiveTensor`) holds a null pointer until a torch op waits on it, which a kernel launched straight
        from Python (Triton, a raw pointer) never does."""
        world_size = 2
        mp.spawn(
            _dispatch_hands_the_experts_the_received_rows,
            args=(world_size, get_torch_dist_unique_port()),
            nprocs=world_size,
        )


if __name__ == "__main__":
    unittest.main()
