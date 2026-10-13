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
import itertools
import tempfile
import unittest

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available, is_torch_distributed_available


if is_torch_available():
    import torch
    import torch.multiprocessing as mp

    from transformers.distributed.utils import clip_grad_norm_

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _gradient_clipping_worker(rank, world_size, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    mesh = init_device_mesh("cpu", (world_size,))
    norm_types = (0.0, 2.0, float("inf"))
    # Which of the two parameters have a sharded (DTensor) gradient: none, all on one mesh, or a plain/DTensor mix.
    sharding_layouts = ((False, False), (True, True), (False, True))
    max_norms = (1.0, 1e4)
    for norm_type, sharded, max_norm in itertools.product(norm_types, sharding_layouts, max_norms):
        grads = [torch.arange(1.0, 65.0).reshape(8, 8) * (i + 1) for i in range(len(sharded))]
        reference = [torch.nn.Parameter(torch.zeros(8, 8)) for _ in grads]
        params = [torch.nn.Parameter(torch.zeros(8, 8)) for _ in grads]
        for ref, param, grad, is_sharded in zip(reference, params, grads, sharded):
            ref.grad = grad.clone()
            param.grad = distribute_tensor(grad, mesh, [Shard(0)]) if is_sharded else grad

        expected_norm = torch.nn.utils.clip_grad_norm_(reference, max_norm, norm_type=norm_type)
        actual_norm = clip_grad_norm_(params, max_norm, norm_type=norm_type)
        torch.testing.assert_close(actual_norm, expected_norm)
        for param, ref in zip(params, reference):
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            torch.testing.assert_close(grad, ref.grad)
    dist.destroy_process_group()


@require_torch
@unittest.skipUnless(is_torch_distributed_available() and dist.is_gloo_available(), "Requires distributed Gloo")
class DistributedUtilsTest(unittest.TestCase):
    def test_clip_grad_norm_mixed_meshes(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(_gradient_clipping_worker, args=(world_size, f"{tmpdir}/init"), nprocs=world_size)


if __name__ == "__main__":
    unittest.main()
