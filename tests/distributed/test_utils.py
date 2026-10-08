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
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from transformers.distributed.utils import clip_grad_norm_

    if dist.is_available():
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _full_tensor(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


@contextmanager
def _distributed_context(rank, directory):
    environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "4"}
    with patch.dict(os.environ, environment), patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=60),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def _gradient_clipping_worker(rank, directory):
    with _distributed_context(rank, directory):
        mesh = init_device_mesh("cpu", (4,))
        for norm_type in (0.0, 2.0, float("inf")):
            for distributed in ((False, False), (True, True), (False, True)):
                for max_norm in (1.0, 10000.0):
                    parameters, reference = [], []
                    for i, is_distributed in enumerate(distributed):
                        gradient = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8) * (i + 1)
                        expected = torch.nn.Parameter(torch.zeros_like(gradient))
                        expected.grad = gradient.clone()
                        reference.append(expected)
                        if is_distributed:
                            gradient = distribute_tensor(gradient, mesh, [Shard(0)])
                        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
                        parameter.grad = gradient
                        parameters.append(parameter)
                    expected_norm = torch.nn.utils.clip_grad_norm_(
                        reference, max_norm, foreach=True, norm_type=norm_type
                    )
                    actual_norm = clip_grad_norm_(parameters, max_norm, foreach=True, norm_type=norm_type)
                    torch.testing.assert_close(_full_tensor(actual_norm), expected_norm)
                    for parameter, expected in zip(parameters, reference):
                        torch.testing.assert_close(_full_tensor(parameter.grad), expected.grad)


@require_torch
@unittest.skipUnless(
    is_torch_available() and dist.is_available() and dist.is_gloo_available(), "Requires distributed Gloo"
)
class DistributedUtilsTest(unittest.TestCase):
    def test_gradient_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_gradient_clipping_worker, args=(directory,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
