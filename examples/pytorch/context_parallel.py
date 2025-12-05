# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss


world_size = int(os.environ.get("WORLD_SIZE", "1"))
cp_mesh = init_device_mesh("cuda", (world_size,))
rank = torch.distributed.get_node_local_rank()

device = "cuda"
dtype = torch.bfloat16
sdpa_backend = SDPBackend.FLASH_ATTENTION

# prepare inputs
batch_size = 1
seq_len = 128

input_ids = torch.randint(low=8, high=64, size=(batch_size, seq_len), device=device)

ignore_index = -100
# When using CP, we need to use `shift_labels`
shift_labels = torch.nn.functional.pad(input_ids, (0, 1), value=ignore_index)
shift_labels = shift_labels[..., 1:].contiguous()

position_ids = (
    torch.cumsum(torch.ones(size=input_ids.size(), dtype=input_ids.dtype, device=input_ids.device), dim=1) - 1
)

# sync input as they are created randomly
dist.broadcast(input_ids, src=0)
dist.broadcast(shift_labels, src=0)
dist.broadcast(position_ids, src=0)

# model and optimizer
repo_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, device_map=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

model.train()
model.zero_grad()
optimizer.zero_grad()

# For loss
vocab_size = model.config.vocab_size

# so training could be synced
model = DDP(model, device_ids=[rank])

# prepare for CP
buffers = (input_ids, shift_labels, position_ids)
buffer_seq_dims = (1, 1, 1)
# `no_restore_buffers=set(buffers)` is required if `loss.backward` is outside `context_parallel`.
# no_restore_buffers = set(buffers)
no_restore_buffers = None

# run with CP
with sdpa_kernel(sdpa_backend):
    with context_parallel(
        cp_mesh,
        buffers=buffers,
        buffer_seq_dims=buffer_seq_dims,
        no_restore_buffers=no_restore_buffers,
    ):
        outputs = model(input_ids, shift_labels=shift_labels, position_ids=position_ids)
        print(outputs.logits.shape)

        # So far we need to compute `loss` outside `model.forward` when using `shift_labels`
        # loss = outputs.loss
        loss = ForCausalLMLoss(logits=outputs.logits, labels=None, shift_labels=shift_labels, vocab_size=vocab_size)

        # This could be outside `context_parallel` context if `no_restore_buffers` is specified
        loss.backward()
        optimizer.step()
