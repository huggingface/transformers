import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from transformers.integrations.pipeline_parallel import add_pipeline_parallel_hooks


class TinyModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.lin0 = nn.Linear(hidden, hidden, bias=False)
        self.lin1 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        x = self.lin0(x)
        x = torch.relu(x)
        x = self.lin1(x)
        return x


def init_dist():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def main():
    init_dist()
    rank = dist.get_rank()
    device = torch.device("cuda", rank % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(0)
    base_model = TinyModel().to("cpu")
    state_dict = base_model.state_dict()
    dist.broadcast_object_list([state_dict], src=0)

    model = TinyModel().to(device)
    model.load_state_dict(state_dict)

    pp_plan = {
        "lin0": (["x"], ["x"]),
        "lin1": (["x"], ["x"]),
    }
    add_pipeline_parallel_hooks(model, pp_plan, device_mesh=None, pp_size=dist.get_world_size())

    x = torch.randn(2, 8, device=device)
    out = model(x)

    if rank == dist.get_world_size() - 1:
        ref = base_model(x.to("cpu")).detach()
        dist.send(out.detach().cpu(), dst=0)
        if rank == 0 and dist.get_world_size() == 1:  # single rank debug
            torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)
    if rank == 0 and dist.get_world_size() > 1:
        recv = torch.empty_like(x.cpu())
        dist.recv(recv, src=dist.get_world_size() - 1)
        ref = base_model(x.to("cpu")).detach()
        torch.testing.assert_close(recv, ref, atol=1e-4, rtol=1e-4)
        print("pipeline_parallel_smoke: OK")


if __name__ == "__main__":
    main()
