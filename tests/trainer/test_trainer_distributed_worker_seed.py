import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_accelerator,
    run_first,
    torch_device,
)


def gather_from_all_gpus(tensor, world_size):
    # Prepare a list to gather tensors from all processes
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return gather_list  # List of tensors from all ranks


class DummyDataset(Dataset):
    def __init__(self):
        self.length = 64

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        x = random.random()
        y = np.random.random()
        z = torch.rand([]).item()
        return {"x": torch.tensor([x, y, z])}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        local_tensor = torch.tensor(x, device=torch_device)
        gathered = gather_from_all_gpus(local_tensor, dist.get_world_size())
        assert not all(torch.allclose(t, gathered[0]) for t in gathered[1:])
        y = self.fc(x)
        return (y.mean(), y)


class TestTrainerDistributedWorkerSeed(TestCasePlus):
    @run_first
    @require_torch_multi_accelerator
    def test_trainer(self):
        device_count = backend_device_count(torch_device)
        output_dir = self.get_auto_remove_tmp_dir()
        distributed_args = f"""--nproc_per_node={device_count}
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_trainer_distributed_worker_seed.py
        """.split()
        args = f"--output_dir {output_dir}".split()
        cmd = ["torchrun"] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())


def run_distributed_training(training_args):
    set_seed(42)
    model = DummyModel()
    dataset = DummyDataset()
    training_args.max_steps = 10
    # dataloader_num_workers must be > 0 to enable worker_init_fn
    training_args.dataloader_num_workers = 2
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    run_distributed_training(training_args)
