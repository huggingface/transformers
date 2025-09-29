import os
import torch
from transformers.utils import is_torch_greater_or_equal

def initialize_parallelism(tp_plan, pp_plan, tp_size=None, pp_size=None):
    """
    Initializes the parallelism and returns all the necessary variables.
    """
    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Tensor parallel is only supported for `torch>=2.5`.")

    # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
    device_type = torch._C._get_accelerator().type
    current_device = getattr(torch, device_type)
    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            assert tp_size * pp_size == world_size, f"tp_size ({tp_size}) * pp_size ({pp_size}) must be equal to world_size ({world_size})"
    
            backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
            backend = backend_map.get(device_type)
            if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", "0")):
                backend = "ccl"
            if device_type == "xpu":

                if not is_torch_greater_or_equal("2.8", accept_dev=True):
                    backend = "ccl"

            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
            current_device = getattr(torch, device_type)
            if device_type != "cpu":
                current_device.set_device(local_rank)

        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use `tp_plan='auto'`."
            ) from e

    if device_type != "cpu":
        current_device.set_device(int(os.environ["LOCAL_RANK"]))
    index = current_device.current_device() if device_type != "cpu" else None

    # # Silence output for non-primary ranks
    # if index is not None and index > 0:
    #     import sys

    #     sys.stdout = open(os.devnull, "w")
    #     sys.stderr = open(os.devnull, "w")

    device_mesh = torch.distributed.init_device_mesh(device_type, (tp_size, pp_size), mesh_dim_names=("tp", "pp"))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = torch.device(f"{device_type}:{local_rank}")

    return tp_plan, pp_plan, device_map, device_mesh