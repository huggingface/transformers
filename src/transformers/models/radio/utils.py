import torch.distributed as dist


def get_rank(group: dist.ProcessGroup | None = None):
    return dist.get_rank(group) if dist.is_initialized() else 0


def get_world_size(group: dist.ProcessGroup | None = None):
    return dist.get_world_size(group) if dist.is_initialized() else 1


def barrier(group: dist.ProcessGroup | None = None):
    if dist.is_initialized():
        dist.barrier(group)


class rank_gate:
    """
    Execute the function on rank 0 first, followed by all other ranks. Useful when caches may need to be populated in a distributed environment.
    """

    def __init__(self, func=None):
        self.func = func

    def __call__(self, *args, **kwargs):
        rank = get_rank()
        if rank == 0:
            result = self.func(*args, **kwargs)
        barrier()
        if rank > 0:
            result = self.func(*args, **kwargs)
        return result

    def __enter__(self, *args, **kwargs):
        if get_rank() > 0:
            barrier()

    def __exit__(self, *args, **kwargs):
        if get_rank() == 0:
            barrier()
