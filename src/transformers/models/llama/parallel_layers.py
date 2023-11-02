import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torch.nn.functional as F
from torch import nn
import math


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export() or get_world_size(None) == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def symbolic(g: torch.Graph, x) -> torch.Value:
        return g.op("com.microsoft::AllReduce", x)


def get_process_group():
    if dist.is_initialized():
        return _get_default_group()
    return None

def get_world_size(process_group):
    group = get_process_group() if process_group is None else process_group
    return dist.get_world_size(group) if group is not None else 1

def get_rank(process_group):
    group = get_process_group() if process_group is None else process_group
    return dist.get_rank(group) if group is not None else 0


class TensorParallelColumnLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        process_group: torch.distributed.ProcessGroup = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_bias = bias

        self.process_group = get_process_group() if process_group is None else process_group
        self.tp_world_size = get_world_size(self.process_group)

        assert out_features % self.tp_world_size == 0

        self.in_features = in_features
        self.out_features = out_features
        # We change from traditional `nn.Linear` and remove unecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def parallel_split(self):
        if self.tp_world_size == 1:
            return

        rank = get_rank(self.process_group)

        weight = self.weight.chunk(self.tp_world_size)[rank]
        self.weight = nn.Parameter(weight)

        if self.use_bias:
            bias = self.bias.chunk(self.tp_world_size)[rank]
            self.bias = nn.Parameter(bias)

    def reset_parameters(self) -> None:
        """From `torch.nn.Linear`"""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        """From `torch.nn.Linear`"""
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, weight=self.weight, bias=self.bias)
        return out


class TensorParallelRowLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        process_group: torch.distributed.ProcessGroup = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_bias = bias

        self.process_group = get_process_group() if process_group is None else process_group
        self.tp_world_size = get_world_size(self.process_group)

        assert in_features % self.tp_world_size == 0

        self.in_features = in_features
        self.out_features = out_features
        # We change from traditional `nn.Linear` and remove unecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def parallel_split(self):
        if self.tp_world_size == 1:
            return

        rank = get_rank(self.process_group)

        weight = self.weight.chunk(self.tp_world_size, dim=1)[rank]
        self.weight = nn.Parameter(weight)

    def reset_parameters(self) -> None:
        """From `torch.nn.Linear`"""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, weight=self.weight, bias=self.bias)

        if self.tp_world_size > 1:
            out = AllReduce.apply(out)

        return out

    def extra_repr(self) -> str:
        """From `torch.nn.Linear`"""
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        process_group: torch.distributed.ProcessGroup = None,
    ):
        self.process_group = get_process_group() if process_group is None else process_group
        self.tp_rank = get_rank(self.process_group)
        self.tp_world_size = get_world_size(self.process_groupsa)

        self.original_num_embeddings = num_embeddings

        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check
        if torch.any(torch.logical_or(0 > input, input >= self.original_num_embeddings)):
            raise IndexError(
                f"Input is required to be in [0, {self.original_num_embeddings}[, got min: {torch.min(input)} and max: {torch.max(input)}"
            )

        input_mask = torch.logical_or(self.min_id > input, input >= self.max_id)
        input = input - self.min_id
        input[input_mask] = 0
        out = super().forward(input)
        input_mask = input_mask.view(*input_mask.shape, 1)  # add a new dim
        input_mask = input_mask.expand(out.shape)
        out[input_mask] = 0.0

        if self.tp_world_size > 1:
            out = AllReduce.apply(out)

        return out
