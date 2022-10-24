import math

import torch
import torch.distributed
from torch import nn


class TensorParallelColumnLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.process_group = process_group
        self.tp_world_size = process_group.size()

        assert out_features % self.tp_world_size == 0

        self.in_features = in_features
        self.out_features = out_features // self.tp_world_size
        # We change from traditional `nn.Linear` and remove unecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.in_features, self.out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

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

    @staticmethod
    @torch.jit.script
    def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        # # Note the the unsharded equivalent requires us to sum over bias instead of averaging.
        # in_features, out_features = weight.shape
        # size_out = input.size()[:-1] + (out_features,)
        # # TODO @thomasw21: when using torch.jit.script, `addmm` is decomposed to `add + mm`
        # return torch.addmm(bias, input.view(-1, in_features), weight).view(size_out)

        in_features, out_features = weight.shape
        size_out = input.size()[:-1] + (out_features,)
        # TODO @thomasw21: when using torch.jit.script, `addmm` is decomposed to `add + mm`
        input = input.view(-1, in_features)
        # HACK @thomas21: turns out `aten::addmm.out` is not decomposed
        out = torch.empty((0,), device=input.device, dtype=input.dtype)
        out = torch.addmm(bias, input, weight, out=out.view(-1, out_features))

        return out.view(size_out)

        # return F.linear(input, weight=weight.transpose(1,0), bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input, weight=self.weight, bias=self.bias)
        return out


class TensorParallelRowLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.process_group = process_group
        self.tp_world_size = process_group.size()

        assert in_features % self.tp_world_size == 0

        self.in_features = in_features // self.tp_world_size
        self.out_features = out_features
        # We change from traditional `nn.Linear` and remove unecessary `torch.Tensor.transpose` operation
        self.weight = nn.Parameter(torch.empty((self.in_features, self.out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

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

    @staticmethod
    @torch.jit.script
    def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        # # Note the the unsharded equivalent requires us to sum over bias instead of averaging.
        # in_features, out_features = weight.shape
        # size_out = input.size()[:-1] + (out_features,)
        # # TODO @thomasw21: when using torch.jit.script, `addmm` is decomposed to `add + mm`
        # input = input.view(-1, in_features)
        # # with torch.jit.strict_fusion():
        # out =  torch.addmm(bias, input, weight)
        # return out.view(size_out)

        in_features, out_features = weight.shape
        size_out = input.size()[:-1] + (out_features,)
        # TODO @thomasw21: when using torch.jit.script, `addmm` is decomposed to `add + mm`
        input = input.view(-1, in_features)
        # HACK @thomas21: turns out `aten::addmm.out` is not decomposed
        out = torch.empty((0,), device=input.device, dtype=input.dtype)
        out = torch.addmm(bias, input, weight, out=out.view(-1, out_features))

        return out.view(size_out)

        # return F.linear(input, weight=weight.transpose(1,0), bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input, weight=self.weight, bias=self.bias)
        torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.original_num_embeddings = num_embeddings

        # TODO @thomasw21 fix and remove that constraint
        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
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
        # if torch.any(torch.logical_or(0 > input, input >= self.original_num_embeddings)):
        #     raise IndexError(
        #         f"Input is required to be in [0, {self.original_num_embeddings}[, got min: {torch.min(input)} and max: {torch.max(input)}"
        #     )

        # `0` if input is in the correct interval, else `1`
        input_mask = torch.logical_or(self.min_id > input, input >= self.max_id)
        # translate for [0, self.max_id - self.min_id[
        input = input - self.min_id
        # default all out of bounds values to `0`
        input[input_mask] = 0
        out = super().forward(input)
        out[input_mask] = 0.0
        torch.distributed.all_reduce(out, group=self.process_group)
        return out
