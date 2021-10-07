# Copyright 2021 TUNiB Inc, NVIDIA CORPORATION and The Hugging Face Team. All Rights Reserved.
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
"""
Model parallelism utils.
Integration with parallelformers https://github.com/tunib-ai/parallelformers
"""

import os
from abc import ABC
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import Function


class MPU(object):
    """
    MPU: Model Parallel Unit

    MPU is key concept of 3D model parallelism and is inspired by Megatron-LM.
    The main difference with Megatron-LM is that each model has an their mpu.
    We implemented this by inheritance.

    We can combine several models later. For example, in the case of Electra,
    there is a generator model and a discriminator model. To parallelize all of them,
    each model must be parallelized in a different process group,
    so the mpu must be maintained in the model level, not the global state.

    Notes:
        Let's say we have a total of 16 GPUs denoted g0 ... g15 and we use 2 GPUs to parallelize the model tensor,
        and 4 GPUs to parallelize the model pipeline. The present method will create 8 tensor model-parallel group,
        4 pipeline model parallel groups and 8 data parallel groups as:

        - width: 4 pipeline parallel group
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        - height: 8 tensor parallel group
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        - depth: 8 data parallel group
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]

                        [g02, g06, g10, g14]
                      /  |              /  |
                     [g00, g04, g08, g12]  |
                     |   |             |   |
        3D parallel  |  [g03, g07, g11, g15]
                     |  /              |  /
                     [g01, g05, g09, g13]

                      +---+ +---------+  +---------+  +---------+  +---------+ +---+
              tensor  |g00| |   g00   |  |   g04   |  |   g08   |  |   g12   | |g12|
        data          |---| +---------+  +---------+  +---------+  +---------+ |---| ===> forward
              tensor  |g01| |   g01   |  |   g05   |  |   g09   |  |   g13   | |g13|
                      +---+ +---------+  +---------+  +---------+  +---------+ +---+
                       emb    pipeline     pipeline     pipeline     pipeline   emb

                      +---+ +---------+  +---------+  +---------+  +---------+ +---+
              tensor  |g02| |   g02   |  |   g06   |  |   g10   |  |   g12   | |g14|
        data          |---| +---------+  +---------+  +---------+  +---------+ |---| ===> forward
              tensor  |g03| |   g03   |  |   g07   |  |   g11   |  |   g15   | |g15|
                      +---+ +---------+  +---------+  +---------+  +---------+ +---+
                       emb    pipeline     pipeline     pipeline     pipeline   emb

    References:
        Original MPU implementation of Megatron-LM. We refactored all the codes to be pythonic.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/initialize.py

    """

    _tensor_model_parallel_group = None
    _pipeline_model_parallel_group = None
    _data_parallel_group = None

    _tensor_model_parallel_world_size = None
    _pipeline_model_parallel_world_size = None
    _data_parallel_world_size = None

    _tensor_model_parallel_rank = None
    _pipeline_model_parallel_rank = None
    _pipeline_global_ranks = None

    def __init__(
        self,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
    ) -> None:
        """
        Initialize MPU object. All process groups are initialized in this method.

        Args:
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
        """

        if not dist.is_initialized():
            self.initialize_distributed()

        current_rank = dist.get_rank()
        global_world_size = dist.get_world_size()

        tensor_model_parallel_size = min(
            tensor_model_parallel_size,
            global_world_size,
        )

        pipeline_model_parallel_size = min(
            pipeline_model_parallel_size,
            global_world_size,
        )

        total_model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size

        assert (
            global_world_size % total_model_parallel_size == 0
        ), "global world sizes must be divisible by model parallel world sizes (tp * pp)"

        num_tensor_model_parallel_groups = global_world_size // tensor_model_parallel_size

        num_pipeline_model_parallel_groups = global_world_size // pipeline_model_parallel_size

        # 1. initialize data parallel group
        self._initialize_data_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 2. initialize tensor model parallel group
        self._initialize_tensor_model_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            num_tensor_model_parallel_groups=num_tensor_model_parallel_groups,
        )

        # 3. initialize pipeline model parallel group
        self._initialize_pipeline_model_parallel_group(
            current_rank=current_rank,
            global_world_size=global_world_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 4. create distributed functions
        functions = self._initialize_functions()
        self._broadcast_fn = functions["broadcast"]
        self._reduce_fn = functions["reduce"]
        self._scatter_fn = functions["scatter"]
        self._gather_fn = functions["gather"]

    def _initialize_data_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize data parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
            num_pipeline_model_parallel_groups (int): the number of pipeline model parallel groups
        """
        assert self._data_parallel_group is None, "data parallel group is already initialized."

        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups

            for j in range(tensor_model_parallel_size):
                ranks = list(range(start_rank + j, end_rank, tensor_model_parallel_size))

                group = dist.new_group(ranks)
                if current_rank in ranks:
                    self._data_parallel_group = group

    def _initialize_tensor_model_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        num_tensor_model_parallel_groups: int,
    ) -> None:
        """
        Initialize tensor model parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            num_tensor_model_parallel_groups (int): the number of tensor model parallel groups
        """
        assert self._tensor_model_parallel_group is None, "tensor model parallel group is already initialized."

        for i in range(num_tensor_model_parallel_groups):
            start_rank = i * tensor_model_parallel_size
            end_rank = (i + 1) * tensor_model_parallel_size

            ranks = list(range(start_rank, end_rank))
            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._tensor_model_parallel_group = group

    def _initialize_pipeline_model_parallel_group(
        self,
        current_rank: int,
        global_world_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize pipeline model parallel group

        Args:
            current_rank (int): current rank
            global_world_size (int): global world size
            num_pipeline_model_parallel_groups (int): the number of model parallel groups
        """
        assert self._pipeline_model_parallel_group is None, "pipeline model parallel group is already initialized."

        for i in range(num_pipeline_model_parallel_groups):
            ranks = list(range(i, global_world_size, num_pipeline_model_parallel_groups))

            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._pipeline_model_parallel_group = group
                self._pipeline_global_ranks = ranks

    def model_parallel_is_initialized(self) -> bool:
        """
        Check if model and data parallel groups are initialized.

        Returns:
            bool: whether MPU is initialized
        """
        if (
            self._tensor_model_parallel_group is None
            or self._pipeline_model_parallel_group is None
            or self._data_parallel_group is None
        ):
            return False
        return True

    def get_model_parallel_group(self):
        """
        Get the tensor model parallel group.

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_group()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            ProcessGroup: tensor model parallel group
        """
        return self.get_tensor_model_parallel_group()

    def get_model_parallel_world_size(self) -> int:
        """
        Get the tensor model parallel world size

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_world_size()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_world_size()

    def get_model_parallel_rank(self) -> int:
        """
        Get the tensor model parallel rank

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_rank()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_rank()

    def get_tensor_model_parallel_group(self):
        """
        Get tensor model parallel group

        Returns:
            ProcessGroup: tensor model parallel group
        """

        assert self._tensor_model_parallel_group is not None, "tensor model parallel group is not initialized."

        return self._tensor_model_parallel_group

    def get_pipeline_model_parallel_group(self):
        """
        Get pipeline model parallel group

        Returns:
            ProcessGroup: pipeline model parallel group
        """
        assert self._pipeline_model_parallel_group is not None, "pipeline model parallel group is not initialized."

        return self._pipeline_model_parallel_group

    def get_data_parallel_group(self):
        assert self._data_parallel_group is not None, "data parallel group is not initialized."

        return self._data_parallel_group

    def get_tensor_model_parallel_world_size(self) -> int:
        """
        Get tensor model parallel world size

        Returns:
            int: tensor model parallel world size
        """
        if self._tensor_model_parallel_world_size is not None:
            return self._tensor_model_parallel_world_size

        return dist.get_world_size(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set tensor model parallel world size

        Args:
            world_size (int): tensor model parallel world size
        """
        self._tensor_model_parallel_world_size = world_size

    def get_pipeline_model_parallel_world_size(self) -> int:
        """
        Get pipeline model parallel world size

        Returns:
            int: pipeline model parallel world size
        """
        if self._pipeline_model_parallel_world_size is not None:
            return self._pipeline_model_parallel_world_size

        return dist.get_world_size(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set pipeline model parallel world size

        Args:
            world_size (int): pipeline model parallel world size
        """
        self._pipeline_model_parallel_world_size = world_size

    def get_tensor_model_parallel_rank(self) -> int:
        """
        Get tensor model parallel rank

        Returns:
            int: tensor model parallel rank
        """
        if self._tensor_model_parallel_rank is not None:
            return self._tensor_model_parallel_rank

        return dist.get_rank(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_rank(self, rank: int) -> None:
        """
        Set tensor model parallel rank

        Args:
            rank (int): tensor model parallel rank
        """

        self._tensor_model_parallel_rank = rank

    def get_pipeline_model_parallel_rank(self) -> int:
        """
        Get pipeline model parallel rank

        Returns:
            int: pipeline model parallel rank
        """
        if self._pipeline_model_parallel_rank is not None:
            return self._pipeline_model_parallel_rank

        return dist.get_rank(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_rank(self, rank: int) -> None:
        """
        Set pipeline model parallel rank

        Args:
            rank (int): pipeline model parallel rank
        """

        self._pipeline_model_parallel_rank = rank

    def is_pipeline_fist_stage(self) -> bool:
        """
        Return `True` if in the first pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is first
        """
        return self.get_pipeline_model_parallel_rank() == 0

    def is_pipeline_last_stage(self) -> bool:
        """
        Return `True` if in the last pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is last
        """
        return self.get_pipeline_model_parallel_rank() == (self.get_pipeline_model_parallel_world_size() - 1)

    def get_tensor_model_parallel_src_rank(self) -> int:
        """
        Calculate the global rank corresponding to the first local rank in the tensor model parallel group.

        Returns:
            int: tensor model parallel source rank
        """

        global_rank = dist.get_rank()
        local_world_size = self.get_tensor_model_parallel_world_size()
        return (global_rank // local_world_size) * local_world_size

    def get_pipeline_model_parallel_fist_rank(self):
        """
        Get the first pipeline model parallel rank

        Returns:
            int: the first pipeline model parallel rank
        """
        return self._pipeline_global_ranks[0]

    def get_pipeline_model_parallel_last_rank(self):
        """
        Get the last pipeline model parallel rank

        Returns:
            int: the last pipeline model parallel rank
        """
        return self._pipeline_global_ranks[self.get_pipeline_model_parallel_world_size() - 1]

    def get_pipeline_model_parallel_next_rank(self) -> int:
        """
        Get the next pipeline model parallel rank comparison with current stage.

        Returns:
            int: the next pipeline model parallel rank
        """
        assert self._pipeline_global_ranks is not None, "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self) -> int:
        """
        Get the previous pipeline model parallel rank comparison with current stage.

        Returns:
            int: the previous pipeline model parallel rank
        """
        assert self._pipeline_global_ranks is not None, "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe - 1) % world_size]

    def get_data_parallel_world_size(self) -> int:
        """
        Get data parallel world size

        Returns:
            int: data parallel world size
        """

        return dist.get_world_size(self.get_data_parallel_group())

    def get_data_parallel_rank(self) -> int:
        """
        Get data parallel rank

        Returns:
            int: data parallel rank
        """
        return dist.get_rank(self.get_data_parallel_group())

    def destroy_model_parallel(self) -> None:
        """
        Destroy all the model parallel groups
        """

        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_group = None
        self._data_parallel_group = None

    def _broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: broadcast tensor
        """
        return inputs

    def _reduce(self, inputs: Tensor):
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """

        if self.get_tensor_model_parallel_world_size() == 1:
            return inputs

        dist.all_reduce(inputs, group=self.get_tensor_model_parallel_group())
        return inputs

    def _scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        last_dim_size = inputs.size()[last_dim] // world_size

        inputs_list = torch.split(
            tensor=inputs,
            split_size_or_sections=last_dim_size,
            dim=last_dim,
        )

        rank = self.get_tensor_model_parallel_rank()
        return inputs_list[rank].contiguous()

    def _gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        rank = self.get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(inputs) for _ in range(world_size)]
        tensor_list[rank] = inputs
        torch.distributed.all_gather(tensor_list, inputs, group=self.get_tensor_model_parallel_group())
        return torch.cat(tensor_list, dim=last_dim).contiguous()

    def broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor):

        Returns:
            Tensor: broadcast tensor
        """

        if self._enable_grad(inputs):
            return self._broadcast_fn.apply(inputs)
        else:
            return self._broadcast(inputs)

    def reduce(self, inputs: Tensor) -> Tensor:
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """

        if self._enable_grad(inputs):
            return self._reduce_fn.apply(inputs)
        else:
            return self._reduce(inputs)

    def scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """

        if self._enable_grad(inputs):
            return self._scatter_fn.apply(inputs)
        else:
            return self._scatter(inputs)

    def gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """

        if self._enable_grad(inputs):
            return self._gather_fn.apply(inputs)
        else:
            return self._gather(inputs)

    @staticmethod
    def _enable_grad(inputs: Tensor) -> bool:
        """
        Check current tensor is enabled to pass gradient.

        Args:
            inputs (Tensor): input tensor

        Returns:
            bool: whether gradient can be passed or not
        """
        return torch.is_grad_enabled() and inputs.requires_grad

    def _initialize_functions(self):
        class Broadcast(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._broadcast(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._reduce(inputs)

        class Reduce(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._reduce(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._broadcast(inputs)

        class Scatter(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._scatter(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._gather(inputs)

        class Gather(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._gather(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._scatter(inputs)

        return {
            "broadcast": Broadcast,
            "reduce": Reduce,
            "scatter": Scatter,
            "gather": Gather,
        }

    @staticmethod
    def initialize_distributed():
        """Initialize torch.distributed and mpu."""
        if not torch.distributed.is_initialized():
            rank = int(os.getenv("RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 1))
            device_count = torch.cuda.device_count()

            if device_count > 0:
                device = rank % device_count
                torch.cuda.set_device(device)

            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += master_ip + ":" + master_port
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
                init_method=init_method,
            )

    def synchronize_across_tensor_model_parallel_world(self, *inputs, **kwargs):
        if not dist.is_initialized():
            return

        for _input in inputs:
            if _input is None or isinstance(_input, str):
                break
            elif torch.is_tensor(_input):
                if not _input.is_contiguous():
                    _input = _input.contiguous()
                dist.broadcast(
                    tensor=_input,
                    src=self.get_tensor_model_parallel_src_rank(),
                    group=self.get_tensor_model_parallel_group(),
                )
            elif isinstance(_input, Iterable):
                self.synchronize_across_tensor_model_parallel_world(*_input)

        for k in kwargs:
            if kwargs[k] is None or isinstance(kwargs[k], str):
                break
            elif torch.is_tensor(kwargs[k]):
                if not kwargs[k].is_contiguous():
                    kwargs[k] = kwargs[k].contiguous()
                dist.broadcast(
                    tensor=kwargs[k],
                    src=self.get_tensor_model_parallel_src_rank(),
                    group=self.get_tensor_model_parallel_group(),
                )
            elif isinstance(kwargs[k], Iterable):
                self.synchronize_across_tensor_model_parallel_world(**kwargs[k])


@dataclass
class Layer:
    name: str = None
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    replace: dict = None
    n_fused: int = None
    reversed: bool = None
    scale_attention: bool = None
    local_attention: bool = None


class LayerPolicy(ABC):
    """
    Layer policy for tensor model parallelism and kernel fusion.
    You can check more details here: https://github.com/tunib-ai/parallelformers/blob/main/POLICY.md

    References:
        The design of the Layer policy class is inspired by Microsoft DeepSpeed.
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py
    """

    @staticmethod
    def replace_arguments(layer, world_size, config):
        pass

    @staticmethod
    def attn_qkv(layer, config):
        return []

    @staticmethod
    def attn_out(layer, config):
        return []

    @staticmethod
    def attn_norm(layer, config):
        return []

    @staticmethod
    def mlp_in(layer, config):
        return []

    @staticmethod
    def mlp_out(layer, config):
        return []

    @staticmethod
    def mlp_norm(layer, config):
        return []

    @staticmethod
    def word_embedding(model, config):
        return []

    @staticmethod
    def layerwise_copy_to_all(layer, config):
        return []

    @staticmethod
    def modelwise_copy_to_all(model, config):
        return []

    @staticmethod
    def original_layer_class():
        raise NotImplementedError


def vocab_size_with_padding(vocab_size, make_vocab_size_divisible_by, world_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = vocab_size
    multiple = make_vocab_size_divisible_by * world_size

    while (after % multiple) != 0:
        after += 1

    return after


class ParallelizationEngine(object):
    """
    Integration with Parallelformers
    https://github.com/tunib-ai/parallelformers
    """

    def __init__(
        self,
        policy: LayerPolicy,
        hold_params: bool,
        vocab_parallel_embedding: bool,
        additional_layers: List[Layer] = None,
    ):
        self.hold_params = hold_params
        self.vocab_parallel_embedding = vocab_parallel_embedding
        self.device = torch.cuda.current_device()
        self.policy = policy
        self.param_dict = {}

        if additional_layers is None:
            self.additional_layers = []
        else:
            self.additional_layers = additional_layers

    def parallelize(self, model, mpu):
        config = model.config

        if dist.is_initialized() and mpu is not None:
            gpu_index = mpu.get_tensor_model_parallel_rank()
            world_size = mpu.get_tensor_model_parallel_world_size()
        else:
            gpu_index = 0
            world_size = 1

        self._process_additional_layers(
            model=model,
            mpu=mpu,
            additional_layers=self.additional_layers,
            config=config,
        )

        self._process_repeated_layers(
            model=model,
            mpu=mpu,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        self._process_word_embedding(
            model=model,
            mpu=mpu,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        non_cuda_tensors = []
        for k, v in dict(model.state_dict()).items():
            if not v.is_cuda:
                if torch.is_tensor(v):
                    non_cuda_tensors.append(k)

        if len(non_cuda_tensors) > 0:
            raise Exception(f"{non_cuda_tensors} are not CUDA tensors now.")

    def _process_word_embedding(self, model, mpu, gpu_index, world_size, config):
        hidden_size = config.hidden_size
        original_vocab_size = config.vocab_size

        if self.vocab_parallel_embedding and hasattr(config, "make_vocab_size_divisible_by"):
            make_vocab_size_divisible_by = config.make_vocab_size_divisible_by
            efficient_vocab_size = vocab_size_with_padding(
                original_vocab_size, make_vocab_size_divisible_by, world_size
            )

            size_of_padding = efficient_vocab_size - original_vocab_size

            assert (
                efficient_vocab_size % world_size == 0
            ), "effective vocab size must be divisible by tensor model parallel world size."

            assert size_of_padding > 0, "pad size for vocab parallel embedding must be positive."
        else:
            size_of_padding = 0

        for layer in self.policy.word_embedding(model, config):
            if size_of_padding > 0:
                padding = torch.zeros(
                    size_of_padding,
                    hidden_size,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.weight = torch.cat([layer.weight, padding], dim=0)

            # embedding layer has (vocab_size, embedding_dims) tensor
            # so dimension for chunking must be 0 for vocab parallel embedding
            if self.vocab_parallel_embedding:
                chunked_weight = torch.chunk(layer.weight, world_size, dim=0)
                layer.weight.data = chunked_weight[gpu_index].to(self.device)
            else:
                layer.weight.data = layer.weight.to(self.device)

            self._postprocess(layer, {"mpu": mpu})

    def _process_repeated_layers(self, model, mpu, gpu_index, world_size, config):
        for _, child in model.named_children():
            if isinstance(child, self.policy.original_layer_class()):
                self.policy.replace_arguments(
                    layer=child,
                    world_size=world_size,
                    config=config,
                )

                parameters = [
                    self._column_slice(
                        self.policy.attn_qkv(child, config),
                        world_size=world_size,
                        gpu_index=gpu_index,
                    ),
                    self._row_slice(
                        self.policy.attn_out(child, config),
                        world_size=world_size,
                        gpu_index=gpu_index,
                    ),
                    self._column_slice(
                        self.policy.mlp_in(child, config),
                        world_size=world_size,
                        gpu_index=gpu_index,
                    ),
                    self._row_slice(
                        self.policy.mlp_out(child, config),
                        world_size=world_size,
                        gpu_index=gpu_index,
                    ),
                ]

                copy_to_all = (
                    self.policy.attn_norm(child, config)
                    + self.policy.mlp_norm(child, config)
                    + self.policy.layerwise_copy_to_all(child, config)
                )

                for layer in copy_to_all:
                    if layer.weight is not None:
                        layer.weight.data = layer.weight.to(self.device)
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.to(self.device)

                    parameters.append([layer])

                for layers in parameters:
                    for layer in layers:
                        self._postprocess(
                            layer,
                            {
                                "mpu": mpu,
                                "reversed": layer.reversed,
                                "skip_bias_add": False,
                                "parallel_output": True,
                            },
                        )

            self._process_repeated_layers(
                model=child,
                mpu=mpu,
                gpu_index=gpu_index,
                world_size=world_size,
                config=config,
            )

    def _slice_layer(
        self,
        layer: Layer,
        dim: int,
        world_size: int,
        gpu_index: int,
        slice_bias: bool,
    ) -> Layer:
        """
        Slice tensors into rows or columns as described in the Megatron-LM paper
        """

        dim = dim if not layer.reversed else abs(dim - 1)
        n_fused = 1 if not layer.n_fused else layer.n_fused

        if layer.weight is not None:
            if layer.weight.dim() >= 1:
                weight = layer.weight.chunk(n_fused * world_size, dim=dim)
                if n_fused > 1:
                    weight = self._realign_fused_tensors(weight, world_size)
                layer.weight.data = weight[gpu_index].to(self.device)
            else:
                layer.weight.data = layer.weight.to(self.device)

        if layer.bias is not None:
            if slice_bias is True:
                if layer.bias.dim() >= 1:
                    bias = layer.bias.chunk(n_fused * world_size, dim=0)
                    if n_fused > 1:
                        bias = self._realign_fused_tensors(bias, world_size)
                    layer.bias.data = bias[gpu_index].to(self.device)
                else:
                    layer.bias.data = layer.bias.to(self.device)
            else:
                layer.bias.data = layer.bias.to(self.device)

        return layer

    @staticmethod
    def _realign_fused_tensors(tensor, world_size):
        ranks = (len(tensor) + world_size - 1) // world_size
        tensor = [tensor[i * world_size : (i + 1) * world_size] for i in range(ranks)]
        tensor = list(map(lambda x: torch.cat([*x], dim=-1), zip(*tensor)))
        return tensor

    def _column_slice(self, layers, world_size, gpu_index):
        return [
            self._slice_layer(
                layer=layer,
                dim=0,
                world_size=world_size,
                gpu_index=gpu_index,
                slice_bias=True,
            )
            for layer in layers
        ]

    def _row_slice(self, layers, world_size, gpu_index):
        return [
            self._slice_layer(
                layer=layer,
                dim=1,
                world_size=world_size,
                gpu_index=gpu_index,
                slice_bias=False,
            )
            for layer in layers
        ]

    def _process_additional_layers(
        self,
        model,
        mpu,
        additional_layers,
        config,
    ):
        additional_layers += self.policy.modelwise_copy_to_all(model, config)

        for layer in additional_layers:
            if layer.weight is not None:
                layer.weight.data = layer.weight.to(self.device)
            if layer.bias is not None:
                layer.bias.data = layer.bias.to(self.device)
            self._postprocess(layer, {"mpu": mpu})

    def _postprocess(self, layer, attributes):
        if layer.replace is not None:
            for k, v in layer.replace.items():
                for k_, v_ in attributes.items():
                    setattr(k, k_, v_)
                k.__class__ = v

        if self.hold_params:
            if layer.name in self.param_dict:
                self.param_dict[layer.name].append(layer)
            else:
                self.param_dict[layer.name] = [layer]


class ParallelizationMixin(object):
    def _get_base_model(self):
        raise NotImplementedError

    def _get_layer_policies(self):
        raise NotImplementedError

    def _get_head_layers(self):
        return []

    def _parallelize(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        vocab_parallel_embedding: bool = None,
    ):
        assert (
            self.is_tensor_parallelizable is True
        ), f"{self.__class__.__name__} does not support tensor model parallelism."
        assert (
            pipeline_model_parallel_size == 1
        ), "Currently, We only support tensor model parallelism, please set param `pipeline_model_parallel_size` to 1"
        assert tensor_model_parallel_size >= 1, "param `tensor_model_parallel_size` must be positive."
        assert (
            tensor_model_parallel_size & (tensor_model_parallel_size - 1) == 0
        ), "param `tensor_model_parallel_size` must be power of 2."

        if vocab_parallel_embedding is None:
            if hasattr(self.config, "vocab_parallel_embedding"):
                vocab_parallel_embedding = self.config.vocab_parallel_embedding
        else:
            self.config.vocab_parallel_embedding = vocab_parallel_embedding

        mpu = MPU(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )

        for policy in self._get_layer_policies():
            engine = ParallelizationEngine(
                hold_params=False,
                policy=policy,
                vocab_parallel_embedding=vocab_parallel_embedding,
                additional_layers=self._get_head_layers(),
            )

            engine.parallelize(
                model=self._get_base_model(),
                mpu=mpu,
            )

        setattr(self, "mpu", mpu)
        # This allows the Trainer to call the MPU for data + model parallelism
        # example `ddp = DistributedDataParallel(..., process_group=model.mpu.get_data_parallel_group())`

    @classmethod
    def from_pretrained_with_parallel(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        vocab_parallel_embedding: bool = None,
        fp16: bool = False,
        *model_args,
        **kwargs,
    ):
        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if fp16 is True:
            model = model.half()

        model._parallelize(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            vocab_parallel_embedding=vocab_parallel_embedding,
        )

        return model

    @classmethod
    def from_config_with_parallel(
        cls,
        config,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        vocab_parallel_embedding: bool = None,
        fp16: bool = False,
        *model_args,
        **kwargs,
    ):

        for k, v in kwargs.items():
            setattr(config, k, v)

        model = cls(config)

        if fp16 is True:
            model = model.half()

        model._parallelize(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            vocab_parallel_embedding=vocab_parallel_embedding,
        )

        return model
