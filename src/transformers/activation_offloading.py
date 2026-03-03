# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/pytorch/torchtune.


import psutil
import torch
from accelerate import logging
from accelerate.utils.versions import is_torch_version
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from transformers import is_torch_npu_available


if is_torch_npu_available():
    import torch_npu  # noqa: F401

# Import DTensor for FSDP v2 support with version-aware import path
DTensor = None
if torch.distributed.is_available():
    try:
        if is_torch_version(">=", "2.5.0"):
            from torch.distributed.tensor import DTensor
        else:
            # from torch 2.0.0 (oldest supported accelerate torch version), DTensor is in torch.distributed._tensor
            from torch.distributed._tensor import DTensor
    except (ImportError, AttributeError):
        DTensor = None

logger = logging.get_logger(__name__)


def _get_unique_tensor_key(tensor: torch.Tensor) -> tuple:
    """
    Get a unique key for a tensor based on its storage pointer and dtype. This allows deduplication of tensors that
    share the same underlying storage. From:
    https://github.com/volcengine/verl/blob/main/verl/utils/activation_offload.py

    Args:
        tensor: The tensor to get the key for

    Returns:
        A tuple of (storage_pointer, dtype) that uniquely identifies the tensor's storage
    """
    # Handle special tensor types - primarily for FSDP v2 DTensor
    actual_tensor = tensor

    # For DTensor (FSDP v2), extract the local tensor
    if DTensor is not None and isinstance(tensor, DTensor) and hasattr(tensor, "_local_tensor"):
        actual_tensor = tensor._local_tensor

    # Try to get storage pointer, but fall back to tensor id if not accessible
    try:
        storage_ptr = actual_tensor.untyped_storage().data_ptr() + actual_tensor.storage_offset()
    except (RuntimeError, AttributeError):
        # For tensors with invalid storage, use tensor id
        # This won't enable deduplication for these tensors, but allows offloading to work
        storage_ptr = id(actual_tensor)

    return (storage_ptr, actual_tensor.dtype)


class OffloadActivations(saved_tensors_hooks):
    """
    Context manager under which activation tensors created in the forward pass will be offloaded.

    Enable the memory efficiency technique of activation offloading, where activations bigger than `min_offload_size`
    bytes will be offloaded to CPU in the forward and brought back in the backward. This is in contrast to maintaining
    the activation on GPU VRAM throughout the program.

    This manager contains the option of using one additional CUDA stream to handle the communication between CUDA and
    CPU, which is intended to overlap with the default computation stream to improve runtime. We designed
    synchronization with a few heuristics for optimizing the tradeoff between runtime vs memory usage.

    Args:
        use_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether to offloaded Tensor will be placed in pinned memory on the CPU. Pinned memory allows the Tensor to
            be moved back onto GPU more quickly but is a limited resource.
        use_streams (`bool`, *optional*, defaults to `True`):
            Whether to use streams for performance optimization where the communications get overlapped with the
            computation. Requires a torch build after torch-2.5.0.
        min_offload_size (`int`, *optional*, defaults to `1024`):
            Minimum number of bytes a Tensor must be in order to qualify for offloading. If the tensor is too small, we
            do not want to waste bandwidth and resources moving it to CPU and back.
        max_fwd_stash_size (`int`, *optional*, defaults to `5`):
            Maximum size of the forward stash, or the maximum number of consecutive activations to keep alive during
            the forward pass. This number must be at least 1. Keeping alive more activations will potentially allow
            more overlap between the communication and compute streams at the cost of increasing memory usage. Keeping
            alive fewer activations will conserve memory, but may cause poor overlap between the streams, increasing
            runtime.

    Raises:
        ValueError: if `max_fwd_stash_size` is not at least `1`.

    Example:
    ```python
    >>> with OffloadActivations():
    ...     outputs = model(inputs, labels=labels)
    >>> loss = outputs.loss
    >>> loss.backward()
    ```
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        min_offload_size: int = 1024,
        max_fwd_stash_size: int = 5,
    ) -> None:
        self.use_streams = use_streams

        self.min_tensor_size_bytes = min_offload_size  # we don't want to bother with small tensors
        self.tracker = {}  # tensor_id => (new_tensor, if_modified)  ---> track what saved/offloaded tensors are where
        self.tensor_id = 0
        self.is_first_forward_call = True
        self.is_first_backward_call = True
        self.is_first_forward_pass = True

        # Storage deduplication: maps storage key to tensor_id to avoid offloading same storage multiple times
        self.storage_to_tensor_id = {}

        # Parameter filtering: track parameter storage pointers to skip them during offloading
        self.param_storages = set()

        # Managing cpu memory
        self.use_pin_memory = use_pin_memory
        self.virtual_memory_safe_pct = 60  # we should not exceed this percentage of memory

        self.accelerator_type = (
            torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        )
        # NOTE: xpu doesn't have `default_stream` API, use `current_stream` instead
        if self.accelerator_type == "xpu":  # comp stream
            self.s0 = torch.xpu.current_stream()
        elif is_torch_npu_available() and self.accelerator_type == "npu":
            self.s0 = torch.npu.current_stream()
        else:
            self.s0 = torch.cuda.default_stream()

        # For streaming
        if self.use_streams:
            if self.accelerator_type == "xpu":  # comms stream
                self.s1 = torch.xpu.Stream()
            elif self.accelerator_type == "npu":
                self.s1 = torch.npu.Stream()
            else:
                self.s1 = torch.cuda.Stream()
            self.fwd_stash = {}  # tensor_id => (activation, ev1)
            if max_fwd_stash_size < 1:
                raise ValueError(f"max_fwd_stash_size should be at least 1 but is {max_fwd_stash_size}")
            self.max_fwd_stash_size = max_fwd_stash_size
            self.bwd_tensor_stash = {}  # tensor_id => activation
            self.bwd_ev_stash = {}  # tensor_id => ev0
            self.curr_graph_id = None
            self.curr_autograd_node = None

        # -------- platform util functions -------- #
        def verify_sufficient_virtual_memory():
            curr_pct = get_cpu_ram_pct()
            if curr_pct > self.virtual_memory_safe_pct:
                logger.warning(f"{curr_pct=}% > {self.virtual_memory_safe_pct=}% of virtual memory used")

        def get_cpu_ram_pct() -> float:
            # get the percentage of memory used by the system
            return psutil.virtual_memory().percent

        def get_tensor_id() -> int:
            # create a unique id for each tensor we are managing
            self.tensor_id += 1
            return self.tensor_id

        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return x.element_size() * x.nelement()  # x.element_size() * x._base_storage().nbytes()

        # -------- core pack / unpack work -------- #
        def pack_tensor(activation: torch.Tensor) -> int:
            # activations are passed in during forward pass - from here we take over and return a unique id
            if self.is_first_forward_call:
                if len(self.tracker) != 0:
                    raise ValueError("Backward pass should have cleared tracker of all tensors")

                # set training phase trackers
                self.is_first_forward_call = False
                self.is_first_backward_call = True
                # Reset deduplication map for new forward pass
                self.storage_to_tensor_id = {}

            # query for basic tensor info
            num_bytes = get_num_bytes_tensor(activation)
            tensor_id = get_tensor_id()

            # Check for tensor deduplication using storage pointer
            # If this storage is already being tracked, we still create a new tensor_id
            # but don't offload again (just keep the tensor in GPU)
            storage_key = _get_unique_tensor_key(activation)
            if storage_key in self.storage_to_tensor_id:
                # Storage already offloaded - don't offload again, just track the reference
                self.tracker[tensor_id] = (activation, False, None, None, None)  # Keep on GPU, don't offload
                return tensor_id

            # Check if tensor is on CPU (skip offloading)
            if activation.device.type not in ["cuda", "xpu", "npu"]:
                self.tracker[tensor_id] = (activation, False, None, None, None)
                return tensor_id

            # Check if tensor is too small
            if num_bytes < self.min_tensor_size_bytes:
                self.tracker[tensor_id] = (activation, False, None, None, None)
                return tensor_id

            # Check if tensor is a parameter or buffer
            if isinstance(activation, torch.nn.Parameter) or (
                hasattr(torch.nn, "Buffer") and isinstance(activation, torch.nn.Buffer)
            ):
                self.tracker[tensor_id] = (activation, False, None, None, None)
                return tensor_id

            # Check if tensor is an FP8 tensor (TorchAO) - skip offloading as they're already compressed
            tensor_class_name = type(activation).__name__
            if tensor_class_name in ["Float8TrainingTensor", "ScaledMMConfig", "LinearMMConfig"]:
                self.tracker[tensor_id] = (activation, False, None, None, None)
                return tensor_id

            # Check if tensor storage is a model parameter (for FSDP compatibility)
            try:
                # Extract actual tensor for DTensor
                check_tensor = activation
                if DTensor is not None and isinstance(activation, DTensor) and hasattr(activation, "_local_tensor"):
                    check_tensor = activation._local_tensor

                if check_tensor.untyped_storage().data_ptr() in self.param_storages:
                    self.tracker[tensor_id] = (activation, False, None, None, None)
                    return tensor_id
            except (RuntimeError, AttributeError):
                # If we can't get data_ptr, skip this check
                pass

            # Tensor qualifies for offloading
            if self.use_streams:
                # First, sync back and dereference previously offloaded tensors
                # as the offloading should be done sufficiently long ago.
                for id in list(self.fwd_stash.keys()):
                    if id <= tensor_id - self.max_fwd_stash_size:
                        _, ev = self.fwd_stash[id]
                        self.s0.wait_event(ev)
                        del self.fwd_stash[id]
                    else:
                        break

                # Sync in, offload, and add an event to sync back later
                self.s1.wait_stream(self.s0)

            stream = self.s1 if self.use_streams else self.s0
            if self.accelerator_type == "xpu":
                stream_ctx = torch.xpu.stream(stream)
            elif self.accelerator_type == "npu":
                stream_ctx = torch.npu.stream(stream)
            else:
                stream_ctx = torch.cuda.stream(stream)
            with stream_ctx:
                # Save original stride and shape information
                original_stride = activation.stride()
                original_storage_offset = activation.storage_offset()
                original_shape = activation.size()

                # Check if tensor has broadcast dimensions (stride == 0)
                # If so, copy the underlying storage directly instead of materializing the broadcast
                has_broadcast = 0 in original_stride

                if has_broadcast:
                    # Copy only the actual underlying storage, not the materialized broadcast
                    # Create CPU tensor with same storage size as original
                    storage_size = activation.untyped_storage().size()
                    cpu_storage = torch.empty(
                        storage_size // activation.element_size(),
                        dtype=activation.dtype,
                        pin_memory=self.use_pin_memory,
                        device="cpu",
                    )
                    # Copy the raw storage
                    cpu_storage_view = torch.as_strided(
                        activation, size=(storage_size // activation.element_size(),), stride=(1,), storage_offset=0
                    )
                    cpu_storage.copy_(cpu_storage_view, non_blocking=True)
                    cpu_tensor = cpu_storage
                else:
                    # No broadcast - use normal contiguous copy
                    cpu_tensor = torch.empty_like(activation, pin_memory=self.use_pin_memory, device="cpu")
                    cpu_tensor.copy_(activation, non_blocking=True)

                # Store CPU tensor along with stride information
                self.tracker[tensor_id] = (
                    cpu_tensor,
                    True,  # True = (in future) modified
                    original_stride,  # Save original GPU stride
                    original_storage_offset,  # Save original storage offset
                    original_shape,  # Save original shape for broadcast restoration
                )

            if self.use_streams:
                event = self.s1.record_event()

                # Stash to keep activation alive til s1 is done
                self.fwd_stash[tensor_id] = (activation, event)

            # Track this storage for deduplication
            self.storage_to_tensor_id[storage_key] = tensor_id

            return tensor_id

        def unpack_tensor_single_stream(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False

            if unpack_tensor_id not in self.tracker:
                raise ValueError(f"Untracked tensor with id {unpack_tensor_id}")

            (
                maybe_accelerator_tensor,
                modified,
                original_stride,
                original_storage_offset,
                original_shape,
            ) = self.tracker[unpack_tensor_id]

            if modified:
                # Restore tensor to GPU
                accelerator_tensor = maybe_accelerator_tensor.to(self.accelerator_type, non_blocking=True)
                # Restore original stride if we saved it (handles both broadcast and non-broadcast cases)
                if original_stride is not None:
                    accelerator_tensor = torch.as_strided(
                        accelerator_tensor,
                        size=original_shape,
                        stride=original_stride,
                        storage_offset=original_storage_offset,
                    )
                maybe_accelerator_tensor = accelerator_tensor

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            # Only set is_first_forward_call to True when all tensors have been unpacked
            if len(self.tracker) == 0:
                self.is_first_forward_call = True
            return maybe_accelerator_tensor

        def unpack_tensor_with_streams(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                self.curr_graph_id = torch._C._current_graph_task_id()

                def wait_and_del_remaining_references() -> None:
                    for id in list(self.bwd_tensor_stash.keys()):
                        if id in self.bwd_ev_stash:
                            event = self.bwd_ev_stash[id]
                            self.s1.wait_event(event)
                        del self.bwd_tensor_stash[id]

                # Register a callback to the end of autograd to clean everything up
                torch.autograd.variable.Variable._execution_engine.queue_callback(wait_and_del_remaining_references)

                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False

            if unpack_tensor_id not in self.tracker:
                raise ValueError(f"untracked tensor with id {unpack_tensor_id}")

            (
                maybe_accelerator_tensor,
                modified,
                original_stride,
                original_storage_offset,
                original_shape,
            ) = self.tracker[unpack_tensor_id]

            if modified:
                # Get data on the current autograd node
                graph_id = torch._C._current_graph_task_id()
                node = torch._C._current_autograd_node()
                prev_node_ids = []

                # If we're on a new node, mark prev node's tensors to be freed later
                if graph_id == self.curr_graph_id and self.curr_autograd_node != node:
                    self.curr_autograd_node = node
                    prev_node_ids = list(self.bwd_tensor_stash.keys())

                brought_back_from_cpu = True
                if unpack_tensor_id in self.fwd_stash:
                    maybe_accelerator_tensor = self.fwd_stash[unpack_tensor_id][0]
                    brought_back_from_cpu = False
                else:
                    # Kick off the process to bring tensors back
                    if self.accelerator_type == "xpu":
                        stream_ctx = torch.xpu.stream(self.s1)
                    elif self.accelerator_type == "npu":
                        stream_ctx = torch.npu.stream(self.s1)
                    else:
                        stream_ctx = torch.cuda.stream(self.s1)
                    with stream_ctx:
                        # Restore tensor to GPU
                        accelerator_tensor = maybe_accelerator_tensor.to(self.accelerator_type, non_blocking=True)
                        # Restore original stride if we saved it (handles both broadcast and non-broadcast cases)
                        if original_stride is not None:
                            accelerator_tensor = torch.as_strided(
                                accelerator_tensor,
                                size=original_shape,
                                stride=original_stride,
                                storage_offset=original_storage_offset,
                            )
                        maybe_accelerator_tensor = accelerator_tensor

                    # Tell comp stream to wait for the info to be loaded before executing
                    self.s0.wait_stream(self.s1)

                    # Stash the tensor to keep memory alive until compute stream is complete
                    self.bwd_tensor_stash[unpack_tensor_id] = maybe_accelerator_tensor

                    # Note: [Track views of the unpacked]
                    # Why do we get the use count of the unpacked tensor here? We want an
                    # initial count to compare to later, during the post-hook of the
                    # backward node, when we need to decide whether we're allowed to free
                    # the tensor yet. In what obscure cases must we delay freeing the
                    # tensor (and thus call record_stream)?
                    # 1. Any of the outputs of the backward node is a view of the unpacked
                    #    tensor.
                    # 2. In the case that this unpacked tensor will be used in a
                    #    checkpointed region, if one of the recomputed saved tensors ends
                    #    up as a view of the unpacked tensor.
                    # 3. The user abuses the system somehow and manually relies on the
                    #    unpacked tensor to exist after the backward node has executed.
                    if self.accelerator_type == "npu":
                        storage_refcount = torch_npu._C._storage_Use_Count(
                            maybe_accelerator_tensor.untyped_storage()._cdata
                        )
                    else:
                        storage_refcount = torch._C._storage_Use_Count(
                            maybe_accelerator_tensor.untyped_storage()._cdata
                        )

                def hook(outputs, inputs):
                    # create events for the current node inputs/outputs if they were streamed in
                    if brought_back_from_cpu:
                        # See Note: [Track views of the unpacked]
                        # IF any of the outputs is a view of the tensor, OR if a view of
                        # the tensor has been saved as a part of checkpoint's recompute
                        # process, OR the user has abusedly incurred a reference on the
                        # unpacked tensor, THEN the tensor might be used later and we
                        # cannot presume to delete it after only the current node is
                        # done! So we use our frenemy, record_stream, to ensure the
                        # Tensor stays unmessed with until it's done getting used in the
                        # compute stream (s0 here). Note that the con here is we introduce
                        # non-deterministic (thus higher) memory usage, but this case
                        # should not happen often.
                        # Check if tensor still exists (might have been cleaned up by a previous node)
                        if unpack_tensor_id in self.bwd_tensor_stash:
                            unpacked_tensor = self.bwd_tensor_stash[unpack_tensor_id]
                            if self.accelerator_type == "npu":
                                storage_count = torch_npu._C._storage_Use_Count(
                                    unpacked_tensor.untyped_storage()._cdata
                                )
                            else:
                                storage_count = torch._C._storage_Use_Count(unpacked_tensor.untyped_storage()._cdata)
                            if storage_count > storage_refcount:
                                unpacked_tensor.record_stream(self.s0)
                                del self.bwd_tensor_stash[unpack_tensor_id]
                            else:
                                event = self.s0.record_event()
                                self.bwd_ev_stash[unpack_tensor_id] = event

                    # if there are still things in the fwd_stash, get rid of them as we're in bwd now
                    for id in list(self.fwd_stash.keys()):
                        _, ev = self.fwd_stash[id]
                        self.s0.wait_event(ev)
                        del self.fwd_stash[id]

                    # wait on prev node's events and del those
                    for id in prev_node_ids:
                        # Only wait on events that exist (some tensors may have used record_stream instead)
                        if id in self.bwd_ev_stash:
                            event = self.bwd_ev_stash[id]
                            self.s1.wait_event(event)
                            del self.bwd_ev_stash[id]
                        if id in self.bwd_tensor_stash:
                            del self.bwd_tensor_stash[id]

                    return outputs

                node.register_hook(hook)

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            # Only set is_first_forward_call to True when all tensors have been unpacked
            if len(self.tracker) == 0:
                self.is_first_forward_call = True
            return maybe_accelerator_tensor

        unpack_tensor = unpack_tensor_with_streams if self.use_streams else unpack_tensor_single_stream
        super().__init__(pack_tensor, unpack_tensor)

    def update_model_params(self, model: nn.Module):
        """
        Update the set of parameter storage pointers from the model. This allows filtering out model parameters during
        offloading, which is especially important for FSDP models where parameters may not be detected by isinstance
        checks.

        For FSDP v2, this method handles DTensor parameters which may be sharded across ranks and not have valid local
        storage on all ranks. We extract the local tensor from DTensors using _local_tensor when available.

        Args:
            model: The model whose parameters should be tracked
        """
        param_storages = set()

        for p in model.parameters():
            # For FSDP v2: extract local tensor from DTensor
            actual_tensor = p
            if DTensor is not None and isinstance(p, DTensor) and hasattr(p, "_local_tensor"):
                actual_tensor = p._local_tensor

            # Try to get storage pointer
            try:
                storage_ptr = actual_tensor.untyped_storage().data_ptr()
                if storage_ptr != 0:
                    param_storages.add(storage_ptr)
            except RuntimeError:
                # Parameter doesn't have accessible storage (e.g., FSDP v2 sharded without local shard, FP8 parameters)
                # These will be caught by other checks (isinstance for Parameter, class name for FP8)
                continue

        self.param_storages = param_storages


class NoOpManager(saved_tensors_hooks):
    """
    A `saved_tensors_hook` manager used to disable any other `saved_tensors_hook` manager applied before. This relies
    on the behavior that only the most recently registered `saved_tensors_hook` will run.

    One example usage is to opt a local region of code out of activations offloading, which is usually applied globally
    to best track state.
    """

    def __init__(self) -> None:
        def noop(tensor):
            return tensor

        super().__init__(noop, noop)


def get_act_offloading_ctx_manager(
    model: nn.Module,
    use_pin_memory: bool = True,
    use_streams: bool = True,
    min_offload_size: int = 1024,
    max_fwd_stash_size: int = 5,
    warn_if_no_head: bool = True,
) -> OffloadActivations:
    """
    Returns the activation offloading context manager for the model. All but the last output Linear in every step will
    be offloaded.

    If activation offloading is enabled, we return the OffloadActivations context manager. If activation offloading is
    disabled, we return a NoOpManager context manager.

    Args:
        model (`nn.Module`):
            Model to wrap with the activation offloading context manager.
        use_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether to offloaded Tensor will be placed in pinned memory on the CPU. Pinned memory allows the Tensor to
            be moved back onto GPU more quickly but is a limited resource.
        use_streams (`bool`, *optional*, defaults to `True`):
            Whether to use streams for performance optimization where the communications get overlapped with the
            computation. Requires a torch build after torch-2.5.0.
        min_offload_size (`int`, *optional*, defaults to `1024`):
            Minimum number of bytes a Tensor must be in order to qualify for offloading. If the tensor is too small, we
            do not want to waste bandwidth and resources moving it to CPU and back.
        max_fwd_stash_size (`int`, *optional*, defaults to `5`):
            Maximum size of the forward stash, or the maximum number of consecutive activations to keep alive during
            the forward pass. This number must be at least 1. Keeping alive more activations will potentially allow
            more overlap between the communication and compute streams at the cost of increasing memory usage. Keeping
            alive fewer activations will conserve memory, but may cause poor overlap between the streams, increasing
            runtime.
        warn_if_no_head (`bool`, *optional*, defaults to `True`):
            Whether to warn if no output head is detected. If set to `False`, no warning will be raised if no output
            head is detected.

    Returns:
        `contextlib.ContextDecorator`:
            Activation offloading context manager for the model.
    """
    activations_handling_ctx = OffloadActivations(
        use_pin_memory=use_pin_memory,
        use_streams=use_streams,
        min_offload_size=min_offload_size,
        max_fwd_stash_size=max_fwd_stash_size,
    )

    # Update parameter storages to filter them during offloading (important for FSDP)
    activations_handling_ctx.update_model_params(model)

    # Below is our hack to disable offloading the last output Linear in every
    # step, as the cost for offloading the activation and then soon after bringing
    # it back is expensive.
    output_head_detected = False
    noop_ctx = NoOpManager()

    # Try to get the actual model if it's wrapped
    unwrapped_model = model
    if hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
    # check for PEFT models
    if hasattr(unwrapped_model, "base_model") and hasattr(unwrapped_model, "peft_config"):
        unwrapped_model = unwrapped_model.base_model

    # Check for different types of output heads
    if hasattr(unwrapped_model, "output"):
        if isinstance(unwrapped_model.output, nn.Module):
            unwrapped_model.output.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            unwrapped_model.output.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
            output_head_detected = True
        elif hasattr(unwrapped_model.output, "linear") and isinstance(unwrapped_model.output.linear, nn.Module):
            unwrapped_model.output.linear.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            unwrapped_model.output.linear.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
            output_head_detected = True

    # Check for HuggingFace model output heads
    elif hasattr(unwrapped_model, "lm_head"):
        unwrapped_model.lm_head.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
        unwrapped_model.lm_head.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
        output_head_detected = True

    # Check for decoder-based models
    elif hasattr(unwrapped_model, "decoder"):
        decoder = unwrapped_model.decoder
        if hasattr(decoder, "output"):
            decoder.output.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            decoder.output.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
            output_head_detected = True
        # Some models have lm_head in the decoder
        elif hasattr(decoder, "lm_head"):
            decoder.lm_head.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            decoder.lm_head.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
            output_head_detected = True

    # Check for transformer models with final layer norm
    elif hasattr(unwrapped_model, "final_layer_norm") or hasattr(unwrapped_model, "ln_f"):
        final_norm = getattr(unwrapped_model, "final_layer_norm", None) or unwrapped_model.ln_f
        final_norm.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
        final_norm.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
        output_head_detected = True

    # Check for models with head module
    elif hasattr(unwrapped_model, "head") and isinstance(unwrapped_model.head, nn.Module):
        unwrapped_model.head.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
        unwrapped_model.head.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)
        output_head_detected = True

    if not output_head_detected and warn_if_no_head:
        logger.warning(
            "During activation offloading, no output head was detected. If your model has an output head, it will be "
            "offloaded. This usually greatly slows training, given the large vocabulary size. To change this "
            "behavior, set your output head as model.output and make it an nn.Module. You can disable this warning by "
            "passing `warn_if_no_head=False`."
        )

    # Disable offloading for any Liger modules
    for name, module in unwrapped_model.named_modules():
        if "liger" in name.lower():
            module.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            module.register_forward_hook(lambda *args: noop_ctx.__exit__(), always_call=True)

    return activations_handling_ctx