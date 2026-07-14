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

"""SonicMoE integration: fused MoE using CuteDSL kernels from `kernels-community/sonic-moe`.

Provides `SONIC_MOE_HANDLE` registered as "sonicmoe" in the ExpertsInterface.
Requirements: CUDA, `kernels`, `nvidia-cutlass-dsl`, has_gate=True.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils import logging
from .hub_kernels import lazy_load_kernel
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class SonicMoE:
    """Entry points exposed by the `kernels-community/sonic-moe` kernel."""

    activation_type_enum: type
    moe_general_routing_inputs: Callable


def _load_sonicmoe_kernel() -> SonicMoE:
    """
    Load sonic-moe once and return its entry points.

    Raises `ImportError` if CUDA/hardware requirements are not met, or if the kernel or
    required symbols are not found.
    """

    if not torch.cuda.is_available():
        raise ImportError(
            "sonic-moe kernel requires CUDA, but CUDA is not available. Use a different `experts_implementation`."
        )

    # sonic-moe requires Hopper (SM90) or newer
    major = torch.cuda.get_device_capability()[0]
    if major < 9:
        raise ImportError(
            f"sonic-moe requires a Hopper (SM90+) or newer GPU, but the current device "
            f"has compute capability {major}.x. Use a different `experts_implementation`."
        )

    # check if cutlass-dsl is installed
    from cutlass.utils.hardware_info import HardwareInfo

    # sonic-moe JIT-builds CuteDSL kernels; bail early if the driver can't load their device image.
    try:
        HardwareInfo().get_max_active_clusters(1)
    except Exception as e:  # cutlass wraps the CUDA driver error in a bare RuntimeError
        raise ImportError(
            f"Image error: sonic-moe's CuteDSL kernels cannot load a device image on this GPU/driver "
            f"({type(e).__name__}: {e}). This usually means cutlass-dsl's bundled CUDA toolchain is "
            f"newer than the driver (e.g. cu13 libs on a CUDA 12.x driver)."
        )

    kernel = lazy_load_kernel("sonic-moe")
    if kernel is None:
        raise ImportError(
            "Failed to load the sonic-moe kernel — check that `kernels-community/sonic-moe` "
            "has a build matching the current torch/CUDA."
        )

    activation_type_enum = getattr(getattr(kernel, "enums", None), "ActivationType", None)
    moe_general_routing_inputs = getattr(kernel, "moe_general_routing_inputs", None)

    missing = [
        name
        for name, attr in [
            ("enums.ActivationType", activation_type_enum),
            ("moe_general_routing_inputs", moe_general_routing_inputs),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"sonic-moe kernel is missing required symbols: {', '.join(missing)}. "
            "Make sure you have the `kernels` package and `nvidia-cutlass-dsl` installed."
        )

    return SonicMoE(
        activation_type_enum=activation_type_enum,
        moe_general_routing_inputs=moe_general_routing_inputs,
    )




class SonicMoeHandle:

    # Map activation function names from HF config to SonicMoE epilogue names
    ACT_MAP = {"silu": "SWIGLU", "gelu": "GEGLU", "relu": "REGLU"}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets the state of the handle, for instance to retry loading the kernel after the first attempt failed."""
        self._loaded: bool = False
        self._loading_error: Exception | None = None
        self._cached_sonicmoe: SonicMoE | None = None

    def _load_sonicmoe_kernel(self) -> SonicMoE:
        """Loads and return a SonicMoE object which contains what's necessary to call the kernel. After the first call
        to this function, the output or the Exception is cached: call `.reset` to reset the cached state. Private
        method because the sonicmoe kernel is meant to be called from the handle.
        """
        # If this is the first time loading the kernel, it is not cached, so we need to actually load it
        if not self._loaded:
            try:
                self._cached_sonicmoe = _load_sonicmoe_kernel()
                self._loaded = True
            # Guard only against import errors: other errors are unexpected, so we raise them
            except ImportError as e:
                self._loading_error = e
                self._loaded = False
                raise
        # Otherwise, re-raise the loading error if it occurred the first time
        if self._loading_error is not None:
            raise ImportError(
                "Tried calling sonicmoe_experts_forward but the kernel failed to load on the first call. You can call "
                "`reset on the handle if you want to retry loading the kernel"
            ) from self._loading_error
        # Sanity check to see if the kernel was loaded successfully
        elif self._cached_sonicmoe is None:
            raise RuntimeError("sonicmoe kernel was marked has loaded but cannot be found. This should never happen.")
        return self._cached_sonicmoe

    @property
    def sonicmoe_is_available(self) -> bool:
        """A boolean indicating whether the sonicmoe kernel is available. Silences regular import errors that would
        indicate that the kernel is not available, but not other errors that are unexpected."""
        try:
            self._load_sonicmoe_kernel()
            return True
        except ImportError:
            pass
        return False

    def sonicmoe_experts_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Calls the underlying sonicmoe kernel with the given inputs. If the kernel has not been loaded yet, it will be
        loaded and cached."""
        # Check arguments before kernel loading: we might avoid loading altogether if the checks don't pass
        if not module.has_gate:
            raise ValueError("sonicmoe requires gated experts (has_gate=True)")
        if hidden_states.device.type != "cuda":
            raise ValueError("sonicmoe requires CUDA device")

        # Retrieve sonicmoe-compatible activation type
        activation_hf = getattr(module.config, "hidden_act", "silu").lower()
        activation_type = self.ACT_MAP.get(activation_hf, "SWIGLU")

        # Prepare auxilary inputs
        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)

        # Flatten — token_indices must be int32, sorted ascending (required by sonic-moe)
        token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        token_idx = token_idx.unsqueeze(1).expand(-1, num_top_k).reshape(-1)
        router_scores = top_k_weights.reshape(-1).to(hidden_states.dtype)
        expert_ids = top_k_index.reshape(-1).int()

        # EP sentinel handling: leave `expert_ids` unclamped — the kernel's metadata stage drops
        # `expert_ids >= num_experts` from the per-expert histogram and masks them out of the
        # scatter indices, so sentinels never enter the grouped GEMM. Their routing weights are
        # already zero (RouterParallel masks them at dispatch), so the per-token reduction
        # contributes nothing for sentinel slots.

        w1 = to_local(module.gate_up_proj)
        w2 = to_local(module.down_proj)
        b1 = to_local(module.gate_up_proj_bias) if module.has_bias else None
        b2 = to_local(module.down_proj_bias) if module.has_bias else None

        # Permute weights as expected by sonic-moe (E=num_experts, H=hidden_size, I=intermediate_size).
        # Non-transposed: gate_up_proj is (E, 2*I, H), down_proj is (E, H, I) -> permute(1, 2, 0).
        # Transposed: gate_up_proj is (E, H, 2*I), down_proj is (E, I, H) -> permute(2, 1, 0).
        perm = (2, 1, 0) if module.is_transposed else (1, 2, 0)
        w1 = w1.permute(*perm)  # (2*I, H, E)
        w2 = w2.permute(*perm)  # (I, H, E)

        return self._sonicmoe_wrapper(
            hidden_states=hidden_states,
            router_scores=router_scores,
            expert_ids=expert_ids,
            token_idx=token_idx,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            activation_type=activation_type,
            num_experts=module.num_experts,
            concat_layout=module.is_concatenated,
            is_inference_mode_enabled=not torch.is_grad_enabled(),
        )

    @torch._dynamo.allow_in_graph
    def _sonicmoe_wrapper(
        self,
        hidden_states: torch.Tensor,
        router_scores: torch.Tensor,
        expert_ids: torch.Tensor,
        token_idx: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        activation_type: str,
        num_experts: int,
        concat_layout: bool,
        is_inference_mode_enabled: bool,
    ) -> torch.Tensor:
        """Handle-level shim around `moe_general_routing_inputs` so `allow_in_graph` can wrap it.

        sonicmoe asserts `not torch.compiler.is_compiling()` internally because it dispatches
        CuteDSL kernels, which Dynamo can't trace. `allow_in_graph` keeps the call in the FX
        graph as a single opaque node (no tracing into the body, no graph break) while still
        running the real Python at runtime — autograd through `_UpProjection` / `_DownProjection`
        flows normally. The decorator must be applied at module load time, not inside the compiled
        function — hence this shim plus the `allow_in_graph` decorator above.
        """
        sonicmoe = self._load_sonicmoe_kernel()

        # Default to SwiGLU if the activation type is not found
        activation_type = getattr(
            sonicmoe.activation_type_enum,
            activation_type,
            sonicmoe.activation_type_enum.SWIGLU  # type: ignore
        )

        output, _ = sonicmoe.moe_general_routing_inputs(
            hidden_states,
            router_scores,
            token_idx,
            expert_ids,
            w1,
            b1,
            w2,
            b2,
            E=num_experts,
            activation_type=activation_type,
            is_inference_mode_enabled=is_inference_mode_enabled,
            concat_layout=concat_layout,
            stream_id=None,
        )
        return output


# Singleton object: this should be the only SonicMoeHandle object instantiated in the codebase
SONIC_MOE_HANDLE = SonicMoeHandle()
