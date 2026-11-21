"""Metal shared events in PyTorch's MPS stream (GPU-side waits the host signals) and CPU pointers into MPS tensors,
built on first use (`torch.utils.cpp_extension`, Objective-C++)."""

import functools
import os


@functools.cache
def load():
    """The compiled extension, or None where it cannot build (not macOS, no compiler)."""
    try:
        import torch
        from torch.utils.cpp_extension import load as load_extension

        if not torch.backends.mps.is_available():
            return None
        return load_extension(
            "transformers_mps_events",
            [os.path.join(os.path.dirname(__file__), "mps_events.mm")],
            extra_cflags=["-std=c++20", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
    except Exception:
        return None
