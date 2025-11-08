"""
Preload libavcodec via ctypes before loading torchcodec, then import transformers.

This is a recommended workaround when third-party code calls
`load_torchcodec_shared_libraries()` early and an incompatible libav* is already loaded.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys


def preload_avcodec() -> None:
    # Locate libavcodec
    path = ctypes.util.find_library("avcodec")
    if path is None:
        # Give a helpful hint and exit non-zero
        print("Could not find libavcodec via ctypes.util.find_library('avcodec').\n"
              "Run `ldconfig -p | grep avcodec` to locate a system lib, or set LD_PRELOAD.")
        sys.exit(2)

    print("Preloading:", path)
    # Load globally so subsequent C-extensions see its symbols
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    preload_avcodec()

    from torchcodec._core.ops import load_torchcodec_shared_libraries  # noqa: E402

    load_torchcodec_shared_libraries()

    # Now safe to import transformers
    from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

    print("Preloaded avcodec, loaded torchcodec and transformers successfully")


if __name__ == "__main__":
    main()
