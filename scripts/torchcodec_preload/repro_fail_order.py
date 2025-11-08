"""
Minimal repro: import torchcodec shared libs first, then transformers.
This may raise an error like: "mpeg version 8: Could not load this library..."
"""
from __future__ import annotations

from torchcodec._core.ops import load_torchcodec_shared_libraries


def main() -> None:
    load_torchcodec_shared_libraries()

    # Import transformers modules that use audio processing
    from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

    print("Loaded torchcodec then transformers successfully")


if __name__ == "__main__":
    main()
