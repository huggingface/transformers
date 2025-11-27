"""python
# Defensive TF32 toggling helper.
# Usage: from transformers.utils.tf32_utils import set_tf32_mode; set_tf32_mode(True|False)

def set_tf32_mode(enable: bool) -> None:
    """
    Safely toggle TF32 on the current environment.

    - No-ops on CPU-only runners.
    - Tries the new PyTorch >=2.9 API (torch.backends.cuda.matmul.fp32_precision / torch.backends.cudnn.conv.fp32_precision)
    - Falls back to the old API (torch.backends.cuda.matmul.allow_tf32 / torch.backends.cudnn.allow_tf32)
    - Handles MUSA/mudnn allow_tf32 if present.
    - Swallows exceptions to avoid failing tests on exotic environments.
    """
    try:
        import torch

        # If CUDA isn't available, bail out early. This avoids AttributeError on CPU CI.
        if not getattr(torch.cuda, "is_available", lambda: False)():
            return

        # MUSA uses mudnn.allow_tf32 in some environments; try setting if present.
        try:
            if hasattr(torch.backends, "mudnn"):
                try:
                    torch.backends.mudnn.allow_tf32 = bool(enable)
                except Exception:
                    # Some builds may not expose this; ignore failures.
                    pass
        except Exception:
            # defensive outer catch for weird torch builds
            pass

        # Safely access cuda.matmul and cudnn.conv where available.
        cuda_backend = getattr(torch.backends, "cuda", None)
        matmul = getattr(cuda_backend, "matmul", None)
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        cudnn_conv = getattr(cudnn_backend, "conv", None)

        # New API (PyTorch >= 2.9): fp32_precision = "tf32" / "ieee"
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            try:
                matmul.fp32_precision = "tf32" if enable else "ieee"
            except Exception:
                pass
        elif matmul is not None and hasattr(matmul, "allow_tf32"):
            try:
                matmul.allow_tf32 = bool(enable)
            except Exception:
                pass

        # cudnn.conv may have fp32_precision (new API) or allow_tf32 (old API).
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = "tf32" if enable else "ieee"
            except Exception:
                pass
        elif hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            try:
                torch.backends.cudnn.allow_tf32 = bool(enable)
            except Exception:
                pass

    except Exception:
        # Never raise here: toggling TF32 should never break tests or examples.
        return
"""
