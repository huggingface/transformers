# src/transformers/utils/cpu_heuristics.py
import os
import logging
import warnings
import torch
from typing import Optional

logger = logging.getLogger(__name__)
_DTYPE_POLICY_ENV = "HF_CPU_DTYPE_POLICY"
_THREADS_OPTIMIZED_ENV = "HF_CPU_THREADS_OPTIMIZED"

def _get_policy() -> str:
    return os.environ.get(_DTYPE_POLICY_ENV, "warn_and_fallback").lower()

def apply_cpu_safety_settings(model):
    import os, torch, warnings
    from transformers.utils import logging

    logger = logging.get_logger(__name__)

    policy = os.environ.get("HF_CPU_DTYPE_POLICY", "warn_and_fallback").lower()

    # Detect dtype of first parameter
    first_param = next(model.parameters(), None)
    if first_param is None:
        return model

    model_dtype = first_param.dtype

    # Detect unsafe dtypes for CPU
    if model_dtype in (torch.float16, torch.bfloat16):
        msg = (
            f"Model loaded with {model_dtype} on CPU — this may cause slowdowns or errors. "
            "You can set HF_CPU_DTYPE_POLICY=warn|auto|error to control behavior."
        )

        if policy == "error":
            raise RuntimeError(msg)

        elif policy in ("warn", "warn_and_fallback", "auto"):
                warnings.warn(
                    f"Model loaded with {model_dtype} (fp16/bf16) on CPU — converting to float32 for safety.",
                    UserWarning,
                )
                model = model.to(dtype=torch.float32)
                logger.info("Converted model to float32 for CPU safety.")
        else:
            # Unknown policy, just warn but don’t crash
            warnings.warn(f"Unknown HF_CPU_DTYPE_POLICY='{policy}', keeping model as-is.")

    return model
