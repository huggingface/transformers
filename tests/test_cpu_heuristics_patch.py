# tests/test_utils_cpu_heuristics.py
import os
import warnings

import torch

from transformers.utils import cpu_heuristics


def test_apply_cpu_safety_settings_fallback(tmp_path, monkeypatch):
    # Build a tiny dummy model with float16 params on CPU
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            # convert param dtype to float16 on CPU
            for p in self.parameters():
                p.data = p.data.to(torch.float16)

    model = TinyModel()
    # Precondition: first param is float16 and on cpu
    first_param = next(model.parameters())
    assert first_param.device.type == "cpu"
    assert first_param.dtype == torch.float16

    # Ensure default policy (warn_and_fallback)
    if "HF_CPU_DTYPE_POLICY" in os.environ:
        del os.environ["HF_CPU_DTYPE_POLICY"]
    if "HF_CPU_THREADS_OPTIMIZED" in os.environ:
        del os.environ["HF_CPU_THREADS_OPTIMIZED"]

    # Capture warnings and run heuristic
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model2 = cpu_heuristics.apply_cpu_safety_settings(model)
        # After fallback model params should be float32
        dtype_after = next(model2.parameters()).dtype
        assert dtype_after == torch.float32, "Expected fallback to float32 on CPU"

        # Ensure a warning was emitted
        assert any("fp16" in str(x.message).lower() or "bf16" in str(x.message).lower() for x in w)


def test_policy_error(monkeypatch):
    import os

    os.environ["HF_CPU_DTYPE_POLICY"] = "error"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            for p in self.parameters():
                p.data = p.data.to(torch.bfloat16)

    model = TinyModel()
    try:
        # When HF_CPU_DTYPE_POLICY=error, a RuntimeError should be raised
        raised = False
        try:
            cpu_heuristics.apply_cpu_safety_settings(model)
        except RuntimeError:
            raised = True
        assert raised, "Expected RuntimeError for HF_CPU_DTYPE_POLICY=error"
    finally:
        del os.environ["HF_CPU_DTYPE_POLICY"]
