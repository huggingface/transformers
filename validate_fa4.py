#!/usr/bin/env python3
"""
Quick FA4 validation script for SSH GPU access.

Usage:
    python validate_fa4.py
"""

import sys

import torch


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def check_cuda():
    print_section("CUDA Environment")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    major, minor = torch.cuda.get_device_capability()
    compute_cap = major * 10 + minor
    print(f"Compute capability: SM {major}.{minor} ({compute_cap})")

    if compute_cap < 80:
        print(f"WARNING: FA4 requires SM 8.0+, you have SM {major}.{minor}")
        return False

    if compute_cap < 90:
        print(f"NOTE: FA4 is optimized for SM 9.0+ (Hopper/Blackwell)")
        print(f"      Your GPU (SM {major}.{minor}) will have limited optimizations")

    return True


def check_transformers():
    print_section("Transformers Installation")
    try:
        import transformers

        print(f"Transformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import transformers: {e}")
        return False


def check_flash_attn_package():
    print_section("Flash Attention Package")
    try:
        import flash_attn

        print(f"flash-attn installed: Yes")
        if hasattr(flash_attn, "__version__"):
            print(f"flash-attn version: {flash_attn.__version__}")
        return True
    except ImportError:
        print("ERROR: flash-attn package not installed")
        print("Install with: pip install flash-attn")
        return False


def check_fa4_availability():
    print_section("FA4 Detection")
    try:
        from transformers import is_flash_attn_4_available

        fa4_available = is_flash_attn_4_available()
        print(f"is_flash_attn_4_available(): {fa4_available}")

        if not fa4_available:
            print("\nFA4 not available. Checking why...")

            # Try manual import
            try:
                from flash_attn.cute import flash_attn_func

                print("  - flash_attn.cute can be imported manually")
                print("  - Detection function may have incorrect logic")
            except ImportError as e:
                print(f"  - flash_attn.cute import failed: {e}")
                print("  - FA4 (CuTe DSL) not available in installed flash-attn")

        return fa4_available
    except Exception as e:
        print(f"ERROR: Detection check failed: {e}")
        return False


def test_fa4_import():
    print_section("FA4 Import Test")
    try:
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        print("Successfully imported:")
        print(f"  - flash_attn_func: {flash_attn_func}")
        print(f"  - flash_attn_varlen_func: {flash_attn_varlen_func}")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import FA4: {e}")
        return False


def test_fa4_signature():
    print_section("FA4 API Signature Check")
    try:
        import inspect

        from flash_attn.cute import flash_attn_varlen_func

        sig = inspect.signature(flash_attn_varlen_func)
        params = list(sig.parameters.keys())

        print(f"flash_attn_varlen_func parameters: {params}")

        has_max_seqlen = "max_seqlen_q" in params
        has_cu_seqlens = "cu_seqlens_q" in params

        print(f"\nAPI check:")
        print(f"  - Has cu_seqlens_q: {has_cu_seqlens} (expected: True)")
        print(f"  - Has max_seqlen_q: {has_max_seqlen} (expected: False)")

        if has_cu_seqlens and not has_max_seqlen:
            print("\nSUCCESS: FA4 API signature is correct")
            return True
        else:
            print("\nWARNING: Unexpected API signature")
            return False

    except Exception as e:
        print(f"ERROR: Signature check failed: {e}")
        return False


def test_fa4_basic_forward():
    print_section("FA4 Basic Forward Test")
    try:
        from flash_attn.cute import flash_attn_func

        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        out = flash_attn_func(q, k, v, causal=False)

        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")

        if out.shape == (batch_size, seq_len, num_heads, head_dim):
            print("\nSUCCESS: FA4 forward pass works correctly")
            return True
        else:
            print(f"\nERROR: Unexpected output shape")
            return False

    except Exception as e:
        print(f"ERROR: Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hf_integration():
    print_section("HF Integration Test")
    try:
        from transformers.modeling_flash_attention_utils import _is_using_fa4, lazy_import_flash_attention

        print("Testing explicit FA4 selection...")
        (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_kwargs_fn = lazy_import_flash_attention(
            "flash_attention_4"
        )

        is_fa4 = _is_using_fa4(flash_varlen_fn)
        print(f"  - _is_using_fa4(): {is_fa4} (expected: True)")

        print("\nTesting auto-selection...")
        (flash_fn_auto, flash_varlen_fn_auto, _, _), _ = lazy_import_flash_attention(None)

        is_fa4_auto = _is_using_fa4(flash_varlen_fn_auto)
        print(f"  - Auto-selected FA4: {is_fa4_auto}")

        if is_fa4:
            print("\nSUCCESS: HF integration works correctly")
            return True
        else:
            print("\nWARNING: HF integration issue detected")
            return False

    except Exception as e:
        print(f"ERROR: HF integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("  Flash Attention 4 (FA4) Validation")
    print("=" * 70)

    results = {
        "CUDA": check_cuda(),
        "Transformers": check_transformers(),
        "flash-attn package": check_flash_attn_package(),
        "FA4 detection": check_fa4_availability(),
        "FA4 import": test_fa4_import(),
        "FA4 API signature": test_fa4_signature(),
        "FA4 forward pass": test_fa4_basic_forward(),
        "HF integration": test_hf_integration(),
    }

    print_section("Summary")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL CHECKS PASSED - FA4 IS READY TO USE")
    else:
        print("  SOME CHECKS FAILED - SEE ABOVE FOR DETAILS")
    print("=" * 70 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
