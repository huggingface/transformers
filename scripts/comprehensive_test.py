#!/usr/bin/env python3
"""Comprehensive end-to-end test of all functionality."""

import sys
import os
import subprocess
import torch

# Add the repo root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")

    try:
        from assist_strict.overlay import build_overlay_config, AssistantModelProxy
        from assist_strict.assisted import assisted_generate_strict
        from transformers.generation.utils import MetaSafeTensorError
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_meta_tensor_safety():
    """Test meta tensor safety functionality."""
    print("🔍 Testing meta tensor safety...")

    try:
        from transformers import GenerationConfig, AutoModelForCausalLM
        from transformers.generation.utils import MetaSafeTensorError

        # Load a small model
        model = AutoModelForCausalLM.from_pretrained('gpt2', device_map=None)
        model = model.cpu()

        # Create a generation config with meta tensors
        generation_config = GenerationConfig()
        meta_device = torch.device('meta')
        generation_config.eos_token_id = torch.tensor(50256, device=meta_device, dtype=torch.long)

        # Test that meta tensors trigger our error
        try:
            model._prepare_special_tokens(
                generation_config=generation_config,
                kwargs_has_attention_mask=True,
                device=torch.device('cpu')
            )
            print("❌ Should have raised MetaSafeTensorError")
            return False
        except MetaSafeTensorError:
            print("✅ Meta tensor safety working correctly")
            return True
    except Exception as e:
        print(f"❌ Meta tensor test failed: {e}")
        return False

def test_pytest_suite():
    """Run the pytest test suite."""
    print("🔍 Running pytest suite...")

    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/test_generation_meta.py', '-q'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ All pytest tests passed")
            return True
        else:
            print(f"❌ Pytest failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Pytest execution failed: {e}")
        return False

def test_validation_scripts():
    """Test validation scripts."""
    print("🔍 Testing validation scripts...")

    scripts_to_test = [
        'scripts/validate_strict_overlay.py',
        'scripts/concurrency_probe.py'
    ]

    all_passed = True
    for script in scripts_to_test:
        try:
            result = subprocess.run([sys.executable, script],
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {script} passed")
            else:
                print(f"❌ {script} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                all_passed = False
        except Exception as e:
            print(f"❌ {script} execution failed: {e}")
            all_passed = False

    return all_passed

def main():
    """Run all tests and report results."""
    print("🚀 Running comprehensive end-to-end test suite...")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Meta Tensor Safety", test_meta_tensor_safety),
        ("Pytest Suite", test_pytest_suite),
        ("Validation Scripts", test_validation_scripts),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        result = test_func()
        results.append((test_name, result))
        print()

    # Summary
    print("=" * 60)
    print("🎯 FINAL RESULTS:")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The system is ready for production.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
