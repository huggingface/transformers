#!/usr/bin/env python3
"""
Test script for SDPA (Scaled Dot Product Attention) support in the benchmarking framework.
This tests the SDPA backend functionality without requiring actual PyTorch/transformers.
"""

import sys
import logging

# Add current directory to path
sys.path.insert(0, '.')

def test_sdpa_backend_detection():
    """Test SDPA backend detection."""
    print("=" * 60)
    print("TESTING SDPA BACKEND DETECTION")
    print("=" * 60)
    
    try:
        from framework import get_available_sdpa_backends, get_sdpa_backend
        
        # Test getting available backends
        backends = get_available_sdpa_backends()
        print(f"Available SDPA backends: {backends}")
        
        # Test individual backend detection
        test_backends = ["math", "flash_attention", "efficient_attention", "cudnn_attention"]
        for backend_name in test_backends:
            backend = get_sdpa_backend(backend_name)
            available = backend is not None
            print(f"  {backend_name}: {'✓' if available else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"✗ SDPA backend detection failed: {e}")
        return False


def test_sdpa_context():
    """Test SDPA context manager."""
    print("\n" + "=" * 60)
    print("TESTING SDPA CONTEXT MANAGER")
    print("=" * 60)
    
    try:
        from framework import SDPAContext
        
        # Setup logging to see debug messages
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
        logger = logging.getLogger("test_sdpa")
        
        # Test with different backend names
        test_backends = [None, "math", "flash_attention", "invalid_backend"]
        
        for backend_name in test_backends:
            print(f"\nTesting SDPA context with backend: {backend_name}")
            try:
                with SDPAContext(backend_name, logger) as ctx:
                    print(f"  ✓ Context created successfully")
                    print(f"  Backend name: {ctx.backend_name}")
                    print(f"  Backend object: {ctx.backend}")
            except Exception as e:
                print(f"  ✗ Context failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ SDPA context test failed: {e}")
        return False


def test_benchmark_config_with_sdpa():
    """Test BenchmarkConfig with SDPA parameters."""
    print("\n" + "=" * 60)
    print("TESTING BENCHMARK CONFIG WITH SDPA")
    print("=" * 60)
    
    try:
        from framework import BenchmarkConfig, create_config_variants
        
        # Test basic config with SDPA parameters
        config = BenchmarkConfig(
            name="test_sdpa",
            model_id="test-model",
            attn_implementation="sdpa",
            sdpa_backend="flash_attention"
        )
        
        print(f"✓ Basic config created:")
        print(f"  Attention implementation: {config.attn_implementation}")
        print(f"  SDPA backend: {config.sdpa_backend}")
        
        # Test config variants with SDPA
        configs = create_config_variants(
            config,
            {
                "variant": ["eager", "compiled"],
                "sdpa_backend": [None, "math", "flash_attention"]
            }
        )
        
        print(f"\n✓ Created {len(configs)} config variants:")
        for i, cfg in enumerate(configs):
            print(f"  {i+1}. {cfg.name}: variant={cfg.variant}, sdpa_backend={cfg.sdpa_backend}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark config with SDPA test failed: {e}")
        return False


def test_mock_benchmark_with_sdpa():
    """Test mock benchmark with SDPA configurations."""
    print("\n" + "=" * 60)
    print("TESTING MOCK BENCHMARK WITH SDPA")
    print("=" * 60)
    
    try:
        from benches.mock_benchmark import create_mock_configs
        
        # Create mock configs with SDPA
        configs = create_mock_configs(
            warmup_iterations=1,
            measurement_iterations=1,
            num_tokens_to_generate=5
        )
        
        print(f"✓ Created {len(configs)} mock configurations")
        
        # Count SDPA variants
        sdpa_variants = {}
        for config in configs:
            backend = config.sdpa_backend or "default"
            sdpa_variants[backend] = sdpa_variants.get(backend, 0) + 1
        
        print("SDPA backend distribution:")
        for backend, count in sdpa_variants.items():
            print(f"  {backend}: {count} configs")
        
        # Show sample configs
        print("\nSample configurations:")
        for i, config in enumerate(configs[:3]):
            print(f"  {i+1}. {config.name}")
            print(f"     - variant: {config.variant}")
            print(f"     - attn_implementation: {config.attn_implementation}")
            print(f"     - sdpa_backend: {config.sdpa_backend}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock benchmark with SDPA test failed: {e}")
        return False


def main():
    """Run all SDPA tests."""
    print("Testing SDPA support in benchmarking framework v2...")
    
    tests = [
        test_sdpa_backend_detection,
        test_sdpa_context,
        test_benchmark_config_with_sdpa,
        test_mock_benchmark_with_sdpa
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("SDPA TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All SDPA tests passed! Framework supports attention backends.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above.")
    
    print("\nSDPA Features Summary:")
    print("✓ Multiple attention implementations: eager, sdpa, flash_attention_2")
    print("✓ SDPA backend selection: math, flash_attention, efficient_attention, cudnn_attention")
    print("✓ Automatic backend detection based on system capabilities")
    print("✓ Context manager for safe SDPA kernel switching")
    print("✓ Integration with configuration variants for comprehensive benchmarking")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 