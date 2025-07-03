#!/usr/bin/env python3
"""
Simple test to verify that mobilenet_v5 implementation fixes the "Unknown Model" error
for Gemma 3n with architecture="mobilenetv5_300m_enc".
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_mobilenet_v5_registration():
    """Test that mobilenet_v5 is properly registered in auto classes."""
    print("Testing mobilenet_v5 registration...")
    
    try:
        from transformers import AutoConfig, AutoModel, AutoImageProcessor
        
        # Test AutoConfig
        config = AutoConfig.for_model('mobilenet_v5')
        print("✓ AutoConfig.for_model('mobilenet_v5') works")
        
        # Test AutoModel
        model = AutoModel.from_config(config)
        print("✓ AutoModel.from_config works for mobilenet_v5")
        
        # Test AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained('mobilenet_v5')
        print("✓ AutoImageProcessor.from_pretrained works for mobilenet_v5")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_gemma3n_with_mobilenetv5():
    """Test that Gemma3nVisionConfig works with mobilenetv5_300m_enc architecture."""
    print("\nTesting Gemma3n with mobilenetv5_300m_enc...")
    
    try:
        from transformers import Gemma3nVisionConfig
        
        # This should not raise "Unknown Model" error anymore
        config = Gemma3nVisionConfig(architecture='mobilenetv5_300m_enc')
        print("✓ Gemma3nVisionConfig with mobilenetv5_300m_enc works")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_direct_imports():
    """Test direct imports of mobilenet_v5 components."""
    print("\nTesting direct imports...")
    
    try:
        from transformers.models.mobilenet_v5 import (
            MobileNetV5Config,
            MobileNetV5Model,
            MobileNetV5ImageProcessor
        )
        print("✓ Direct imports work")
        
        # Test instantiation
        config = MobileNetV5Config()
        model = MobileNetV5Model(config)
        processor = MobileNetV5ImageProcessor()
        print("✓ Direct instantiation works")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing mobilenet_v5 implementation...\n")
    
    tests = [
        test_mobilenet_v5_registration,
        test_gemma3n_with_mobilenetv5,
        test_direct_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The mobilenet_v5 implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 