#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_all_fixes():
    print("🧪 Testing all DETR max_size parameter fixes...")
    
    try:
        from transformers.models.conditional_detr.image_processing_conditional_detr import ConditionalDetrImageProcessor
        from transformers.models.detr.image_processing_detr import DetrImageProcessor  
        from transformers.models.deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor
        
        processors = [
            ("ConditionalDetr", ConditionalDetrImageProcessor),
            ("Detr", DetrImageProcessor),
            ("DeformableDetr", DeformableDetrImageProcessor)
        ]
        
        for name, ProcessorClass in processors:
            print(f"\n🔧 Testing {name}ImageProcessor...")
            
            # Test 1: from_dict with size=42, max_size=84
            processor = ProcessorClass.from_dict({
                "do_resize": True,
                "do_normalize": True,
                "do_pad": True,
            }, size=42, max_size=84)
            expected = {"shortest_edge": 42, "longest_edge": 84}
            actual = processor.size
            assert actual == expected, f"❌ Test 1 failed: expected {expected}, got {actual}"
            print(f"✅ Test 1 passed: from_dict(size=42, max_size=84) = {actual}")
            
            # Test 2: from_dict with size dict without longest_edge + max_size
            processor = ProcessorClass.from_dict({
                "do_resize": True,
                "do_normalize": True,
                "do_pad": True,
                "size": {"shortest_edge": 100}
            }, max_size=200)
            expected = {"shortest_edge": 100, "longest_edge": 200}
            actual = processor.size
            assert actual == expected, f"❌ Test 2 failed: expected {expected}, got {actual}"
            print(f"✅ Test 2 passed: size without longest_edge + max_size = {actual}")
            
            # Test 3: init with max_size only
            processor = ProcessorClass(max_size=500)
            expected = {"shortest_edge": 800, "longest_edge": 500}
            actual = processor.size
            assert actual == expected, f"❌ Test 3 failed: expected {expected}, got {actual}"
            print(f"✅ Test 3 passed: init(max_size=500) = {actual}")
            
            print(f"🎉 All tests passed for {name}ImageProcessor!")
            
        print("\n🌟 All DETR image processors work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    sys.exit(0 if success else 1) 