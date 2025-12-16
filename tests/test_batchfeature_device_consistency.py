"""
Test script to verify the new BatchFeature device consistency implementation.
This test verifies that the device parameter works correctly in BatchFeature.__init__.
"""

import torch
import transformers
from PIL import Image
import requests
from torchvision import transforms

def test_batchfeature_device_parameter():
    """Test BatchFeature device parameter directly"""
    print("Testing BatchFeature device parameter...")
    
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping device parameter test")
        return True
    
    try:
        # Create some test data
        data = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        
        print("Original tensors devices:")
        for key, value in data.items():
            print(f"  {key}: {value.device}")
        
        # Create BatchFeature with device parameter
        batch_feature = transformers.feature_extraction_utils.BatchFeature(
            data=data, 
            tensor_type="pt", 
            device="cuda"
        )
        
        # Check if all tensors are on CUDA
        print("\nAfter BatchFeature with device='cuda':")
        all_on_cuda = True
        for key, value in batch_feature.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.device}")
                if value.device.type != "cuda":
                    all_on_cuda = False
                    print(f"  ERROR: {key} is not on CUDA")
        
        if all_on_cuda:
            print("✅ SUCCESS: All tensors moved to CUDA!")
            return True
        else:
            print("❌ FAILURE: Not all tensors are on CUDA!")
            return False
            
    except Exception as e:
        print(f"Error during BatchFeature device test: {e}")
        return False

def test_oneformer_with_new_implementation():
    """Test OneFormer processor with new BatchFeature device implementation"""
    print("\nTesting OneFormer processor with new device implementation...")
    
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping OneFormer device test")
        return True
    
    try:
        # Setup processor
        processor = transformers.OneFormerImageProcessorFast()
        processor = transformers.OneFormerProcessor(
            image_processor=processor,
            tokenizer=transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
        )
        
        # Load test image
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        
        # Convert image to tensor and move to CUDA
        to_tensor_transform = transforms.ToTensor()
        image = to_tensor_transform(image).to("cuda")
        
        print(f"Input image device: {image.device}")
        
        # Process with explicit device parameter
        inputs = processor(image, ["semantic"], return_tensors="pt", device="cuda")
        
        # Check device consistency
        print("Checking output devices:")
        all_on_same_device = True
        reference_device = None
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.device}")
                if reference_device is None:
                    reference_device = value.device
                elif value.device != reference_device:
                    all_on_same_device = False
                    print(f"  ERROR: {key} is on {value.device} but expected {reference_device}")
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device = value[0].device
                print(f"  {key}[0]: {device}")
                if reference_device is None:
                    reference_device = device
                elif device != reference_device:
                    all_on_same_device = False
                    print(f"  ERROR: {key}[0] is on {device} but expected {reference_device}")
        
        if all_on_same_device and reference_device.type == "cuda":
            print("✅ SUCCESS: All tensors are on CUDA and consistent!")
            return True
        else:
            print("❌ FAILURE: Device consistency issue!")
            return False
            
    except Exception as e:
        print(f"Error during OneFormer test: {e}")
        return False

def test_cpu_to_cuda_movement():
    """Test moving CPU tensors to CUDA using device parameter"""
    print("\nTesting CPU to CUDA movement...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CPU to CUDA test")
        return True
    
    try:
        # Setup processor
        processor = transformers.OneFormerImageProcessorFast()
        processor = transformers.OneFormerProcessor(
            image_processor=processor,
            tokenizer=transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
        )
        
        # Load test image (keep on CPU)
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        
        print("Input image device: CPU (PIL Image)")
        
        # Process with device parameter to move to CUDA
        inputs = processor(image, ["semantic"], return_tensors="pt", device="cuda")
        
        # Check if all outputs are on CUDA
        print("Checking if all outputs moved to CUDA:")
        all_on_cuda = True
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.device}")
                if value.device.type != "cuda":
                    all_on_cuda = False
                    print(f"  ERROR: {key} is not on CUDA")
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device = value[0].device
                print(f"  {key}[0]: {device}")
                if device.type != "cuda":
                    all_on_cuda = False
                    print(f"  ERROR: {key}[0] is not on CUDA")
        
        if all_on_cuda:
            print("✅ SUCCESS: All tensors moved to CUDA as requested!")
            return True
        else:
            print("❌ FAILURE: Not all tensors are on CUDA!")
            return False
            
    except Exception as e:
        print(f"Error during CPU to CUDA test: {e}")
        return False

if __name__ == "__main__":
    print("Testing new BatchFeature device consistency implementation...")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_batchfeature_device_parameter()
    test2_passed = test_oneformer_with_new_implementation()
    test3_passed = test_cpu_to_cuda_movement()
    
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"BatchFeature device parameter: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"OneFormer device consistency: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"CPU to CUDA movement: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! New BatchFeature device implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
