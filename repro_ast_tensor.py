
import torch
from transformers import ASTFeatureExtractor

def test_ast_tensor_input():
    print("Testing ASTFeatureExtractor ...")
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    sampling_rate = 16000
    
    # 1. Test 1D Tensor
    print("\n--- Test 1D Tensor ---")
    audio_1d = torch.randn(16000)
    try:
        inputs = feature_extractor(audio_1d, sampling_rate=sampling_rate, return_tensors="pt")
        print(f"Success! Output shape: {inputs['input_values'].shape}")
    except Exception as e:
        print(f"Failed 1D: {e}")

    # 2. Test 2D Tensor (Batch)
    print("\n--- Test 2D Tensor (Batch) ---")
    audio_2d = torch.randn(2, 16000)
    try:
        inputs = feature_extractor(audio_2d, sampling_rate=sampling_rate, return_tensors="pt")
        print(f"Success! Output shape: {inputs['input_values'].shape}")
    except Exception as e:
        print(f"Failed 2D: {e}")

    # 3. Test Lists (Legacy)
    print("\n--- Test NumPy List (Legacy) ---")
    import numpy as np
    audio_list = [np.random.randn(16000).astype(np.float32) for _ in range(2)]
    try:
        inputs = feature_extractor(audio_list, sampling_rate=sampling_rate, return_tensors="pt")
        print(f"Success! Output shape: {inputs['input_values'].shape}")
    except Exception as e:
        print(f"Failed List: {e}")

if __name__ == "__main__":
    test_ast_tensor_input()
