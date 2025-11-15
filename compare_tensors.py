import sys
import torch

def compare_tensors(filepath1, filepath2):
    """Compare two tensors loaded from files."""
    tensor1 = torch.load(filepath1)
    tensor2 = torch.load(filepath2)
    
    print(f"Tensor 1 shape: {tensor1.shape}")
    print(f"Tensor 2 shape: {tensor2.shape}")
    
    if tensor1.shape != tensor2.shape:
        print("❌ Shapes differ!")
        return
    
    if torch.allclose(tensor1, tensor2, atol=1e-3):
        print("✓ Tensors are equal (within tolerance)")
    else:
        max_diff = (tensor1 - tensor2).abs().max().item()
        mean_diff = (tensor1 - tensor2).abs().mean().item()
        print(f"❌ Tensors differ")
        print(f"   Max difference: {max_diff}")
        print(f"   Mean difference: {mean_diff}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_tensors.py <filepath1> <filepath2>")
        sys.exit(1)
    
    compare_tensors(sys.argv[1], sys.argv[2])