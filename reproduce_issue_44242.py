#!/usr/bin/env python3
"""
Reproduce issue #44242: Load balancing loss not added when output_router_logits=False

This script demonstrates the bug where auxiliary load balancing loss is not computed
when output_router_logits=False, even when router_aux_loss_coef != 0.
"""

from transformers import MixtralConfig, MixtralForCausalLM
import torch


def test_aux_loss_with_output_router_logits(output_router_logits: bool):
    """Test auxiliary loss computation with different output_router_logits settings."""
    print(f"\n=== Testing with output_router_logits={output_router_logits} ===")
    
    # Configure the model
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=2,  # small for demonstration
        num_local_experts=8,
        output_router_logits=output_router_logits,  
        router_aux_loss_coef=0.001   # Non-zero coefficient should enable aux loss
    )

    # Initialize the model
    model = MixtralForCausalLM(config)

    # Create dummy inputs and labels for a training step
    input_ids = torch.tensor([[1, 254, 99, 32]])
    labels = torch.tensor([[1, 254, 99, 32]])  # Next-token prediction labels

    # Perform the forward pass
    outputs = model(input_ids=input_ids, labels=labels)

    # Check the losses
    total_loss = outputs.loss
    aux_loss = outputs.aux_loss 
    router_logits = outputs.router_logits

    print(f"Config router_aux_loss_coef: {config.router_aux_loss_coef}")
    print(f"Auxiliary Load Balancing Loss: {aux_loss}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Router logits available: {router_logits is not None}")
    
    if aux_loss is not None:
        print(f"Auxiliary Loss value: {aux_loss.item():.4f}")
        expected_total = total_loss.item() if config.router_aux_loss_coef == 0 else "loss + coef * aux_loss"
        print(f"Expected relationship: {expected_total}")
    else:
        print("BUG: Auxiliary loss is None even though router_aux_loss_coef != 0")
    
    return aux_loss is not None


if __name__ == "__main__":
    print("Reproducing issue #44242...")
    print("Expected: aux_loss should be computed when router_aux_loss_coef != 0, regardless of output_router_logits")
    
    # Test with output_router_logits=True (should work)
    works_with_true = test_aux_loss_with_output_router_logits(True)
    
    # Test with output_router_logits=False (currently broken)
    works_with_false = test_aux_loss_with_output_router_logits(False)
    
    print(f"\n=== Results ===")
    print(f"With output_router_logits=True: {'✓ WORKS' if works_with_true else '✗ BROKEN'}")
    print(f"With output_router_logits=False: {'✓ WORKS' if works_with_false else '✗ BROKEN (this is the bug)'}")
    
    if not works_with_false:
        print("\nBUG CONFIRMED: aux_loss is not computed when output_router_logits=False")
        print("This violates the documented behavior that aux_loss should be computed when router_aux_loss_coef != 0")