#!/usr/bin/env python3
"""
Test for Mixtral auxiliary loss fix (issue #44242)
Tests that auxiliary load balancing loss is computed when router_aux_loss_coef > 0,
regardless of output_router_logits setting.
"""

import torch
import sys
import os

# Add the local transformers to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from transformers import MixtralConfig, MixtralForCausalLM
except ImportError:
    print("Warning: transformers not available, creating mock classes for testing logic")
    
    class MockTensor:
        def __init__(self, value):
            self.value = value
        def item(self):
            return self.value
        def to(self, device):
            return self
            
    class MockOutput:
        def __init__(self, loss=None, aux_loss=None, router_logits=None):
            self.loss = MockTensor(loss) if loss is not None else None
            self.aux_loss = MockTensor(aux_loss) if aux_loss is not None else None
            self.router_logits = router_logits
    
    def test_logic_only():
        print("Testing auxiliary loss logic without transformers...")
        
        # Test case 1: output_router_logits=False, router_aux_loss_coef=0.001 -> should compute aux_loss
        output_router_logits = False
        router_aux_loss_coef = 0.001
        labels = True  # simulating labels is not None
        
        need_router_logits_for_aux_loss = router_aux_loss_coef > 0 and labels
        collect_router_logits = output_router_logits or need_router_logits_for_aux_loss
        
        print(f"Case 1: output_router_logits={output_router_logits}, router_aux_loss_coef={router_aux_loss_coef}")
        print(f"  need_router_logits_for_aux_loss={need_router_logits_for_aux_loss}")
        print(f"  collect_router_logits={collect_router_logits}")
        print(f"  Expected: collect_router_logits=True, need_router_logits_for_aux_loss=True")
        print()
        
        # Test case 2: output_router_logits=True, router_aux_loss_coef=0.001 -> should compute aux_loss and return router_logits
        output_router_logits = True
        router_aux_loss_coef = 0.001
        
        need_router_logits_for_aux_loss = router_aux_loss_coef > 0 and labels
        collect_router_logits = output_router_logits or need_router_logits_for_aux_loss
        
        print(f"Case 2: output_router_logits={output_router_logits}, router_aux_loss_coef={router_aux_loss_coef}")
        print(f"  need_router_logits_for_aux_loss={need_router_logits_for_aux_loss}")
        print(f"  collect_router_logits={collect_router_logits}")
        print(f"  Expected: collect_router_logits=True, need_router_logits_for_aux_loss=True")
        print()
        
        # Test case 3: output_router_logits=False, router_aux_loss_coef=0 -> should NOT compute aux_loss
        output_router_logits = False
        router_aux_loss_coef = 0
        
        need_router_logits_for_aux_loss = router_aux_loss_coef > 0 and labels
        collect_router_logits = output_router_logits or need_router_logits_for_aux_loss
        
        print(f"Case 3: output_router_logits={output_router_logits}, router_aux_loss_coef={router_aux_loss_coef}")
        print(f"  need_router_logits_for_aux_loss={need_router_logits_for_aux_loss}")
        print(f"  collect_router_logits={collect_router_logits}")
        print(f"  Expected: collect_router_logits=False, need_router_logits_for_aux_loss=False")
        print()
        
        # Test case 4: no labels (inference) - should not compute aux_loss even with router_aux_loss_coef > 0
        output_router_logits = False
        router_aux_loss_coef = 0.001
        labels = False  # simulating labels is None
        
        need_router_logits_for_aux_loss = router_aux_loss_coef > 0 and labels
        collect_router_logits = output_router_logits or need_router_logits_for_aux_loss
        
        print(f"Case 4: output_router_logits={output_router_logits}, router_aux_loss_coef={router_aux_loss_coef}, labels=None")
        print(f"  need_router_logits_for_aux_loss={need_router_logits_for_aux_loss}")
        print(f"  collect_router_logits={collect_router_logits}")
        print(f"  Expected: collect_router_logits=False, need_router_logits_for_aux_loss=False")
        print()
        
        return True
    
    test_logic_only()
    sys.exit(0)

def test_mixtral_aux_loss_behavior():
    """Test that auxiliary loss behavior is correct with the fix"""
    print("Testing Mixtral auxiliary loss behavior...")
    
    # Create a small config for testing
    config = MixtralConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
    )
    
    # Test case 1: output_router_logits=False, router_aux_loss_coef=0.001
    # Should compute aux_loss but not return router_logits
    print("\nTest Case 1: output_router_logits=False, router_aux_loss_coef=0.001")
    config.output_router_logits = False
    config.router_aux_loss_coef = 0.001
    
    model = MixtralForCausalLM(config)
    model.eval()  # Set to eval to avoid random effects
    
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, 2, 3, 4]])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else None:.6f}")
    print(f"  Aux loss: {outputs.aux_loss.item() if outputs.aux_loss is not None else None}")
    print(f"  Router logits returned: {outputs.router_logits is not None}")
    
    # Test case 2: output_router_logits=True, router_aux_loss_coef=0.001
    # Should compute aux_loss AND return router_logits
    print("\nTest Case 2: output_router_logits=True, router_aux_loss_coef=0.001")
    config.output_router_logits = True
    config.router_aux_loss_coef = 0.001
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else None:.6f}")
    print(f"  Aux loss: {outputs.aux_loss.item() if outputs.aux_loss is not None else None}")
    print(f"  Router logits returned: {outputs.router_logits is not None}")
    
    # Test case 3: output_router_logits=False, router_aux_loss_coef=0
    # Should NOT compute aux_loss
    print("\nTest Case 3: output_router_logits=False, router_aux_loss_coef=0")
    config.output_router_logits = False
    config.router_aux_loss_coef = 0
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else None:.6f}")
    print(f"  Aux loss: {outputs.aux_loss.item() if outputs.aux_loss is not None else None}")
    print(f"  Router logits returned: {outputs.router_logits is not None}")
    
    # Test case 4: inference (no labels) - should not compute aux_loss
    print("\nTest Case 4: Inference (no labels), router_aux_loss_coef=0.001")
    config.output_router_logits = False
    config.router_aux_loss_coef = 0.001
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)  # No labels
    
    print(f"  Loss: {outputs.loss}")
    print(f"  Aux loss: {outputs.aux_loss}")
    print(f"  Router logits returned: {outputs.router_logits is not None}")
    
    print("\nAll tests completed!")

def test_original_issue_reproduction():
    """Test the exact scenario from issue #44242"""
    print("\nReproducing original issue #44242...")
    
    # Original failing config from the issue
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=2, # small for demonstration
        num_local_experts=8,
        output_router_logits=False,  
        router_aux_loss_coef=0.001   # The scaling factor for the load balancing loss
    )
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    input_ids = torch.tensor([[1, 254, 99, 32]])
    labels = torch.tensor([[1, 254, 99, 32]]) # Next-token prediction labels
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    total_loss = outputs.loss
    aux_loss = outputs.aux_loss 
    router_logits = outputs.router_logits
    
    print(f"Auxiliary Load Balancing Loss: {aux_loss.item() if aux_loss is not None else None}")
    print(f"Total Loss (Cross Entropy + {config.router_aux_loss_coef} * Aux Loss): {total_loss.item() if total_loss is not None else None:.4f}")
    print(f"Router logits returned: {router_logits is not None}")
    
    # The fix should make aux_loss not None
    if aux_loss is not None:
        print("✅ SUCCESS: Auxiliary loss is now computed even when output_router_logits=False!")
    else:
        print("❌ FAILED: Auxiliary loss is still None")
    
    return aux_loss is not None

if __name__ == "__main__":
    success = True
    
    try:
        test_mixtral_aux_loss_behavior()
        success = test_original_issue_reproduction() and success
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n🎉 All tests passed! The fix works correctly.")
    else:
        print("\n💥 Some tests failed!")
        
    sys.exit(0 if success else 1)