#!/usr/bin/env python3
"""
Test script to reproduce Issue #40376: Training Arguments resets accelerator

This reproduces the bug where TrainingArguments silently resets the accelerator state.
"""

def test_issue_40376():
    """Reproduce the Training Arguments reset bug."""
    print("🔍 Testing Issue #40376: Training Arguments Reset Bug")
    print("=" * 60)
    
    try:
        # Import the required modules
        from accelerate import Accelerator
        from transformers import TrainingArguments
        
        print("✅ Successfully imported required modules")
        
        # Create accelerator
        print("\n1. Creating Accelerator...")
        accelerator = Accelerator()
        
        # Check initial state
        print("2. Checking initial accelerator state...")
        print(f"   (L3) AcceleratorState has distributed_type: {hasattr(accelerator.state, 'distributed_type')}")
        print(f"   (L4) distributed_type value: {getattr(accelerator.state, 'distributed_type', 'NOT_FOUND')}")
        
        # This should NOT affect the accelerator state
        print("\n3. Creating TrainingArguments...")
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            save_steps=1000,
            save_total_limit=2,
        )
        
        # Check state after TrainingArguments creation
        print("4. Checking accelerator state after TrainingArguments creation...")
        print(f"   (L7) AcceleratorState has distributed_type: {hasattr(accelerator.state, 'distributed_type')}")
        print(f"   (L8) distributed_type value: {getattr(accelerator.state, 'distributed_type', 'NOT_FOUND')}")
        
        # Analysis
        print("\n5. Analysis:")
        initial_has_attr = hasattr(accelerator.state, 'distributed_type')
        final_has_attr = hasattr(accelerator.state, 'distributed_type')
        
        if initial_has_attr and not final_has_attr:
            print("   🚨 BUG CONFIRMED: TrainingArguments silently reset accelerator state!")
            print("   ❌ This is the issue described in #40376")
            return False
        elif initial_has_attr and final_has_attr:
            print("   ✅ No bug detected: Accelerator state preserved")
            return True
        else:
            print("   ⚠️  Unexpected state: Need further investigation")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This might be related to the TensorFlow import issue (#40292)")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run the test."""
    print("🚀 Hugging Face Transformers - Issue #40376 Reproduction")
    print("=" * 70)
    
    success = test_issue_40376()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Test completed successfully - no bug detected")
    else:
        print("🐛 Bug reproduced - Issue #40376 confirmed!")
        print("\n💡 Next steps:")
        print("   1. Investigate TrainingArguments.__init__ method")
        print("   2. Find where accelerator state is being modified")
        print("   3. Implement fix to preserve accelerator state")
    
    return success

if __name__ == "__main__":
    main()
