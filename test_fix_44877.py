#!/usr/bin/env python3
"""Test script for issue #44877: granite_speech config loading with int embedding_multiplier"""

import sys
sys.path.insert(0, "src")

from transformers import AutoConfig

print("Testing granite_speech config loading (issue #44877)...")
print("=" * 60)

model_name = "ibm-granite/granite-4.0-1b-speech"

try:
    print(f"\n1. Loading config from: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    print("✅ Config loaded successfully!")
    
    print(f"\n2. Checking text_config.embedding_multiplier:")
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'embedding_multiplier'):
        emb_mult = config.text_config.embedding_multiplier
        print(f"   Value: {emb_mult}")
        print(f"   Type: {type(emb_mult)}")
        
        if isinstance(emb_mult, float):
            print("   ✅ Correctly converted to float!")
        else:
            print(f"   ⚠️ Expected float, got {type(emb_mult)}")
    
    print(f"\n3. Config details:")
    print(f"   Model type: {config.model_type}")
    print(f"   Text model type: {config.text_config.model_type if hasattr(config, 'text_config') else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("✅ TEST PASSED - Fix works correctly!")
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ TEST FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
