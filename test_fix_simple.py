#!/usr/bin/env python3
"""Simple test for the int-to-float conversion fix"""

import sys
sys.path.insert(0, "src")

# Test the fix directly without loading from Hub
from transformers.models.granite_speech.configuration_granite_speech import GraniteSpeechConfig

print("Testing granite_speech config with int embedding_multiplier...")
print("=" * 60)

# Simulate the problematic config that comes from the Hub
# (embedding_multiplier is an int instead of float)
test_config_dict = {
    "text_config": {
        "model_type": "granite",
        "vocab_size": 32000,
        "hidden_size": 2048,
        "embedding_multiplier": 12,  # ← This is an INT, should be FLOAT
        "logits_scaling": 2,           # ← Also INT
    }
}

try:
    print("\n1. Creating GraniteSpeechConfig with int multipliers...")
    config = GraniteSpeechConfig(**test_config_dict)
    
    print("✅ Config created successfully!")
    
    print(f"\n2. Checking type conversion:")
    emb_mult = config.text_config.embedding_multiplier
    logits_scale = config.text_config.logits_scaling
    
    print(f"   embedding_multiplier: {emb_mult} (type: {type(emb_mult).__name__})")
    print(f"   logits_scaling: {logits_scale} (type: {type(logits_scale).__name__})")
    
    assert isinstance(emb_mult, float), f"Expected float, got {type(emb_mult)}"
    assert isinstance(logits_scale, float), f"Expected float, got {type(logits_scale)}"
    assert emb_mult == 12.0
    assert logits_scale == 2.0
    
    print("\n" + "=" * 60)
    print("✅ TEST PASSED - Int-to-float conversion works!")
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ TEST FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
