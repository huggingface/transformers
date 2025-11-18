#!/usr/bin/env python3

"""
Quick test to verify the VibeVoice processor works with the CSM approach (no custom tokenizer).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from transformers import Qwen2TokenizerFast, VibeVoiceFeatureExtractor, VibeVoiceProcessor

def test_processor_speech_tokens():
    """Test that processor correctly handles speech tokens like CSM."""
    
    # Create tokenizer and feature extractor
    tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-1.5B")
    feature_extractor = VibeVoiceFeatureExtractor()
    
    # Create processor 
    processor = VibeVoiceProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Test that speech tokens are correctly identified
    print("Testing speech token handling...")
    print(f"Speech start token: {processor.speech_start_token}")
    print(f"Speech start ID: {processor.speech_start_id}")
    print(f"Speech end token: {processor.speech_end_token}")
    print(f"Speech end ID: {processor.speech_end_id}")
    print(f"Speech diffusion token: {processor.speech_diffusion_token}")
    print(f"Speech diffusion ID: {processor.speech_diffusion_id}")
    
    # Test backward compatibility aliases
    print(f"Backward compat _speech_start_token: {processor._speech_start_token}")
    print(f"Backward compat _speech_end_token: {processor._speech_end_token}")
    print(f"Backward compat _speech_diffusion_token: {processor._speech_diffusion_token}")
    
    # Test token ID conversion
    expected_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    expected_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    expected_diffusion_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    
    assert processor.speech_start_id == expected_start_id, f"Expected {expected_start_id}, got {processor.speech_start_id}"
    assert processor.speech_end_id == expected_end_id, f"Expected {expected_end_id}, got {processor.speech_end_id}"
    assert processor.speech_diffusion_id == expected_diffusion_id, f"Expected {expected_diffusion_id}, got {processor.speech_diffusion_id}"
    
    print("✅ All tests passed! Processor correctly handles speech tokens using CSM approach.")

if __name__ == "__main__":
    test_processor_speech_tokens()
