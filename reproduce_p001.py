#!/usr/bin/env python3
"""
Reproduction script for P-001: LLaVA-OneVision 7B eager attention issue
⚠️ WARNING: This downloads and runs a 7B model - requires ~14GB+ RAM/VRAM
"""
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

def test_p001_reproduction():
    """Reproduce the P-001 issue with 7B model."""
    
    print("🧪 Reproducing P-001: LLaVA-OneVision 7B eager attention issue")
    print("⚠️  This will download ~13GB model - ensure sufficient storage/memory!")
    
    model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model with problematic settings
    print("Loading 7B model with eager+fp16 (problematic settings)...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
        device_map='auto'
    )
    
    # Prepare inputs
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
    
    # Generate with problematic settings
    print("Generating with eager+fp16...")
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = processor.decode(output[0][2:], skip_special_tokens=True)
    
    print(f"Result: {result}")
    
    if result.strip() in ["!", "assistant\n!", ""] or len(result.strip()) < 10:
        print("❌ CONFIRMED: P-001 issue reproduced - garbage output detected")
        return False
    else:
        print("✅ UNEXPECTED: Model generated reasonable output")
        return True

if __name__ == "__main__":
    success = test_p001_reproduction()
    if not success:
        print("\n💡 Try these workarounds:")
        print("   - Use torch.bfloat16 instead of torch.float16")
        print("   - Use attn_implementation='sdpa' instead of 'eager'")
