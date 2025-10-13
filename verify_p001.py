#!/usr/bin/env python3
"""
Verification script for P-001: LLaVA-OneVision 7B eager attention issue
Tests the reported garbage output problem with eager attention in fp16.
"""
import sys
import torch

def verify_p001():
    """Verify that P-001 (LLaVA-OneVision eager attention) exists."""
    
    print("🔍 Verifying P-001: LLaVA-OneVision 7B eager attention issue")
    print("=" * 70)
    
    try:
        # Check if transformers is available
        print("1. Checking transformers availability...")
        try:
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
            print("   ✅ Transformers imported successfully")
        except ImportError as e:
            print(f"   ❌ Cannot import transformers: {e}")
            print("   💡 Run: pip install transformers")
            return False
        
        # Check CUDA availability
        print("2. Checking CUDA availability...")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  CUDA not available - will use CPU (very slow)")
        
        # Check model availability (without downloading)
        print("3. Checking model configuration...")
        model_id_7b = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        model_id_0_5b = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        
        print(f"   📦 Target model (problematic): {model_id_7b}")
        print(f"   📦 Control model (works): {model_id_0_5b}")
        
        # Note: We don't actually load the 7B model here as it's large
        # This is just verification of setup
        
        print("\n4. Test scenario summary:")
        print("   🔧 Configuration that fails:")
        print("      - Model: llava-onevision-qwen2-7b-ov-hf")
        print("      - torch_dtype: torch.float16")
        print("      - attn_implementation: 'eager'")
        print("      - Expected result: Garbage output (e.g., '!')")
        print()
        print("   ✅ Configuration that works:")
        print("      - Model: llava-onevision-qwen2-0.5b-ov-hf (same settings)")
        print("      - OR: 7B model with torch.bfloat16")
        print("      - OR: 7B model with attn_implementation='sdpa'")
        
        print("\n5. Suspected root cause:")
        print("   - Numerical instability in eager attention with fp16")
        print("   - Softmax computed in fp16 instead of fp32")
        print("   - Non-additive attention mask handling")
        
        print("\n📋 P-001 VERIFICATION: SETUP READY")
        print("Environment is prepared to reproduce the issue.")
        print("⚠️  To fully verify, need to load 7B model (requires >14GB RAM)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

def create_reproduction_script():
    """Create a script to reproduce P-001 when ready."""
    
    script_content = '''#!/usr/bin/env python3
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
    
    if result.strip() in ["!", "assistant\\n!", ""] or len(result.strip()) < 10:
        print("❌ CONFIRMED: P-001 issue reproduced - garbage output detected")
        return False
    else:
        print("✅ UNEXPECTED: Model generated reasonable output")
        return True

if __name__ == "__main__":
    success = test_p001_reproduction()
    if not success:
        print("\\n💡 Try these workarounds:")
        print("   - Use torch.bfloat16 instead of torch.float16")
        print("   - Use attn_implementation='sdpa' instead of 'eager'")
'''
    
    with open("reproduce_p001.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📝 Created reproduce_p001.py for full testing when ready")

if __name__ == "__main__":
    success = verify_p001()
    if success:
        create_reproduction_script()
    sys.exit(0 if success else 1)