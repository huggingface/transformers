from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Use Hugging Face model ID (will download only config & small files for CPU test)
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Starting CPU-only model load...")

# Load model on CPU only to avoid large GPU memory usage
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=None,  # forces CPU-only
    torch_dtype="auto"  # automatically picks float16/32 if available
)

print("Model loaded successfully on CPU!")