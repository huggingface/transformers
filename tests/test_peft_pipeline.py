import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Test with a small model
model_name = "gpt2"
text = "Hello, I'm a language model"

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a dummy PEFT model
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)
peft_model = get_peft_model(model, peft_config)

# Test base model pipeline
print("Testing base model pipeline:")
base_generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer
)
print(base_generator(text, max_length=30))

# Test PEFT model pipeline
print("\nTesting PEFT model pipeline:")
peft_generator = pipeline(
    task="text-generation",
    model=peft_model,
    tokenizer=tokenizer
)
print(peft_generator(text, max_length=30))

print("\nIf no error message appears between the two outputs, the fix is working correctly!")
