import torch
from transformers import Evo2ForCausalLM, Evo2Tokenizer

# Path to the converted model
model_path = "/tmp/evo2_hf"

print(f"Loading model from {model_path}...")
model = Evo2ForCausalLM.from_pretrained(model_path)
tokenizer = Evo2Tokenizer.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Input sequence (DNA)
sequence = "ACGTACGT"
print(f"Input: {sequence}")

# Tokenize
input_ids = tokenizer.encode(sequence, return_tensors="pt").to(device)

# Generate
print("Generating...")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20)

# Decode
generated_sequence = tokenizer.decode(output[0])
print(f"Output: {generated_sequence}")
