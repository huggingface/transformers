import safetensors.torch
import os

# Define the path to our fake model's weights file
file_path = os.path.join("fake_qwen_model", "model.safetensors")

print(f"Creating a valid, empty safetensors file at: {file_path}")

# Create an empty dictionary to represent a model with zero tensors
empty_tensors = {}

# Use the safetensors library to save this empty dictionary to a file.
# This will create a file with a valid header that describes zero tensors.
safetensors.torch.save_file(empty_tensors, file_path)

print("File created successfully.")