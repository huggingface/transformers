from transformers import MarianTokenizer, MarianMTModel
import torch
import numpy as np
import onnxruntime as ort

# Load Marian model & tokenizer
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model.eval()

# Input sentence
input_sentence = (
    "Using handheld GPS devices and programs like Google Earth, "
    "members of the Trio Tribe, who live in the rainforests of southern Suriname, "
    "map out their ancestral lands to help strengthen their territorial claims."
)
inputs = tokenizer(
    input_sentence,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=64
)

# PyTorch generation
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=1,      # greedy decoding
        do_sample=False    # deterministic output
    )

print("=== PyTorch Generation ===")
print("Generated token IDs:", output_ids[0].tolist())
print("Decoded output:", tokenizer.decode(
    output_ids[0], skip_special_tokens=True))
