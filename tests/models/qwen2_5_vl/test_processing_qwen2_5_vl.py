from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
import torch

tokenizer = Qwen2Tokenizer.from_pretrained(
    "./local_models/Qwen2.5-0.5B",
    trust_remote_code=False,
    local_files_only=True
)

text_inputs = tokenizer([""], return_tensors="pt")

if text_inputs.get("input_ids", None) is not None and text_inputs["input_ids"].numel() == 0:
    print("Triggered your patch code")
    text_inputs["input_ids"] = torch.empty((1, 0), dtype=torch.long)

# check output
print("Token IDs:", text_inputs["input_ids"])
print("Shape:", text_inputs["input_ids"].shape)
print("Dtype:", text_inputs["input_ids"].dtype)
assert text_inputs["input_ids"].shape == torch.Size([1, 0])
