import torch

from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")

question = "what does the front of the bus say at the top?"
input_ids = processor(text=question, add_special_tokens=False).input_ids
input_ids = [processor.tokenizer.cls_token_id] + input_ids
input_ids = torch.tensor(input_ids).unsqueeze(0)

print(processor.decode(input_ids[0].tolist()))


input_ids = processor(text=question, return_tensors="pt").input_ids

print(processor.decode(input_ids[0].tolist()))
