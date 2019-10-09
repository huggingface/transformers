import os

os.environ["TPU_IP_ADDRESS"] = "192.168.0.2"
os.environ["TPU_NAME"] = "node-1"
os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;192.168.0.2:8470"


import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sequence = "This runs on TPU"
input_ids = torch.tensor([tokenizer.encode(sequence)], device=device)

model.train().to(device)

print(input_ids)
