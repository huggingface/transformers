#!/usr/bin/env python3
import torch

from transformers import GPT2DoubleHeadsModel


model = GPT2DoubleHeadsModel.from_pretrained("gpt2")
outputs = model.generate(torch.tensor([8 * [0]]), num_beams=2)
