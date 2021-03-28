#!/usr/bin/env python3
from transformers import BigBirdModel
import torch

model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
model(torch.ones((1, 20), dtype=torch.long))

