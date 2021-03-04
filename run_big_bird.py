#!/usr/bin/env python3
from transformers import BigBirdForQuestionAnswering
import torch

model = BigBirdForQuestionAnswering.from_pretrained("patrickvonplaten/big-bird-base-trivia-qa")

model(torch.tensor([4096 * [0]], dtype=torch.long))
