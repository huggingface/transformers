
import os
os.environ['PYTHONPATH'] = '.'
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("HannahRoseKirk/Hatemoji")
print(model.forward(input_ids = torch.tensor([[0]])))

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
print(model.forward(input_ids = torch.tensor([[0]])))





