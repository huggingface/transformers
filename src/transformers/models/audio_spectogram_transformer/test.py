import torch

from transformers import AudioSpectogramTransformerConfig, AudioSpectogramTransformerForSequenceClassification


config = AudioSpectogramTransformerConfig(num_labels=527)
model = AudioSpectogramTransformerForSequenceClassification(config)

dummy_inputs = torch.randn(1, 1024, 128)

outputs = model(dummy_inputs)

print("Shape of logits:", outputs.logits.shape)

for name, param in model.named_parameters():
    print(name, param.shape)
