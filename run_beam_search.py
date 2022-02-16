#!/usr/bin/env python3
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("hf-internal-testing/processor_with_lm")
model = AutoModelForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")

input_values = processor([160_000 * [0.1]], return_tensors="pt").input_values
logit_ids = torch.argmax(model(input_values).logits, axis=-1)[0]
logit_ids[40:60] = 8
logit_ids[160:240] = 20

processor._retrieve_time_stamps(logit_ids.detach().numpy(), model.stride)

out = processor.decode(logits.detach().numpy()[0, :, :16])
