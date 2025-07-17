# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers"
# ]
#
# [tool.uv.sources]
# transformers = { path = ".", editable = true }
# ///
import torch
from transformers import StateEmbeddingModel


model_name = "arcinstitute/SE-600M"
model = StateEmbeddingModel.from_pretrained(model_name)

torch.manual_seed(0)
input_ids = torch.randn((1, 1, 5120), dtype=torch.float32)
mask = torch.ones((1, 1, 5120), dtype=torch.bool)
mask[:, :, 2560:] = False
print("Input sum:\t", input_ids.sum())
print("Mask sum:\t", mask.sum())

outputs = model(input_ids, mask)
print("Output sum:\t", outputs["gene_output"].sum())

# Input sum:	tensor(-38.6611)
# Mask sum:	    tensor(2560)
# Output sum:	tensor(-19.6819, grad_fn=<SumBackward0>)