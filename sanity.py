import torch
from transformers import StateEmbeddingModel, StateEmbeddingConfig

# Make reproducible
torch.manual_seed(42)

# Initializing a State Embedding style configuration
configuration = StateEmbeddingConfig()

# # Initializing a model from the State Embedding style configuration
model = StateEmbeddingModel(configuration)

# # Accessing the model configuration
configuration = model.config

print(configuration)
print(model)

# Example input for the model
input_ids = torch.ones((1, 1, 5120), dtype=torch.float32)
outputs = model(input_ids)
print(outputs["gene_output"].sum()) # tensor(3.4709, grad_fn=<SumBackward0>)


#### TODO

# from transformers import AutoTokenizer, StateModel, set_seed

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("arcinstitute/SE-600M")

# print(model)

