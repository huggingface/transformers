<!--Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# HindiCausalLM

## Overview

The HindiCausalLM model is a causal language model specifically designed for the Hindi language. It was developed by ConvaiInnovations and is based on a modern transformer architecture with grouped query attention mechanisms.

The original model implementation and weights are available on the [Hugging Face Hub](https://huggingface.co/convaiinnovations/hindi-causal-lm).

## Model Details

HindiCausalLM is a decoder-only transformer model designed for Hindi language generation. The model architecture includes the following key features:

- Decoder-only transformer architecture
- Grouped Query Attention (GQA) for better inference efficiency
- SiLU activation functions in the feed-forward networks
- Layer normalization applied before attention and feed-forward blocks
- Rotary Position Embeddings (RoPE) for handling token positions

The base model has 12 layers, 768 hidden dimensions, 16 attention heads, 4 key-value heads, and approximately 125 million parameters.

## Usage Examples

### Text Generation

```python
from transformers import HindiCausalLMForCausalLM, HindiCausalLMTokenizer

# Load model and tokenizer
model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")

# Generate text
input_text = "भारत एक विशाल देश है"  # "India is a vast country"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate
outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_return_sequences=1, 
    temperature=0.7, 
    top_p=0.9,
    do_sample=True
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Inference with TensorFlow

```python
from transformers import TFHindiCausalLMForCausalLM, HindiCausalLMTokenizer
import tensorflow as tf

# Load model and tokenizer
model = TFHindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")

# Process input
input_text = "हिंदी भाषा बहुत समृद्ध है"  # "Hindi language is very rich"
inputs = tokenizer(input_text, return_tensors="tf")

# Forward pass
outputs = model(inputs)
logits = outputs.logits

# Get the next token prediction
next_token_logits = logits[:, -1, :]
next_token_id = tf.argmax(next_token_logits, axis=-1)
next_token = tokenizer.decode(next_token_id.numpy())

print(f"Predicted next token: {next_token}")
```

### Using the Model for Classification Tasks

```python
from transformers import HindiCausalLMForSequenceClassification, HindiCausalLMTokenizer
import torch

# Load model and tokenizer
model = HindiCausalLMForSequenceClassification.from_pretrained(
    "convaiinnovations/hindi-causal-lm", num_labels=3
)
tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")

# Example texts
texts = ["यह एक सकारात्मक समीक्षा है", "यह एक नकारात्मक समीक्षा है", "यह एक तटस्थ समीक्षा है"]
inputs = tokenizer(texts, padding=True, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

print(f"Predictions: {predictions}")
```

## Resources

* [HindiCausalLM model card](https://huggingface.co/convaiinnovations/hindi-causal-lm)
* [Hindi language resources](https://huggingface.co/datasets?language=hi&multilinguality=monolingual)

## Model Inputs

The model accepts the following inputs:

* **input_ids** (torch.LongTensor of shape (batch_size, sequence_length)): Indices of input sequence tokens in the vocabulary.
* **attention_mask** (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
* **position_ids** (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of positions of each input sequence tokens in the position embeddings.
* **past_key_values** (Cache or tuple of tuple(torch.FloatTensor), optional): Contains pre-computed hidden-states to speed up sequential decoding.

## Outputs

### HindiCausalLMModel

```python
class transformers.models.hindicausallm.modeling_hindicausallm.HindiCausalLMModel
```

Base model outputs a `BaseModelOutputWithPast` with the following components:

* **last_hidden_state** (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)): Hidden states of the last layer.
* **past_key_values** (tuple(tuple(torch.FloatTensor)), optional): Cache containing pre-computed key and value states.
* **hidden_states** (tuple(torch.FloatTensor), optional): Hidden states of all layers.
* **attentions** (tuple(torch.FloatTensor), optional): Attention weights of all layers.

### HindiCausalLMForCausalLM

```python
class transformers.models.hindicausallm.modeling_hindicausallm.HindiCausalLMForCausalLM
```

Causal language modeling head outputs a `CausalLMOutputWithPast` with the following components:

* **loss** (torch.FloatTensor, optional): Language modeling loss (if labels provided).
* **logits** (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)): Prediction scores.
* **past_key_values** (tuple(tuple(torch.FloatTensor)), optional): Pre-computed key and value states.
* **hidden_states** (tuple(torch.FloatTensor), optional): Hidden states of all layers.
* **attentions** (tuple(torch.FloatTensor), optional): Attention weights of all layers.

### HindiCausalLMForSequenceClassification

```python
class transformers.models.hindicausallm.modeling_hindicausallm.HindiCausalLMForSequenceClassification
```

Sequence classification outputs a `SequenceClassifierOutputWithPast` with the following components:

* **loss** (torch.FloatTensor, optional): Classification loss (if labels provided).
* **logits** (torch.FloatTensor of shape (batch_size, config.num_labels)): Classification scores.
* **past_key_values** (tuple(tuple(torch.FloatTensor)), optional): Pre-computed key and value states.
* **hidden_states** (tuple(torch.FloatTensor), optional): Hidden states of all layers.
* **attentions** (tuple(torch.FloatTensor), optional): Attention weights of all layers.

## Bias and Limitations

As with all language models, HindiCausalLM might reproduce biases present in the training data. The model should be used with care, particularly in automated decision-making processes that affect people. The model's output should be reviewed by humans and may require additional filtering and validation.

The model is primarily designed for Hindi text generation and may not work well for other languages or dialects. Performance depends on the quality and diversity of the training data.

## Citation

```bibtex
@misc{hindicausallm,
  author = {ConvaiInnovations},
  title = {Hindi Causal Language Model},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/convaiinnovations/hindi-causal-lm}}
}
```