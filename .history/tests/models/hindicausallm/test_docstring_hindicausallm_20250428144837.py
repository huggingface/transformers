# HindiCausalLM

This directory contains the implementation of the HindiCausalLM model for the Transformers library. This model is a causal language model designed specifically for the Hindi language, based on a transformer architecture.

## Model Overview

HindiCausalLM is a transformer-based language model trained on Hindi text data. The model architecture follows a standard decoder-only transformer design with the following features:

- **Grouped Query Attention (GQA)**: Improves inference efficiency by using fewer key-value heads than query heads
- **SiLU activation function**: Used in the feed-forward network layers
- **Layer normalization**: Applied before attention and feed-forward blocks
- **Rotary Position Embeddings (RoPE)**: For handling token positions

The model is available in the following sizes:
- **HindiCausalLM**: 12 layers, 768 hidden size, 16 attention heads, 4 key-value heads, ~125M parameters

## Usage Example

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

## Model Inputs

- **input_ids** (torch.LongTensor of shape (batch_size, sequence_length)): Indices of input sequence tokens in the vocabulary.
- **attention_mask** (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
- **position_ids** (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of positions of each input sequence tokens in the position embeddings.
- **past_key_values** (Cache or tuple of tuple(torch.FloatTensor), optional): Contains pre-computed hidden-states to speed up sequential decoding.
- **inputs_embeds** (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional): Embedded representation of input tokens.
- **use_cache** (bool, optional): Whether to use past key values for faster decoding.
- **output_attentions** (bool, optional): Whether to return attention weights of all layers.
- **output_hidden_states** (bool, optional): Whether to return hidden states of all layers.
- **return_dict** (bool, optional): Whether to return a ModelOutput object instead of a tuple.

## Model Outputs

- For **HindiCausalLMModel**: `BaseModelOutputWithPast` with the following fields:
  - **last_hidden_state** (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)): Hidden states of the last layer.
  - **past_key_values** (tuple(tuple(torch.FloatTensor)), optional): Contains pre-computed key and value hidden states of the attention blocks.
  - **hidden_states** (tuple(torch.FloatTensor), optional): Hidden states of all layers.
  - **attentions** (tuple(torch.FloatTensor), optional): Attention weights of all layers.

- For **HindiCausalLMForCausalLM**: `CausalLMOutputWithPast` with the following fields:
  - **loss** (torch.FloatTensor, optional): Language modeling loss (if labels provided).
  - **logits** (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)): Prediction scores of the language modeling head.
  - **past_key_values** (tuple(tuple(torch.FloatTensor)), optional): Contains pre-computed key and value hidden states of the attention blocks.
  - **hidden_states** (tuple(torch.FloatTensor), optional): Hidden states of all layers.
  - **attentions** (tuple(torch.FloatTensor), optional): Attention weights of all layers.

## Training

The model was trained on a large corpus of Hindi text. Training details:
- Trained with cross-entropy loss on the language modeling objective
- Used a cosine learning rate schedule with warmup
- Employed mixed-precision training

## Limitations

- The model is primarily designed for Hindi text generation and may not work well for other languages.
- Performance depends on the quality and diversity of the training data.
- Like all language models, it may generate inappropriate content if prompted with such.

## Original Implementation

This model is based on the ConvaiCausalLM architecture developed by ConvaiInnovations. The original implementation is available at: https://huggingface.co/convaiinnovations/hindi-causal-lm

## Citation

If you use this model in your research, please cite:

```
@misc{hindicausallm,
  author = {ConvaiInnovations},
  title = {Hindi Causal Language Model},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/convaiinnovations/hindi-causal-lm}}
}
```