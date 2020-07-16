---
language: pt

widget:
- text: "Quem era Jim Henson? Jim Henson era um"
- text: "Em um achado chocante, o cientista descobriu um"
- text: "Barack Hussein Obama II, nascido em 4 de agosto de 1961, é"
- text: "Corrida por vacina contra Covid-19 já tem"
license: mit
datasets: 
- wikipedia
---

# GPorTuguese-2: a Language Model for Portuguese text generation (and more NLP tasks...)

## Introduction

GPorTuguese-2 (Portuguese GPT-2 small) is a state-of-the-art language model for Portuguese based on the GPT-2 small model. 

It was trained on Portuguese Wikipedia using **Transfer Learning and Fine-tuning techniques** in just over a day, on one GPU NVIDIA V100 32GB and with a little more than 1GB of training data. 

It is a proof-of-concept that it is possible to get a state-of-the-art language model in any language with low ressources. 

It was fine-tuned from the [English pre-trained GPT-2 small](https://huggingface.co/gpt2) using the Hugging Face libraries (Transformers and Tokenizers) wrapped into the [fastai v2](https://dev.fast.ai/) Deep Learning framework. All the fine-tuning fastai v2 techniques were used.

It is now available on Hugging Face. For further information or requests, please go to "[Faster than training from scratch — Fine-tuning the English GPT-2 in any language with Hugging Face and fastai v2 (practical case with Portuguese)](https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787)".

## Model

| Model                   | #params | Model file (pt/tf) | Arch.       | Training /Validation data (text)         |
|-------------------------|---------|--------------------|-------------|------------------------------------------|
| `gpt2-small-portuguese` | 124M    | 487M / 475M        | GPT-2 small | Portuguese Wikipedia (1.28 GB / 0.32 GB) |

## Evaluation results
In a little more than a day (we only used one GPU NVIDIA V100 32GB; through a Distributed Data Parallel (DDP) training mode, we could have divided by three this time to 10 hours, just with 2 GPUs), we got a loss of 3.17, an **accuracy of 37.99%** and a **perplexity of 23.76** (see the validation results table below).

| after ... epochs | loss | accuracy (%) | perplexity | time by epoch | cumulative time |
|------------------|------|--------------|------------|---------------|-----------------|
|         0        | 9.95 |      9.90    |  20950.94  |    00:00:00   |     00:00:00    |
|         1        | 3.64 |     32.52    |     38.12  |     5:48:31   |      5:48:31    |
|         2        | 3.30 |     36.29    |     27.16  |     5:38:18   |     11:26:49    |
|         3        | 3.21 |     37.46    |     24.71  |     6:20:51   |     17:47:40    |
|         4        | 3.19 |     37.74    |     24.21  |     6:06:29   |     23:54:09    |
|         5        | 3.17 |     37.99    |     23.76  |     6:16:22   |     30:10:31    |

## GPT-2 

*Note: information copied/pasted from [Model: gpt2 >> GPT-2](https://huggingface.co/gpt2#gpt-2)*

Pretrained model on English language using a causal language modeling (CLM) objective. It was introduced in this [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and first released at this [page](https://openai.com/blog/better-language-models/) (February 14, 2019).

Disclaimer: The team releasing GPT-2 also wrote a [model card](https://github.com/openai/gpt-2/blob/master/model_card.md) for their model. Content from this model card has been written by the Hugging Face team to complete the information they provided and give specific examples of bias.

## Model description

*Note: information copied/pasted from [Model: gpt2 >> Model description](https://huggingface.co/gpt2#model-description)*

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

## How to use GPorTuguese-2 with HuggingFace (PyTorch)

The following code use PyTorch. To use TensorFlow, check the below corresponding paragraph.

### Load GPorTuguese-2 and its sub-word tokenizer (Byte-level BPE)

```python
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
model = AutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Get sequence length max of 1024
tokenizer.model_max_length=1024 

model.eval()  # disable dropout (or leave in train mode to finetune)
```

### Generate one word

```python
# input sequence
text = "Quem era Jim Henson? Jim Henson era um"
inputs = tokenizer(text, return_tensors="pt")

# model output
outputs = model(**inputs, labels=inputs["input_ids"])
loss, logits = outputs[:2]
predicted_index = torch.argmax(logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])

# results
print('input text:', text)
print('predicted text:', predicted_text)

# input text: Quem era Jim Henson? Jim Henson era um
# predicted text:  homem
```

### Generate one full sequence

```python
# input sequence
text = "Quem era Jim Henson? Jim Henson era um"
inputs = tokenizer(text, return_tensors="pt")

# model output using Top-k sampling text generation method
sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=50256,
                                do_sample=True, 
                                max_length=50, # put the token number you want
                                top_k=40,
                                num_return_sequences=1)

# generated sequence
for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))

# >> Generated text
# Quem era Jim Henson? Jim Henson era um executivo de televisão e diretor de um grande estúdio de cinema mudo chamado Selig,
# depois que o diretor de cinema mudo Georges Seuray dirigiu vários filmes para a Columbia e o estúdio.    
```

## How to use GPorTuguese-2 with HuggingFace (TensorFlow)

The following code use TensorFlow. To use PyTorch, check the above corresponding paragraph.

### Load GPorTuguese-2 and its sub-word tokenizer (Byte-level BPE)

```python
from transformers import AutoTokenizer, TFAutoModelWithLMHead
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
model = TFAutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Get sequence length max of 1024
tokenizer.model_max_length=1024 

model.eval()  # disable dropout (or leave in train mode to finetune)
```

### Generate one full sequence

```python
# input sequence
text = "Quem era Jim Henson? Jim Henson era um"
inputs = tokenizer.encode(text, return_tensors="tf")

# model output using Top-k sampling text generation method
outputs = model.generate(inputs, eos_token_id=50256, pad_token_id=50256, 
                         do_sample=True,
                         max_length=40,
                         top_k=40)
print(tokenizer.decode(outputs[0]))

# >> Generated text
# Quem era Jim Henson? Jim Henson era um amigo familiar da família. Ele foi contratado pelo seu pai 
# para trabalhar como aprendiz no escritório de um escritório de impressão, e então começou a ganhar dinheiro

```

## Limitations and bias

The training data used for this model come from Portuguese Wikipedia. We know it contains a lot of unfiltered content from the internet, which is far from neutral. As the openAI team themselves point out in their model card:

> Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true. Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans > unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes.

## Author

Portuguese GPT-2 small was trained and evaluated by [Pierre GUILLOU](https://www.linkedin.com/in/pierreguillou/) thanks to the computing power of the GPU (GPU NVIDIA V100 32 Go) of the [AI Lab](https://www.linkedin.com/company/ailab-unb/) (University of Brasilia) to which I am attached as an Associate Researcher in NLP and the participation of its directors in the definition of NLP strategy, Professors Fabricio Ataides Braz and Nilton Correia da Silva.

## Citation
If you use our work, please cite:

```bibtex
@inproceedings{pierre2020gpt2smallportuguese,
  title={GPorTuguese-2 (Portuguese GPT-2 small): a Language Model for Portuguese text generation (and more NLP tasks...)},
  author={Pierre Guillou},
  year={2020}
}
```
