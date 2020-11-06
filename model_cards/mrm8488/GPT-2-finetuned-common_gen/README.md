---
language: en
datasets:
- common_gen
widget:
- text: "<|endoftext|> apple, tree, pick:"
---

# GPT-2 fine-tuned on CommonGen

[GPT-2](https://huggingface.co/gpt2) fine-tuned on [CommonGen](https://inklab.usc.edu/CommonGen/index.html) for *Generative Commonsense Reasoning*.

## Details of GPT-2

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This
means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots
of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely,
it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence,
shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the
predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a
prompt.


## Details of the dataset 📚 

CommonGen is a constrained text generation task, associated with a benchmark dataset, to explicitly test machines for the ability of generative commonsense reasoning. Given a set of common concepts; the task is to generate a coherent sentence describing an everyday scenario using these concepts.

CommonGen is challenging because it inherently requires 1) relational reasoning using background commonsense knowledge, and 2) compositional generalization ability to work on unseen concept combinations. Our dataset, constructed through a combination of crowd-sourcing from AMT and existing caption corpora, consists of 30k concept-sets and 50k sentences in total.


| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| common_gen | train  | 67389   |
| common_gen | valid  | 4018    |
| common_gen | test   | 1497    |


## Model fine-tuning 🏋️‍

You can find the fine-tuning script [here](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)

## Model in Action 🚀

```bash
python ./transformers/examples/text-generation/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path="mrm8488/GPT-2-finetuned-common_gen" \
    --num_return_sequences 1 \
    --prompt "<|endoftext|> kid, room, dance:" \
    --stop_token "."
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain



