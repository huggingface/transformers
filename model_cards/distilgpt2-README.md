---
language: en
tags:
- exbert

license: apache-2.0
datasets:
- openwebtext
---

# DistilGPT2

DistilGPT2 English language model pretrained with the supervision of [GPT2](https://huggingface.co/gpt2) (the smallest version of GPT2) on [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a reproduction of OpenAI's WebText dataset. The model has 6 layers, 768 dimension and 12 heads, totalizing 82M parameters (compared to 124M parameters for GPT2). On average, DistilGPT2 is two times faster than GPT2.

On the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) benchmark, GPT2 reaches a perplexity on the test set of 16.3 compared to 21.1 for DistilGPT2 (after fine-tuning on the train set).

We encourage to check [GPT2](https://huggingface.co/gpt2) to know more about usage, limitations and potential biases.

<a href="https://huggingface.co/exbert/?model=distilgpt2">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
