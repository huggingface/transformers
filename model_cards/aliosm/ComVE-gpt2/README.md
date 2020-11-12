---
language: "en"
tags:
- exbert
- commonsense
- semeval2020
- comve
license: "mit"
datasets:
- ComVE
metrics:
- bleu
widget:
- text: "Chicken can swim in water. <|continue|>"
---

# ComVE-gpt2

## Model description

Finetuned model on Commonsense Validation and Explanation (ComVE) dataset introduced in [SemEval2020 Task4](https://competitions.codalab.org/competitions/21080) using a causal language modeling (CLM) objective.
The model is able to generate a reason why a given natural language statement is against commonsense.

## Intended uses & limitations

You can use the raw model for text generation to generate reasons why natural language statements are against commonsense.

#### How to use

You can use this model directly to generate reasons why the given statement is against commonsense using [`generate.sh`](https://github.com/AliOsm/SemEval2020-Task4-ComVE/tree/master/TaskC-Generation) script.

*Note:* make sure that you are using version `2.4.1` of `transformers` package. Newer versions has some issue in text generation and the model repeats the last token generated again and again.

#### Limitations and bias

The model biased to negate the entered sentence usually instead of producing a factual reason.

## Training data

The model is initialized from the [gpt2](https://github.com/huggingface/transformers/blob/master/model_cards/gpt2-README.md) model and finetuned using [ComVE](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation) dataset which contains 10K against commonsense sentences, each of them is paired with three reference reasons.

## Training procedure

Each natural language statement that against commonsense is concatenated with its reference reason with `<|continue|>` as a separator, then the model finetuned using CLM objective.
The model trained on Nvidia Tesla P100 GPU from Google Colab platform with 5e-5 learning rate, 5 epochs, 128 maximum sequence length and 64 batch size.

<center>
  <img src="https://i.imgur.com/xKbrwBC.png">
</center>

## Eval results

The model achieved 14.0547/13.6534 BLEU scores on SemEval2020 Task4: Commonsense Validation and Explanation development and testing dataset.

### BibTeX entry and citation info

```bibtex
@article{fadel2020justers,
  title={JUSTers at SemEval-2020 Task 4: Evaluating Transformer Models Against Commonsense Validation and Explanation},
  author={Fadel, Ali and Al-Ayyoub, Mahmoud and Cambria, Erik},
  year={2020}
}
```

<a href="https://huggingface.co/exbert/?model=aliosm/ComVE-gpt2">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
