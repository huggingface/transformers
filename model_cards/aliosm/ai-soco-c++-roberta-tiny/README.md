---
language: "c++"
tags:
- exbert
- authorship-identification
- fire2020
- pan2020
- ai-soco
license: "mit"
datasets:
- ai-soco
metrics:
- perplexity
---

# ai-soco-c++-roberta-tiny

## Model description

From scratch pre-trained RoBERTa model with 1 layers and 12 attention heads using [AI-SOCO](https://sites.google.com/view/ai-soco-2020) dataset which consists of C++ codes crawled from CodeForces website.

## Intended uses & limitations

The model can be used to do code classification, authorship identification and other downstream tasks on C++ programming language.

#### How to use

You can use the model directly after tokenizing the text using the provided tokenizer with the model files.

#### Limitations and bias

The model is limited to C++ programming language only.

## Training data

The model initialized randomly and trained using [AI-SOCO](https://sites.google.com/view/ai-soco-2020) dataset which contains 100K C++ source codes.

## Training procedure

The model trained on Google Colab platform with 8 TPU cores for 200 epochs, 32\*8 batch size, 512 max sequence length and MLM objective. Other parameters were defaulted to the values mentioned in [`run_language_modelling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) script. Each continues 4 spaces were converted to a single tab character (`\t`) before tokenization.

### BibTeX entry and citation info

```bibtex
@inproceedings{ai-soco-2020-fire,
    title = "Overview of the {PAN@FIRE} 2020 Task on {Authorship Identification of SOurce COde (AI-SOCO)}",
    author = "Fadel, Ali and Musleh, Husam and Tuffaha, Ibraheem and Al-Ayyoub, Mahmoud and Jararweh, Yaser and Benkhelifa, Elhadj and Rosso, Paolo",
    booktitle = "Proceedings of The 12th meeting of the Forum for Information Retrieval Evaluation (FIRE 2020)",
    year = "2020"
}
```

<a href="https://huggingface.co/exbert/?model=aliosm/ai-soco-c++-roberta-tiny">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
