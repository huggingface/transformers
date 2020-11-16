---
language: "c++"
tags:
- exbert
- authorship-identification
- fire2020
- pan2020
- ai-soco
- classification
license: "mit"
datasets:
- ai-soco
metrics:
- accuracy
---

# ai-soco-c++-roberta-tiny-clas

## Model description

`ai-soco-c++-roberta-tiny` model fine-tuned on [AI-SOCO](https://sites.google.com/view/ai-soco-2020) task.

#### How to use

You can use the model directly after tokenizing the text using the provided tokenizer with the model files.

#### Limitations and bias

The model is limited to C++ programming language only.

## Training data

The model initialized from [`ai-soco-c++-roberta-tiny`](https://github.com/huggingface/transformers/blob/master/model_cards/aliosm/ai-soco-c++-roberta-tiny) model and trained using [AI-SOCO](https://sites.google.com/view/ai-soco-2020) dataset to do text classification.

## Training procedure

The model trained on Google Colab platform using V100 GPU for 10 epochs, 32 batch size, 512 max sequence length (sequences larger than 512 were truncated). Each continues 4 spaces were converted to a single tab character (`\t`) before tokenization.

## Eval results

The model achieved 87.66%/87.46% accuracy on AI-SOCO task and ranked in the 9th place.

### BibTeX entry and citation info

```bibtex
@inproceedings{ai-soco-2020-fire,
    title = "Overview of the {PAN@FIRE} 2020 Task on {Authorship Identification of SOurce COde (AI-SOCO)}",
    author = "Fadel, Ali and Musleh, Husam and Tuffaha, Ibraheem and Al-Ayyoub, Mahmoud and Jararweh, Yaser and Benkhelifa, Elhadj and Rosso, Paolo",
    booktitle = "Proceedings of The 12th meeting of the Forum for Information Retrieval Evaluation (FIRE 2020)",
    year = "2020"
}
```

<a href="https://huggingface.co/exbert/?model=aliosm/ai-soco-c++-roberta-tiny-clas">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
