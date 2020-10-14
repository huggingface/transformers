---
language: protein
tags:
- protein language model
datasets:
- BFD
---

# ProtT5-XL-BFD model

Pretrained model on protein sequences using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://doi.org/10.1101/2020.07.12.199554) and first released in
[this repository](https://github.com/agemagician/ProtTrans). This model is trained on uppercase amino acids: it only works with capital letter amino acids.


## Model description

ProtT5-XL-BFD is based on the `t5-3b` model and was pretrained on a large corpus of protein sequences in a self-supervised fashion.
This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those protein sequences.

One important difference between this T5 model and the original T5 version is the denosing objective.
The original T5-3B model was pretrained using a span denosing objective, while this model was pre-trained with a Bart-like MLM denosing objective.
The masking probability is consistent with the original T5 training by randomly masking 15% of the amino acids in the input.

It has been shown that the features extracted from this self-supervised model (LM-embeddings) captured important biophysical properties governing protein shape.
shape.
This implied learning some of the grammar of the language of life realized in protein sequences.

## Intended uses & limitations

The model could be used for protein feature extraction or to be fine-tuned on downstream tasks.
We have noticed in some tasks on can gain more accuracy by fine-tuning the model rather than using it as a feature extractor.
We have also noticed that for feature extraction, its better to use the feature extracted from the encoder not from the decoder.

### How to use

Here is how to use this model to extract the features of a given protein sequence in PyTorch:

```python
from transformers import T5Tokenizer, T5Model
import re
import torch

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)

model = T5Model.from_pretrained("Rostlab/prot_t5_xl_bfd")

sequences_Example = ["A E T C Z A O","S K T Z P"]

sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

input_ids = torch.tensor(ids['input_ids'])
attention_mask = torch.tensor(ids['attention_mask'])

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)

# For feature extraction we recommend to use the encoder embedding
encoder_embedding = embedding[2].cpu().numpy()
decoder_embedding = embedding[0].cpu().numpy()
```

## Training data

The ProtT5-XL-BFD model was pretrained on [BFD](https://bfd.mmseqs.com/), a dataset consisting of 2.1 billion protein sequences.

## Training procedure

### Preprocessing

The protein sequences are uppercased and tokenized using a single space and a vocabulary size of 21. The rare amino acids "U,Z,O,B" were mapped to "X".
The inputs of the model are then of the form:

```
Protein Sequence [EOS]
```

The preprocessing step was performed on the fly, by cutting and padding the protein sequences up to 512 tokens.

The details of the masking procedure for each sequence are as follows:
- 15% of the amino acids are masked.
- In 90% of the cases, the masked amino acids are replaced by `[MASK]` token.
- In 10% of the cases, the masked amino acids are replaced by a random amino acid (different) from the one they replace.

### Pretraining

The model was trained on a single TPU Pod V3-1024 for 1.2 million steps in total, using sequence length 512 (batch size 4k).
It has a total of approximately 3B parameters and was trained using the encoder-decoder architecture.
The optimizer used is AdaFactor with inverse square root learning rate schedule for pre-training.


## Evaluation results

When the model is used for feature etraction, this model achieves the following results:

Test results :

| Task/Dataset | secondary structure (3-states) | secondary structure (8-states)  |  Localization | Membrane  |
|:-----:|:-----:|:-----:|:-----:|:-----:|
|   CASP12  | 77 | 66 |    |    |
|   TS115   | 85 | 74 |    |    | 
|   CB513   | 84 | 71 |    |    |
|  DeepLoc  |    |    | 77 | 91 |

### BibTeX entry and citation info

```bibtex
@article {Elnaggar2020.07.12.199554,
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and BHOWMIK, DEBSINDHU and Rost, Burkhard},
	title = {ProtTrans: Towards Cracking the Language of Life{\textquoteright}s Code Through Self-Supervised Deep Learning and High Performance Computing},
	elocation-id = {2020.07.12.199554},
	year = {2020},
	doi = {10.1101/2020.07.12.199554},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models (LMs) taken from Natural Language Processing (NLP). These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive language models (Transformer-XL, XLNet) and two auto-encoder models (Bert, Albert) on data from UniRef and BFD containing up to 393 billion amino acids (words) from 2.1 billion protein sequences (22- and 112 times the entire English Wikipedia). The LMs were trained on the Summit supercomputer at Oak Ridge National Laboratory (ORNL), using 936 nodes (total 5616 GPUs) and one TPU Pod (V3-512 or V3-1024). We validated the advantage of up-scaling LMs to larger models supported by bigger data by predicting secondary structure (3-states: Q3=76-84, 8 states: Q8=65-73), sub-cellular localization for 10 cellular compartments (Q10=74) and whether a protein is membrane-bound or water-soluble (Q2=89). Dimensionality reduction revealed that the LM-embeddings from unlabeled data (only protein sequences) captured important biophysical properties governing protein shape. This implied learning some of the grammar of the language of life realized in protein sequences. The successful up-scaling of protein LMs through HPC to larger data sets slightly reduced the gap between models trained on evolutionary information and LMs. Availability ProtTrans: \&lt;a href="https://github.com/agemagician/ProtTrans"\&gt;https://github.com/agemagician/ProtTrans\&lt;/a\&gt;Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554},
	eprint = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554.full.pdf},
	journal = {bioRxiv}
}
```

> Created by [Ahmed Elnaggar/@Elnaggar_AI](https://twitter.com/Elnaggar_AI) | [LinkedIn](https://www.linkedin.com/in/prof-ahmed-elnaggar/)
