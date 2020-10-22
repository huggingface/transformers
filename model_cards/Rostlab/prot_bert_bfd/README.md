---
language: protein
tags:
- protein language model
datasets:
- BFD
---

# ProtBert-BFD model

Pretrained model on protein sequences using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://doi.org/10.1101/2020.07.12.199554) and first released in
[this repository](https://github.com/agemagician/ProtTrans). This model is trained on uppercase amino acids: it only works with capital letter amino acids.


## Model description

ProtBert-BFD is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion.
This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those protein sequences.

One important difference between our Bert model and the original Bert version is the way of dealing with sequences as separate documents
This means the Next sentence prediction is not used, as each sequence is treated as a complete document.
The masking follows the original Bert training with randomly masks 15% of the amino acids in the input. 

At the end, the feature extracted from this model revealed that the LM-embeddings from unlabeled data (only protein sequences) captured important biophysical properties governing protein
shape.
This implied learning some of the grammar of the language of life realized in protein sequences.

## Intended uses & limitations

The model could be used for protein feature extraction or to be fine-tuned on downstream tasks.
We have noticed in some tasks you could gain more accuracy by fine-tuning the model rather than using it as a feature extractor.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import BertForMaskedLM, BertTokenizer, pipeline
>>> tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
>>> model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
>>> unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T')

[{'score': 0.1165614128112793,
  'sequence': '[CLS] D L I P T S S K L V V L D T S L Q V K K A F F A L V T [SEP]',
  'token': 5,
  'token_str': 'L'},
 {'score': 0.08976086974143982,
  'sequence': '[CLS] D L I P T S S K L V V V D T S L Q V K K A F F A L V T [SEP]',
  'token': 8,
  'token_str': 'V'},
 {'score': 0.08864385634660721,
  'sequence': '[CLS] D L I P T S S K L V V S D T S L Q V K K A F F A L V T [SEP]',
  'token': 10,
  'token_str': 'S'},
 {'score': 0.06227643042802811,
  'sequence': '[CLS] D L I P T S S K L V V A D T S L Q V K K A F F A L V T [SEP]',
  'token': 6,
  'token_str': 'A'},
 {'score': 0.06194969266653061,
  'sequence': '[CLS] D L I P T S S K L V V T D T S L Q V K K A F F A L V T [SEP]',
  'token': 15,
  'token_str': 'T'}]
```

Here is how to use this model to get the features of a given protein sequence in PyTorch:

```python
from transformers import BertModel, BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model(**encoded_input)
```

## Training data

The ProtBert-BFD model was pretrained on [BFD](https://bfd.mmseqs.com/), a dataset consisting of 2.1 billion protein sequences.

## Training procedure

### Preprocessing

The protein sequences are uppercased and tokenized using a single space and a vocabulary size of 21.
The inputs of the model are then of the form:

```
[CLS] Protein Sequence A [SEP] Protein Sequence B [SEP]
```

Furthermore, each protein sequence was treated as a separate document.
The preprocessing step was performed twice, once for a combined length (2 sequences) of less than 512 amino acids, and another time using a combined length (2 sequences) of less than 2048 amino acids.

The details of the masking procedure for each sequence followed the original Bert model as following:
- 15% of the amino acids are masked.
- In 80% of the cases, the masked amino acids are replaced by `[MASK]`.
- In 10% of the cases, the masked amino acids are replaced by a random amino acid (different) from the one they replace.
- In the 10% remaining cases, the masked amino acids are left as is.

### Pretraining

The model was trained on a single TPU Pod V3-1024 for one million steps in total.
800k steps using sequence length 512 (batch size 32k), and 200K steps using sequence length 2048 (batch size 6k).
The optimizer used is Lamb with a learning rate of 0.002, a weight decay of 0.01, learning rate warmup for 140k steps and linear decay of the learning rate after.

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Test results :

| Task/Dataset | secondary structure (3-states) | secondary structure (8-states)  |  Localization | Membrane  |
|:-----:|:-----:|:-----:|:-----:|:-----:|
|   CASP12  | 76 | 65 |    |    |
|   TS115   | 84 | 73 |    |    | 
|   CB513   | 83 | 70 |    |    |
|  DeepLoc  |    |    | 78 | 91 |

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
