---
language: en
tags:
- tapas
- masked-lm
license: apache-2.0
---

# TAPAS base model 

This model corresponds to the `tapas_inter_masklm_base_reset` checkpoint of the [original Github repository](https://github.com/google-research/tapas). 

Disclaimer: The team releasing TAPAS did not write a model card for this model so this model card has been written by
the Hugging Face team and contributors.

## Model description

TAPAS is a BERT-like transformers model pretrained on a large corpus of English data from Wikipedia in a self-supervised fashion. 
This means it was pretrained on the raw tables and associated texts only, with no humans labelling them in any way (which is why it
can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a (flattened) table and associated context, the model randomly masks 15% of the words in 
  the input, then runs the entire (partially masked) sequence through the model. The model then has to predict the masked words. 
  This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, 
  or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional 
  representation of a table and associated text.
- Intermediate pre-training: to encourage numerical reasoning on tables, the authors additionally pre-trained the model by creating 
  a balanced dataset of millions of syntactically created training examples. Here, the model must predict (classify) whether a sentence 
  is supported or refuted by the contents of a table. The training examples are created based on synthetic as well as counterfactual statements.

This way, the model learns an inner representation of the English language used in tables and associated texts, which can then be used 
to extract features useful for downstream tasks such as answering questions about a table, or determining whether a sentence is entailed
or refuted by the contents of a table. Fine-tuning is done by adding classification heads on top of the pre-trained model, and then jointly
train the randomly initialized classification heads with the base model on a labelled dataset. 

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task. 
See the [model hub](https://huggingface.co/models?filter=tapas) to look for fine-tuned versions on a task that interests you.


Here is how to use this model to get the features of a given table-text pair in PyTorch:

```python
from transformers import TapasTokenizer, TapasModel
import pandas as pd
tokenizer = TapasTokenizer.from_pretrained('tapase-base')
model = TapasModel.from_pretrained("tapas-base")
data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
         'Age': ["56", "45", "59"],
         'Number of movies': ["87", "53", "69"]
}
table = pd.DataFrame.from_dict(data)
queries = ["How many movies has George Clooney played in?"]
text = "Replace me by any text you'd like."
encoded_input = tokenizer(table=table, queries=queries, return_tensors='pt')
output = model(**encoded_input)
```

## Training data

For masked language modeling (MLM), a collection of 6.2 million tables was extracted from English Wikipedia: 3.3M of class [Infobox](https://en.wikipedia.org/wiki/Help:Infobox)
and 2.9M of class WikiTable. The author only considered tables with at most 500 cells. As a proxy for questions that appear in the 
downstream tasks, the authros extracted the table caption, article title, article description, segment title and text of the segment 
the table occurs in as relevant text snippets. In this way, 21.3M snippets were created. For more info, see the original [TAPAS paper](https://www.aclweb.org/anthology/2020.acl-main.398.pdf).

For intermediate pre-training, 2 tasks are introduced: one based on synthetic and the other from counterfactual statements. The first one 
generates a sentence by sampling from a set of logical expressions that filter, combine and compare the information on the table, which is 
required in table entailment (e.g., knowing that Gerald Ford is taller than the average president requires summing
all presidents and dividing by the number of presidents). The second one corrupts sentences about tables appearing on Wikipedia by swapping 
entities for plausible alternatives. Examples of the two tasks can be seen in Figure 1. The procedure is described in detail in section 3 of 
the [TAPAS follow-up paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27.pdf).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Context [SEP] Flattened table [SEP]
```

The details of the masking procedure for each sequence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

The details of the creation of the synthetic and counterfactual examples can be found in the [follow-up paper](https://arxiv.org/abs/2010.00571). 

### Pretraining

The model was trained on 32 Cloud TPU v3 cores for one million steps with maximum sequence length 512 and batch size of 512.
In this setup, pre-training takes around 3 days. The optimizer used is Adam with a learning rate of 5e-5, and a warmup ratio 
of 0.10. 


### BibTeX entry and citation info

```bibtex
@misc{herzig2020tapas,
      title={TAPAS: Weakly Supervised Table Parsing via Pre-training}, 
      author={Jonathan Herzig and Paweł Krzysztof Nowak and Thomas Müller and Francesco Piccinno and Julian Martin Eisenschlos},
      year={2020},
      eprint={2004.02349},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

```bibtex
@misc{eisenschlos2020understanding,
      title={Understanding tables with intermediate pre-training}, 
      author={Julian Martin Eisenschlos and Syrine Krichene and Thomas Müller},
      year={2020},
      eprint={2010.00571},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```