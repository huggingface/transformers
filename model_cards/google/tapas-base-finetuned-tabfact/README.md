---
language: en
tags:
- tapas
- sequence-classification
license: apache-2.0
---

# TAPAS base model fine-tuned on Tabular Fact Checking (TabFact) 

This model corresponds to the `tapas_tabfact_inter_masklm_base_reset` checkpoint of the [original Github repository](https://github.com/google-research/tapas). 

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
or refuted by the contents of a table. Fine-tuning is done by adding a classification head on top of the pre-trained model, and then 
jointly train this randomly initialized classification head with the base model on TabFact. 


## Intended uses & limitations

You can use this model for classifying whether a sentence is supported or refuted by the contents of a table.

For code examples, we refer to the documentation of TAPAS on the HuggingFace website. 


## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Sentence [SEP] Flattened table [SEP]
```

### Fine-tuning

The model was fine-tuned on 32 Cloud TPU v3 cores for 80,000 steps with maximum sequence length 512 and batch size of 512.
In this setup, fine-tuning takes around 14 hours. The optimizer used is Adam with a learning rate of 2e-5, and a warmup 
ratio of 0.05. See the [paper](https://arxiv.org/abs/2010.00571) for more details (appendix A2). 


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