# *De Novo* Drug Design with MLM

## What is it?

An approximation to [Generative Recurrent Networks for De Novo Drug Design](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201700111) but training a MLM (RoBERTa like) from scratch.

## Why?

As mentioned in the paper:
Generative artificial intelligence models present a fresh approach to chemogenomics and de novo drug design, as they provide researchers with the ability to narrow down their search of the chemical space and focus on regions of interest.
They used a generative *recurrent neural network (RNN)* containing long short‐term memory (LSTM) cell to capture the syntax of molecular representations in terms of SMILES strings.
The learned pattern probabilities can be used for de novo SMILES generation. This molecular design concept **eliminates the need for virtual compound library enumeration** and **enables virtual compound design without requiring secondary or external activity prediction**.


## My Goal 🎯

By training a MLM from scratch on 438552 (cleaned*) SMILES I wanted to build a model that learns this kind of molecular combinations so that given a partial SMILE it can generate plausible combinations so that it can be proposed as new drugs.
By cleaned SMILES I mean that I used their [SMILES cleaning script](https://github.com/topazape/LSTM_Chem/blob/master/cleanup_smiles.py) to remove duplicates, salts, and stereochemical information.
You can see the detailed process of gathering the data, preprocess it and train the LSTM in their [repo](https://github.com/topazape/LSTM_Chem).

## Fast usage with ```pipelines``` 🧪

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model='/mrm8488/chEMBL_smiles_v1',
    tokenizer='/mrm8488/chEMBL_smiles_v1'
)

# CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1 Atazanavir
smile1 = "CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)<mask>"

fill_mask(smile1)

# Output:
'''
[{'score': 0.6040295958518982,
  'sequence': '<s> CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)nc</s>',
  'token': 265},
 {'score': 0.2185731679201126,
  'sequence': '<s> CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)N</s>',
  'token': 50},
 {'score': 0.0642734169960022,
  'sequence': '<s> CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc</s>',
  'token': 261},
 {'score': 0.01932266168296337,
  'sequence': '<s> CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)CCCl</s>',
  'token': 452},
 {'score': 0.005068355705589056,
  'sequence': '<s> CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)C</s>',
  'token': 39}]
  '''
  ```
  ## More
  I also created a [second version](https://huggingface.co/mrm8488/chEMBL26_smiles_v2) without applying the cleaning SMILES script mentioned above. You can use it in the same way as this one.
  
  ```python
  fill_mask = pipeline(
    "fill-mask",
    model='/mrm8488/chEMBL26_smiles_v2',
    tokenizer='/mrm8488/chEMBL26_smiles_v2'
)
```
  
 [Original paper](https://www.ncbi.nlm.nih.gov/pubmed/29095571) Authors:
 <details>
Swiss Federal Institute of Technology (ETH), Department of Chemistry and Applied Biosciences, Vladimir–Prelog–Weg 4, 8093, Zurich, Switzerland,
Stanford University, Department of Computer Science, 450 Sierra Mall, Stanford, CA, 94305, USA,
inSili.com GmbH, 8049, Zurich, Switzerland,
Gisbert Schneider, Email: hc.zhte@trebsig.
</details>
  
 > Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
