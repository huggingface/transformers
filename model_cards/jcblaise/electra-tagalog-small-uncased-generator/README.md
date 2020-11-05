---
language: tl
tags:
- electra
- tagalog
- filipino
license: gpl-3.0
inference: false
---

# ELECTRA Tagalog Small Uncased Generator
Tagalog ELECTRA model pretrained with a large corpus scraped from the internet. This model is part of a larger research project. We open-source the model to allow greater usage within the Filipino NLP community.

This is the generator model used to sample synthetic text and pretrain the discriminator. Only use this model for retraining and mask-filling. For the actual model for downstream tasks, please refer to the discriminator models.

## Usage
The model can be loaded and used in both PyTorch and TensorFlow through the HuggingFace Transformers package.

```python
from transformers import TFAutoModel, AutoModel, AutoTokenizer

# TensorFlow
model = TFAutoModel.from_pretrained('jcblaise/electra-tagalog-small-uncased-generator', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('jcblaise/electra-tagalog-small-uncased-generator', do_lower_case=False)

# PyTorch
model = AutoModel.from_pretrained('jcblaise/electra-tagalog-small-uncased-generator')
tokenizer = AutoTokenizer.from_pretrained('jcblaise/electra-tagalog-small-uncased-generator', do_lower_case=False)
```
Finetuning scripts and other utilities we use for our projects can be found in our centralized repository at https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks

## Citations
All model details and training setups can be found in our papers. If you use our model or find it useful in your projects, please cite our work:

```
@article{cruz2020investigating,
  title={Investigating the True Performance of Transformers in Low-Resource Languages: A Case Study in Automatic Corpus Creation},
  author={Jan Christian Blaise Cruz and Jose Kristian Resabal and James Lin and Dan John Velasco and Charibeth Cheng},
  journal={arXiv preprint arXiv:2010.11574},
  year={2020}
}
```

## Data and Other Resources
Data used to train this model as well as other benchmark datasets in Filipino can be found in my website at https://blaisecruz.com

## Contact
If you have questions, concerns, or if you just want to chat about NLP and low-resource languages in general, you may reach me through my work email at jan_christian_cruz@dlsu.edu.ph
