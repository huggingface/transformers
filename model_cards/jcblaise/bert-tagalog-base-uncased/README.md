---
language: tl
tags:
- bert
- tagalog
- filipino
license: gpl-3.0
inference: false
---

# BERT Tagalog Base Uncased
Tagalog version of BERT trained on a large preprocessed text corpus scraped and sourced from the internet. This model is part of a larger research project. We open-source the model to allow greater usage within the Filipino NLP community.

## Usage
The model can be loaded and used in both PyTorch and TensorFlow through the HuggingFace Transformers package.

```python
from transformers import TFAutoModel, AutoModel, AutoTokenizer

# TensorFlow
model = TFAutoModel.from_pretrained('jcblaise/bert-tagalog-base-uncased', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-uncased', do_lower_case=True)

# PyTorch
model = AutoModel.from_pretrained('jcblaise/bert-tagalog-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-uncased', do_lower_case=True)
```
Finetuning scripts and other utilities we use for our projects can be found in our centralized repository at https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks

## Citations
All model details and training setups can be found in our papers. If you use our model or find it useful in your projects, please cite our work:

```
@inproceedings{localization2020cruz,
  title={{Localization of Fake News Detection via Multitask Transfer Learning}},
  author={Cruz, Jan Christian Blaise and Tan, Julianne Agatha and Cheng, Charibeth},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={2589--2597},
  year={2020},
  url={https://www.aclweb.org/anthology/2020.lrec-1.315}
}

@article{cruz2020establishing,
  title={Establishing Baselines for Text Classification in Low-Resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:2005.02068},
  year={2020}
}

@article{cruz2019evaluating,
  title={Evaluating Language Model Finetuning Techniques for Low-resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:1907.00409},
  year={2019}
}
```

## Data and Other Resources
Data used to train this model as well as other benchmark datasets in Filipino can be found in my website at https://blaisecruz.com

## Contact
If you have questions, concerns, or if you just want to chat about NLP and low-resource languages in general, you may reach me through my work email at jan_christian_cruz@dlsu.edu.ph
