---
language:
- en
- ar
datasets:
- gigaword
- oscar
- wikipedia
---

## GigaBERT-v3
GigaBERT-v3 is a customized bilingual BERT for English and Arabic. It was pre-trained in a large-scale corpus (Gigaword+Oscar+Wikipedia) with ~10B tokens, showing state-of-the-art zero-shot transfer performance from English to Arabic on information extraction (IE) tasks. More details can be found in the following paper:

	@inproceedings{lan2020gigabert,
	  author     = {Lan, Wuwei and Chen, Yang and Xu, Wei and Ritter, Alan},
  	  title      = {GigaBERT: Zero-shot Transfer Learning from English to Arabic},
  	  booktitle  = {Proceedings of The 2020 Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  	  year       = {2020}
  	} 

## Usage
```
from transformers import *
tokenizer = BertTokenizer.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)
model = BertForTokenClassification.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English")
```
More code examples can be found [here](https://github.com/lanwuwei/GigaBERT).
