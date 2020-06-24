---
language: romanian
---

# bert-base-romanian-uncased-v1

The BERT **base**, **uncased** model for Romanian, trained on a 15GB corpus, version ![v1.0](https://img.shields.io/badge/v1.0-21%20Apr%202020-ff6666)

### How to use

```python
from transformers import AutoTokenizer, AutoModel
import torch

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", do_lower_case=True)
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")

# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

### Evaluation

Evaluation is performed on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) UPOS, XPOS and LAS, and on a NER task based on [RONEC](https://github.com/dumitrescustefan/ronec). Details, as well as more in-depth tests not shown here, are given in the dedicated [evaluation page](https://github.com/dumitrescustefan/Romanian-Transformers/tree/master/evaluation/README.md). 

The baseline is the [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) model ``bert-base-multilingual-(un)cased``, as at the time of writing it was the only available BERT model that works on Romanian.

| Model                          |  UPOS |  XPOS  |  NER  |  LAS  |
|--------------------------------|:-----:|:------:|:-----:|:-----:|
| bert-base-multilingual-uncased | 97.65 |  95.72 | 83.91 | 87.65 |
| bert-base-romanian-uncased-v1  | **98.18** |  **96.84** | **85.26** | **89.61** |

### Corpus 

The model is trained on the following corpora (stats in the table below are after cleaning):

| Corpus    	| Lines(M) 	| Words(M) 	| Chars(B) 	| Size(GB) 	|
|-----------	|:--------:	|:--------:	|:--------:	|:--------:	|
| OPUS      	|   55.05  	|  635.04  	|   4.045  	|    3.8   	|
| OSCAR     	|   33.56  	|  1725.82 	|  11.411  	|    11    	|
| Wikipedia 	|   1.54   	|   60.47  	|   0.411  	|    0.4   	|
| **Total**     	|   **90.15**  	|  **2421.33** 	|  **15.867**  	|   **15.2**   	|

#### Acknowledgements

- We'd like to thank [Sampo Pyysalo](https://github.com/spyysalo) from TurkuNLP for helping us out with the compute needed to pretrain the v1.0 BERT models. He's awesome!
