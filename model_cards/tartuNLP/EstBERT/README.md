---
language: et
---
# EstBERT


### What's this?
The EstBERT model is a pretrained BERT<sub>Base</sub> model exclusively trained on Estonian cased corpus on both 128 and 512 sequence length of data. 

### How to use?
You can use the model transformer library both in tensorflow and pytorch version. 
```
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("tartuNLP/EstBERT")
model = AutoModelForMaskedLM.from_pretrained("tartuNLP/EstBERT")
```
You can also download the pretrained model from here, [EstBERT_128]() [EstBERT_512]()
#### Dataset used to train the model
The EstBERT model is trained both on 128 and 512 sequence length of data. For training the EstBERT we used the [Estonian National Corpus 2017](https://metashare.ut.ee/repository/browse/estonian-national-corpus-2017/b616ceda30ce11e8a6e4005056b40024880158b577154c01bd3d3fcfc9b762b3/), which was the largest Estonian language corpus available at the time. It consists of four sub-corpora: Estonian Reference Corpus 1990-2008, Estonian Web Corpus 2013, Estonian Web Corpus 2017 and Estonian Wikipedia Corpus 2017.

### Why would I use?
Overall EstBERT performs better in parts of speech (POS), name entity recognition (NER), rubric, and sentiment classification tasks compared to mBERT and XLM-RoBERTa. The comparative results can be found below;

|Model   |UPOS                  |XPOS   |Morph  |bf UPOS   |bf XPOS                  |Morph                 |
|--------------|----------------------------|-------------|-------------|-------------|----------------------------|----------------------------|
| EstBERT      | **_97.89_** | **98.40** | **96.93** | **97.84** | **_98.43_** | **_96.80_** |
| mBERT        | 97.42                     | 98.06      | 96.24      | 97.43      | 98.13                     | 96.13                     |
| XLM-RoBERTa | 97.78                     | 98.36      | 96.53      | 97.80      | 98.40                     | 96.69                     |


|Model|Rubric<sub>128</sub>        |Sentiment<sub>128</sub>  | Rubric<sub>128</sub>   |Sentiment<sub>512</sub>         |
|-------------------|----------------------------|--------------------|-----------------------------------------------|----------------------------|
| EstBERT           | **_81.70_** | 74.36             | **80.96**                                   | 74.50                     |
| mBERT             | 75.67                     | 70.23             | 74.94                                        | 69.52                     |
| XLM\-RoBERTa      | 80.34                     | **74.50**        | 78.62                                        | **_76.07_**|

|Model   |Precicion<sub>128</sub>   |Recall<sub>128</sub>                  |F1-Score<sub>128</sub>               |Precision<sub>512</sub>               |Recall<sub>512</sub>   |F1-Score<sub>512</sub>   |
|--------------|----------------|----------------------------|----------------------------|----------------------------|-------------|----------------|
| EstBERT      | **88.42**    | 90.38                     |**_89.39_** | 88.35                     | 89.74      | 89.04         |
| mBERT        | 85.88         | 87.09                     | 86.51                     |**_88.47_** | 88.28      | 88.37         |
| XLM\-RoBERTa | 87.55         |**_91.19_** | 89.34                     | 87.50                     | **90.76** | **89.10**    |
