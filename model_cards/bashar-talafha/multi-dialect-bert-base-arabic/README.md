---
language: ar
thumbnail: https://raw.githubusercontent.com/mawdoo3/Multi-dialect-Arabic-BERT/master/multidialct_arabic_bert.png
datasets:
- nadi
---
# Multi-dialect-Arabic-BERT
This is a repository of Multi-dialect Arabic BERT model.

By [Mawdoo3-AI](https://ai.mawdoo3.com/). 

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/mawdoo3/Multi-dialect-Arabic-BERT/master/multidialct_arabic_bert.png" alt="Background reference: http://www.qfi.org/wp-content/uploads/2018/02/Qfi_Infographic_Mother-Language_Final.pdf" width="500"/>
    <br>
<p>



### About our Multi-dialect-Arabic-BERT model
Instead of training the Multi-dialect Arabic BERT model from scratch, we initialized the weights of the model using [Arabic-BERT](https://github.com/alisafaya/Arabic-BERT) and trained it on 10M arabic tweets from the unlabled data of [The Nuanced Arabic Dialect Identification (NADI) shared task](https://sites.google.com/view/nadi-shared-task).

### To cite this work

```
@misc{talafha2020multidialect,
    title={Multi-Dialect Arabic BERT for Country-Level Dialect Identification},
    author={Bashar Talafha and Mohammad Ali and Muhy Eddin Za'ter and Haitham Seelawi and Ibraheem Tuffaha and Mostafa Samir and Wael Farhan and Hussein T. Al-Natsheh},
    year={2020},
    eprint={2007.05612},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Usage
The model weights can be loaded using `transformers` library by HuggingFace.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")
model = AutoModel.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")
```

Example using `pipeline`:

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="bashar-talafha/multi-dialect-bert-base-arabic ",
    tokenizer="bashar-talafha/multi-dialect-bert-base-arabic "
)

fill_mask(" سافر الرحالة من مطار [MASK] ")
```
```
[{'sequence': '[CLS] سافر الرحالة من مطار الكويت [SEP]', 'score': 0.08296813815832138, 'token': 3226},
 {'sequence': '[CLS] سافر الرحالة من مطار دبي [SEP]', 'score': 0.05123933032155037, 'token': 4747},
 {'sequence': '[CLS] سافر الرحالة من مطار مسقط [SEP]', 'score': 0.046838656067848206, 'token': 13205},
 {'sequence': '[CLS] سافر الرحالة من مطار القاهرة [SEP]', 'score': 0.03234650194644928, 'token': 4003},
 {'sequence': '[CLS] سافر الرحالة من مطار الرياض [SEP]', 'score': 0.02606341242790222, 'token': 2200}]
```
### Repository
Please check the [original repository](https://github.com/mawdoo3/Multi-dialect-Arabic-BERT) for more information. 


