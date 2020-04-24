---
language: chinese
---

# Introduction
This model was trained on TPU and the details are as follows:

## Model 
## 

| Model_name                                    | params | size | Training_corpus               |    Vocab |    
| :------------------------------------------ | :----- | :------- | :----------------- | :-----------: | 
| **`RoBERTa-tiny-clue`** <br/>Super_small_model       | 7.5M   | 28.3M    | **CLUECorpus2020** | **CLUEVocab** |
| **`RoBERTa-tiny-pair`** <br/>Super_small_sentence_pair_model | 7.5M   | 28.3M    | **CLUECorpus2020** | **CLUEVocab** | 
| **`RoBERTa-tiny3L768-clue`** <br/>small_model    | 38M    | 110M     | **CLUECorpus2020** | **CLUEVocab** | 
| **`RoBERTa-tiny3L312-clue`** <br/>small_model    | <7.5M  | 24M      | **CLUECorpus2020** | **CLUEVocab** | 
| **`RoBERTa-large-clue`** <br/> Large_model       | 290M   | 1.20G    | **CLUECorpus2020** | **CLUEVocab** | 
| **`RoBERTa-large-pair`** <br/>Large_sentence_pair_model  | 290M   | 1.20G    | **CLUECorpus2020** | **CLUEVocab** | 

### Usage

With the help of[Huggingface-Transformers 2.5.1](https://github.com/huggingface/transformers), you could use these model as follows

```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

`MODEL_NAME`：

| Model_NAME                 | MODEL_LINK                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **RoBERTa-tiny-clue**      | [`clue/roberta_chinese_clue_tiny`](https://huggingface.co/clue/roberta_chinese_clue_tiny) |
| **RoBERTa-tiny-pair**      | [`clue/roberta_chinese_pair_tiny`](https://huggingface.co/clue/roberta_chinese_pair_tiny) |
| **RoBERTa-tiny3L768-clue** | [`clue/roberta_chinese_3L768_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L768_clue_tiny) |
| **RoBERTa-tiny3L312-clue** | [`clue/roberta_chinese_3L312_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny) |
| **RoBERTa-large-clue**     | [`clue/roberta_chinese_clue_large`](https://huggingface.co/clue/roberta_chinese_clue_large) |
| **RoBERTa-large-pair**     | [`clue/roberta_chinese_pair_large`](https://huggingface.co/clue/roberta_chinese_pair_large) |

## Details
Please read <a href='https://arxiv.org/pdf/2003.01355'>https://arxiv.org/pdf/2003.01355.

Please visit our repository: https://github.com/CLUEbenchmark/CLUEPretrainedModels.git
