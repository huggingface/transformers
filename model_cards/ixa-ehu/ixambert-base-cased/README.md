---
language: 
- en
- es
- eu
---

# IXAmBERT base cased

This is a multilingual language pretrained for English, Spanish and Basque. The training corpora is composed by the English, Spanish and Basque Wikipedias, together with Basque crawled news articles from online newspapers. The model has been successfully used to transfer knowledge from English to Basque in a conversational QA system, as reported in the paper [Conversational Question Answering in Low Resource Scenarios: A Dataset and Case Study for Basque](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.55.pdf). In the paper, IXAmBERT performed better than mBERT when transferring knowledge from English to Basque, as shown in the following Table:

| Model              | Zero-shot | Transfer learning |
|--------------------|-----------|-------------------|
| Baseline           |      28.7 |              28.7 |
| mBERT              |      31.5 |              37.4 |
| IXAmBERT           |      38.9 |          **41.2** |
| mBERT + history    |      33.3 |              28.7 |
| IXAmBERT + history |  **40.7** |              40.0 |

This Table shows the results on a Basque CQA dataset. *Zero-shot* means that the model is fine-tuned using using QuaC, an English CQA dataset. In the *Transfer Learning* setting the model is first fine-tuned on QuaC, and then on a Basque CQA dataset. 

If using this model, please cite the following paper:
```
@inproceedings{otegi2020conversational,
  title={Conversational Question Answering in Low Resource Scenarios: A Dataset and Case Study for Basque},
  author={Otegi, Arantxa and Agirre, Aitor and Campos, Jon Ander and Soroa, Aitor and Agirre, Eneko},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={436--442},
  year={2020}
}
```
