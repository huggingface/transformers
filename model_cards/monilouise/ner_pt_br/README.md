---
language: 
- pt
tags:
- ner
metrics:
- f1
- accuracy
- precision
- recall
---

# RiskData Brazilian Portuguese NER

## Model description

This is a finetunned version from [Neuralmind BERTimbau] (https://github.com/neuralmind-ai/portuguese-bert/blob/master/README.md) for Portuguese language.

For more details, please see, (https://github.com/SecexSaudeTCU/noticias_ner).

## Intended uses & limitations

#### How to use

#### Limitations and bias

- The finetunned model was trained on a corpus with around 180 news articles crawled from Google News.  The original project's purpose was to recognize named entities in news 
related to fraud and corruption, classifying these entities in four classes: PERSON, ORGANIZATION, PUBLIC INSITUITION and LOCAL (PESSOA, ORGANIZAÇÃO, INSTITUIÇÃO PÚBLICA and LOCAL).

## Training data

The training data can be found at (https://github.com/SecexSaudeTCU/noticias_ner/blob/master/dados/labeled_4_labels.jsonl).


## Training procedure


## Eval results

accuracy: 0.98, 
precision: 0.86 
recall: 0.91
f1: 0.88


The score was calculated using this code:

```python
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
```

### BibTeX entry and citation info

For further information about BERTimbau language model:

```bibtex
@inproceedings{souza2020bertimbau,
    author    = {Souza, F{\'a}bio and Nogueira, Rodrigo and Lotufo, Roberto},
    title     = {{BERT}imbau: pretrained {BERT} models for {B}razilian {P}ortuguese},
    booktitle = {9th Brazilian Conference on Intelligent Systems, {BRACIS}, Rio Grande do Sul, Brazil, October 20-23 (to appear)},
    year      = {2020}
}

@article{souza2019portuguese,
    title={Portuguese Named Entity Recognition using BERT-CRF},
    author={Souza, F{\'a}bio and Nogueira, Rodrigo and Lotufo, Roberto},
    journal={arXiv preprint arXiv:1909.10649},
    url={http://arxiv.org/abs/1909.10649},
    year={2019}
}
```
