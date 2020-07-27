---
language: fr
---

# camembert-base-squadFR-fquad-piaf

## Description

Question-answering French model, using base [CamemBERT](https://camembert-model.fr/) fine-tuned on a combo of three French Q&A datasets:

1. [PIAFv1.1](https://www.data.gouv.fr/en/datasets/piaf-le-dataset-francophone-de-questions-reponses/)
2. [FQuADv1.0](https://fquad.illuin.tech/)
3. [SQuAD-FR (SQuAD automatically translated to French)](https://github.com/Alikabbadj/French-SQuAD)

## Training hyperparameters

```shell
python run_squad.py \
--model_type camembert \
--model_name_or_path camembert-base \
--do_train --do_eval \
--train_file data/SQuAD+fquad+piaf.json \
--predict_file data/fquad_valid.json \
--per_gpu_train_batch_size 12 \ 
--learning_rate 3e-5 \ 
--num_train_epochs 4 \  
--max_seq_length 384 \ 
--doc_stride 128 \
--save_steps 10000 
``` 

## Evaluation results
### Fquad v1.0 Evaluation
```shell
{"f1": 79.81, "exact_match": 55.14}
```
### SQuAD-FR Evaluation
```shell
{"f1": 59.54, "exact_match": 80.61}
```

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
```

## Citation

### PIAF
```
@inproceedings{KeraronLBAMSSS20,
  author    = {Rachel Keraron and
               Guillaume Lancrenon and
               Mathilde Bras and
               Fr{\'{e}}d{\'{e}}ric Allary and
               Gilles Moyse and
               Thomas Scialom and
               Edmundo{-}Pavel Soriano{-}Morales and
               Jacopo Staiano},
  title     = {Project {PIAF:} Building a Native French Question-Answering Dataset},
  booktitle = {{LREC}},
  pages     = {5481--5490},
  publisher = {European Language Resources Association},
  year      = {2020}
}

```

### Fquad
```
@article{dHoffschmidt2020FQuADFQ,
  title={FQuAD: French Question Answering Dataset},
  author={Martin d'Hoffschmidt and Maxime Vidal and Wacim Belblidia and Tom Brendl'e and Quentin Heinrich},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.06071}
}
```

### SQuAD-FR
```
 @MISC{maldives,
   author =       "Kabbadj, Ali",
   title =        "Something new in French Text Mining and Information Extraction (Universal Chatbot): Largest Q&A French training dataset (110 000+) ",
   editor =       "linkedin.com",
   month =        "November",
   year =         "2018",
   url =          "\url{https://www.linkedin.com/pulse/something-new-french-text-mining-information-chatbot-largest-kabbadj/}",
   note =         "[Online; posted 11-November-2018]",
 }
 ```
