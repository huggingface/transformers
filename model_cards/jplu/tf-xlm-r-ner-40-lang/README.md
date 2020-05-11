
# XLM-R + NER

This model is a fine-tuned  [XLM-Roberta-base](https://arxiv.org/abs/1911.02116) over the 40 languages proposed in [XTREME]([https://github.com/google-research/xtreme](https://github.com/google-research/xtreme)) from [Wikiann](https://aclweb.org/anthology/P17-1178). This is still an on-going work and the results will be updated everytime an improvement is reached. 

The covered labels are:
```
LOC
ORG
PER
O
```

## Metrics on evaluation set:
### Average over the 40 languages
Number of documents: 262300
```
           precision    recall  f1-score   support

      ORG       0.81      0.81      0.81    102452
      PER       0.90      0.91      0.91    108978
      LOC       0.86      0.89      0.87    121868

micro avg       0.86      0.87      0.87    333298
macro avg       0.86      0.87      0.87    333298
```

### Afrikaans
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.89      0.88      0.88       582
      PER       0.89      0.97      0.93       369
      LOC       0.84      0.90      0.86       518

micro avg       0.87      0.91      0.89      1469
macro avg       0.87      0.91      0.89      1469
``` 

### Arabic
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.83      0.84      0.84      3507
      PER       0.90      0.91      0.91      3643
      LOC       0.88      0.89      0.88      3604

micro avg       0.87      0.88      0.88     10754
macro avg       0.87      0.88      0.88     10754
```

### Basque
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.88      0.93      0.91      5228
      ORG       0.86      0.81      0.83      3654
      PER       0.91      0.91      0.91      4072

micro avg       0.89      0.89      0.89     12954
macro avg       0.89      0.89      0.89     12954
```

### Bengali
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.86      0.89      0.87       325
      LOC       0.91      0.91      0.91       406
      PER       0.96      0.95      0.95       364

micro avg       0.91      0.92      0.91      1095
macro avg       0.91      0.92      0.91      1095
```

### Bulgarian
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.86      0.83      0.84      3661
      PER       0.92      0.95      0.94      4006
      LOC       0.92      0.95      0.94      6449

micro avg       0.91      0.92      0.91     14116
macro avg       0.91      0.92      0.91     14116
```

### Burmese
Number of documents: 100
```
           precision    recall  f1-score   support

      LOC       0.60      0.86      0.71        37
      ORG       0.68      0.63      0.66        30
      PER       0.44      0.44      0.44        36

micro avg       0.57      0.65      0.61       103
macro avg       0.57      0.65      0.60       103
```

### Chinese
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.70      0.69      0.70      4022
      LOC       0.76      0.81      0.78      3830
      PER       0.84      0.84      0.84      3706

micro avg       0.76      0.78      0.77     11558
macro avg       0.76      0.78      0.77     11558
```

### Dutch
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.87      0.87      0.87      3930
      PER       0.95      0.95      0.95      4377
      LOC       0.91      0.92      0.91      4813

micro avg       0.91      0.92      0.91     13120
macro avg       0.91      0.92      0.91     13120
```

### English
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.83      0.84      0.84      4781
      PER       0.89      0.90      0.89      4559
      ORG       0.75      0.75      0.75      4633

micro avg       0.82      0.83      0.83     13973
macro avg       0.82      0.83      0.83     13973
```

### Estonian
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.89      0.92      0.91      5654
      ORG       0.85      0.85      0.85      3878
      PER       0.94      0.94      0.94      4026

micro avg       0.90      0.91      0.90     13558
macro avg       0.90      0.91      0.90     13558
```

### Finnish
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.84      0.83      0.84      4104
      LOC       0.88      0.90      0.89      5307
      PER       0.95      0.94      0.94      4519

micro avg       0.89      0.89      0.89     13930
macro avg       0.89      0.89      0.89     13930
```

### French
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.90      0.89      0.89      4808
      ORG       0.84      0.87      0.85      3876
      PER       0.94      0.93      0.94      4249

micro avg       0.89      0.90      0.90     12933
macro avg       0.89      0.90      0.90     12933
```

### Georgian
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.90      0.91      0.90      3964
      ORG       0.83      0.77      0.80      3757
      LOC       0.82      0.88      0.85      4894

micro avg       0.84      0.86      0.85     12615
macro avg       0.84      0.86      0.85     12615
```

### German
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.85      0.90      0.87      4939
      PER       0.94      0.91      0.92      4452
      ORG       0.79      0.78      0.79      4247

micro avg       0.86      0.86      0.86     13638
macro avg       0.86      0.86      0.86     13638
```

### Greek
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.86      0.85      0.85      3771
      LOC       0.88      0.91      0.90      4436
      PER       0.91      0.93      0.92      3894

micro avg       0.88      0.90      0.89     12101
macro avg       0.88      0.90      0.89     12101
```

### Hebrew
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.87      0.88      0.87      4206
      ORG       0.76      0.75      0.76      4190
      LOC       0.85      0.85      0.85      4538

micro avg       0.83      0.83      0.83     12934
macro avg       0.82      0.83      0.83     12934
```

### Hindi
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.78      0.81      0.79       362
      LOC       0.83      0.85      0.84       422
      PER       0.90      0.95      0.92       427

micro avg       0.84      0.87      0.85      1211
macro avg       0.84      0.87      0.85      1211
```

### Hungarian
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.95      0.95      0.95      4347
      ORG       0.87      0.88      0.87      3988
      LOC       0.90      0.92      0.91      5544

micro avg       0.91      0.92      0.91     13879
macro avg       0.91      0.92      0.91     13879
```

### Indonesian
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.88      0.89      0.88      3735
      LOC       0.93      0.95      0.94      3694
      PER       0.93      0.93      0.93      3947

micro avg       0.91      0.92      0.92     11376
macro avg       0.91      0.92      0.92     11376
```

### Italian
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.88      0.88      0.88      4592
      ORG       0.86      0.86      0.86      4088
      PER       0.96      0.96      0.96      4732

micro avg       0.90      0.90      0.90     13412
macro avg       0.90      0.90      0.90     13412
```

### Japanese
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.62      0.61      0.62      4184
      PER       0.76      0.81      0.78      3812
      LOC       0.68      0.74      0.71      4281

micro avg       0.69      0.72      0.70     12277
macro avg       0.69      0.72      0.70     12277
```

### Javanese
Number of documents: 100
```
           precision    recall  f1-score   support

      ORG       0.79      0.80      0.80        46
      PER       0.81      0.96      0.88        26
      LOC       0.75      0.75      0.75        40

micro avg       0.78      0.82      0.80       112
macro avg       0.78      0.82      0.80       112
```

### Kazakh
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.76      0.61      0.68       307
      LOC       0.78      0.90      0.84       461
      PER       0.87      0.91      0.89       367

micro avg       0.81      0.83      0.82      1135
macro avg       0.81      0.83      0.81      1135
```

### Korean
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.86      0.89      0.88      5097
      ORG       0.79      0.74      0.77      4218
      PER       0.83      0.86      0.84      4014

micro avg       0.83      0.83      0.83     13329
macro avg       0.83      0.83      0.83     13329
```

### Malay
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.87      0.89      0.88       368
      PER       0.92      0.91      0.91       366
      LOC       0.94      0.95      0.95       354

micro avg       0.91      0.92      0.91      1088
macro avg       0.91      0.92      0.91      1088
```

### Malayalam
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.75      0.74      0.75       347
      PER       0.84      0.89      0.86       417
      LOC       0.74      0.75      0.75       391

micro avg       0.78      0.80      0.79      1155
macro avg       0.78      0.80      0.79      1155
```

### Marathi
Number of documents: 1000
```
           precision    recall  f1-score   support

      PER       0.89      0.94      0.92       394
      LOC       0.82      0.84      0.83       457
      ORG       0.84      0.78      0.81       339

micro avg       0.85      0.86      0.85      1190
macro avg       0.85      0.86      0.85      1190
```

### Persian
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.93      0.92      0.93      3540
      LOC       0.93      0.93      0.93      3584
      ORG       0.89      0.92      0.90      3370

micro avg       0.92      0.92      0.92     10494
macro avg       0.92      0.92      0.92     10494
```

### Portuguese
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.90      0.91      0.91      4819
      PER       0.94      0.92      0.93      4184
      ORG       0.84      0.88      0.86      3670

micro avg       0.89      0.91      0.90     12673
macro avg       0.90      0.91      0.90     12673
```

### Russian
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.93      0.96      0.95      3574
      LOC       0.87      0.89      0.88      4619
      ORG       0.82      0.80      0.81      3858

micro avg       0.87      0.88      0.88     12051
macro avg       0.87      0.88      0.88     12051
```

### Spanish
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.95      0.93      0.94      3891
      ORG       0.86      0.88      0.87      3709
      LOC       0.89      0.91      0.90      4553

micro avg       0.90      0.91      0.90     12153
macro avg       0.90      0.91      0.90     12153
```

### Swahili
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.82      0.85      0.83       349
      PER       0.95      0.92      0.94       403
      LOC       0.86      0.89      0.88       450

micro avg       0.88      0.89      0.88      1202
macro avg       0.88      0.89      0.88      1202
```

### Tagalog
Number of documents: 1000
```
           precision    recall  f1-score   support

      LOC       0.90      0.91      0.90       338
      ORG       0.83      0.91      0.87       339
      PER       0.96      0.93      0.95       350

micro avg       0.90      0.92      0.91      1027
macro avg       0.90      0.92      0.91      1027
```

### Tamil
Number of documents: 1000
```
           precision    recall  f1-score   support

      PER       0.90      0.92      0.91       392
      ORG       0.77      0.76      0.76       370
      LOC       0.78      0.81      0.79       421

micro avg       0.82      0.83      0.82      1183
macro avg       0.82      0.83      0.82      1183
```

### Telugu
Number of documents: 1000
```
           precision    recall  f1-score   support

      ORG       0.67      0.55      0.61       347
      LOC       0.78      0.87      0.82       453
      PER       0.73      0.86      0.79       393

micro avg       0.74      0.77      0.76      1193
macro avg       0.73      0.77      0.75      1193
```

### Thai
Number of documents: 10000
```
           precision    recall  f1-score   support

      LOC       0.63      0.76      0.69      3928
      PER       0.78      0.83      0.80      6537
      ORG       0.59      0.59      0.59      4257

micro avg       0.68      0.74      0.71     14722
macro avg       0.68      0.74      0.71     14722
```

### Turkish
Number of documents: 10000
```
           precision    recall  f1-score   support

      PER       0.94      0.94      0.94      4337
      ORG       0.88      0.89      0.88      4094
      LOC       0.90      0.92      0.91      4929

micro avg       0.90      0.92      0.91     13360
macro avg       0.91      0.92      0.91     13360
```

### Urdu
Number of documents: 1000
```
           precision    recall  f1-score   support

      LOC       0.90      0.95      0.93       352
      PER       0.96      0.96      0.96       333
      ORG       0.91      0.90      0.90       326

micro avg       0.92      0.94      0.93      1011
macro avg       0.92      0.94      0.93      1011
```

### Vietnamese
Number of documents: 10000
```
           precision    recall  f1-score   support

      ORG       0.86      0.87      0.86      3579
      LOC       0.88      0.91      0.90      3811
      PER       0.92      0.93      0.93      3717

micro avg       0.89      0.90      0.90     11107
macro avg       0.89      0.90      0.90     11107
```

### Yoruba
Number of documents: 100
```
           precision    recall  f1-score   support

      LOC       0.54      0.72      0.62        36
      ORG       0.58      0.31      0.41        35
      PER       0.77      1.00      0.87        36

micro avg       0.64      0.68      0.66       107
macro avg       0.63      0.68      0.63       107
```

## Reproduce the results
Download and prepare the dataset from the [XTREME repo](https://github.com/google-research/xtreme#download-the-data). Next, from the root of the transformers repo run:
```
cd examples/ner
python run_tf_ner.py \
--data_dir . \
--labels ./labels.txt \
--model_name_or_path jplu/tf-xlm-roberta-base \
--output_dir model \
--max-seq-length 128 \
--num_train_epochs 2 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 32 \
--do_train \
--do_eval \
--logging_dir logs \
--mode token-classification \
--evaluate_during_training \
--optimizer_name adamw
```

## Usage with pipelines
```python
from transformers import pipeline

nlp_ner = pipeline(
    "ner",
    model="jplu/tf-xlm-r-ner-40-lang",
    tokenizer=(
        'jplu/tf-xlm-r-ner-40-lang',  
        {"use_fast": True}),
    framework="tf"
)

text_fr = "Barack Obama est né à Hawaï."
text_en = "Barack Obama was born in Hawaii."
text_es = "Barack Obama nació en Hawai."
text_zh = "巴拉克·奧巴馬（Barack Obama）出生於夏威夷。"
text_ar = "ولد باراك أوباما في هاواي."

nlp_ner(text_fr)
#Output: [{'word': '▁Barack', 'score': 0.9894659519195557, 'entity': 'PER'}, {'word': '▁Obama', 'score': 0.9888848662376404, 'entity': 'PER'}, {'word': '▁Hawa', 'score': 0.998701810836792, 'entity': 'LOC'}, {'word': 'ï', 'score': 0.9987035989761353, 'entity': 'LOC'}]
nlp_ner(text_en)
#Output: [{'word': '▁Barack', 'score': 0.9929141998291016, 'entity': 'PER'}, {'word': '▁Obama', 'score': 0.9930834174156189, 'entity': 'PER'}, {'word': '▁Hawaii', 'score': 0.9986202120780945, 'entity': 'LOC'}]
nlp_ner(test_es)
#Output: [{'word': '▁Barack', 'score': 0.9944776296615601, 'entity': 'PER'}, {'word': '▁Obama', 'score': 0.9949177503585815, 'entity': 'PER'}, {'word': '▁Hawa', 'score': 0.9987911581993103, 'entity': 'LOC'}, {'word': 'i', 'score': 0.9984861612319946, 'entity': 'LOC'}]
nlp_ner(test_zh)
#Output: [{'word': '夏威夷', 'score': 0.9988449215888977, 'entity': 'LOC'}]
nlp_ner(test_ar)
#Output: [{'word': '▁با', 'score': 0.9903655648231506, 'entity': 'PER'}, {'word': 'راك', 'score': 0.9850614666938782, 'entity': 'PER'}, {'word': '▁أوباما', 'score': 0.9850308299064636, 'entity': 'PER'}, {'word': '▁ها', 'score': 0.9477543234825134, 'entity': 'LOC'}, {'word': 'وا', 'score': 0.9428229928016663, 'entity': 'LOC'}, {'word': 'ي', 'score': 0.9319471716880798, 'entity': 'LOC'}]

```