# ALECTRA-small-OWT

This is an extension of
[ELECTRA](https://openreview.net/forum?id=r1xMH1BtvB) small model, trained on the
[OpenWebText corpus](https://skylion007.github.io/OpenWebTextCorpus/).
The training task (discriminative LM / replaced-token-detection) can be generalized to any transformer type.  Here, we train an ALBERT model under the same scheme.

## Pretraining task
![electra task diagram](https://github.com/shoarora/lmtuners/raw/master/assets/electra.png)
(figure from [Clark et al. 2020](https://openreview.net/pdf?id=r1xMH1BtvB))

ELECTRA uses discriminative LM / replaced-token-detection for pretraining.
This involves a generator (a Masked LM model) creating examples for a discriminator
to classify as original or replaced for each token.

The generator generalizes to any `*ForMaskedLM` model and the discriminator could be
any `*ForTokenClassification` model.  Therefore, we can extend the task to ALBERT models,
not just BERT as in the original paper.

## Usage
```python
from transformers import AlbertForSequenceClassification, BertTokenizer

# Both models use the bert-base-uncased tokenizer and vocab.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
alectra = AlbertForSequenceClassification.from_pretrained('shoarora/alectra-small-owt')
```
NOTE: this ALBERT model uses a BERT WordPiece tokenizer.

## Code
The pytorch module that implements this task is available [here](https://github.com/shoarora/lmtuners/blob/master/lmtuners/lightning_modules/discriminative_lm.py).

Further implementation information [here](https://github.com/shoarora/lmtuners/tree/master/experiments/disc_lm_small),
and [here](https://github.com/shoarora/lmtuners/blob/master/experiments/disc_lm_small/train_alectra_small.py) is the script that created this model.

This specific model was trained with the following params:
- `batch_size: 512`
- `training_steps: 5e5`
- `warmup_steps: 4e4`
- `learning_rate: 2e-3`


## Downstream tasks
#### GLUE Dev results
| Model                    | # Params | CoLA | SST | MRPC | STS  | QQP  | MNLI | QNLI | RTE |
| ---                      | ---      | ---  | --- | ---  | ---  | ---  | ---  | ---  | --- |
| ELECTRA-Small++          | 14M      | 57.0 | 91. | 88.0 | 87.5 | 89.0 | 81.3 | 88.4 | 66.7|
| ELECTRA-Small-OWT        | 14M      | 56.8 | 88.3| 87.4 | 86.8 | 88.3 | 78.9 | 87.9 | 68.5|
| ELECTRA-Small-OWT (ours) | 17M      | 56.3 | 88.4| 75.0 | 86.1 | 89.1 | 77.9 | 83.0 | 67.1|
| ALECTRA-Small-OWT (ours) |  4M      | 50.6 | 89.1| 86.3 | 87.2 | 89.1 | 78.2 | 85.9 | 69.6|


#### GLUE Test results
| Model                    | # Params | CoLA | SST | MRPC | STS  | QQP  | MNLI | QNLI | RTE |
| ---                      | ---      | ---  | --- | ---  | ---  | ---  | ---  | ---  | --- |
| BERT-Base                | 110M     | 52.1 | 93.5| 84.8 | 85.9 | 89.2 | 84.6 | 90.5 | 66.4|
| GPT                      | 117M     | 45.4 | 91.3| 75.7 | 80.0 | 88.5 | 82.1 | 88.1 | 56.0|
| ELECTRA-Small++          | 14M      | 57.0 | 91.2| 88.0 | 87.5 | 89.0 | 81.3 | 88.4 | 66.7|
| ELECTRA-Small-OWT (ours) | 17M      | 57.4 | 89.3| 76.2 | 81.9 | 87.5 | 78.1 | 82.4 | 68.1|
| ALECTRA-Small-OWT (ours) |  4M      | 43.9 | 87.9| 82.1 | 82.0 | 87.6 | 77.9 | 85.8 | 67.5|
