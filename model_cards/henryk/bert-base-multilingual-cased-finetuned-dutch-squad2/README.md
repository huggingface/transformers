---
language: dutch
---

# Multilingual + Dutch SQuAD2.0

This model is the multilingual model provided by the Google research team with a fine-tuned dutch Q&A downstream task.

## Details of the language model

Language model ([**bert-base-multilingual-cased**](https://github.com/google-research/bert/blob/master/multilingual.md)):
12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased text in the top 104 languages with the largest Wikipedias.

## Details of the downstream task
Using the `mtranslate` Python module, [**SQuAD2.0**](https://rajpurkar.github.io/SQuAD-explorer/) was machine-translated. In order to find the start tokens, the direct translations of the answers were searched in the corresponding paragraphs. Due to the different translations depending on the context (missing context in the pure answer), the answer could not always be found in the text, and thus a loss of question-answer examples occurred. This is a potential problem where errors can occur in the data set.

| Dataset                | # Q&A |
| ---------------------- | ----- |
| SQuAD2.0 Train         | 130 K |
| Dutch SQuAD2.0 Train   | 99  K |
| SQuAD2.0 Dev           | 12  K |
| Dutch SQuAD2.0 Dev     | 10  K |


## Model benchmark


| Model                | EM/F1 |HasAns (EM/F1) | NoAns |
| ---------------------- | ----- | ----- | ----- |
| [robBERT](https://huggingface.co/pdelobelle/robBERT-base)   | 58.04/60.95  | 33.08/40.64 | 73.67 |
| [dutchBERT](https://huggingface.co/wietsedv/bert-base-dutch-cased)   | 64.25/68.45 | 45.59/56.49  | 75.94 |
| [multiBERT](https://huggingface.co/bert-base-multilingual-cased) | **67.38**/**71.36**  | 47.42/57.76 | 79.88 |

## Model training

The model was trained on a **Tesla V100** GPU with the following command:

```python
export SQUAD_DIR=path/to/nl_squad

python run_squad.py 
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/nl_squadv2_train_clean.json \
  --predict_file $SQUAD_DIR/nl_squadv2_dev_clean.json \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps=8000 \
  --output_dir ../../output \
  --overwrite_cache \
  --overwrite_output_dir
```

**Results**:

{'exact': 67.38028751680629, 'f1': 71.362297054268, 'total': 9669, 'HasAns_exact': 47.422126745435015, 'HasAns_f1': 57.761023151910734, 'HasAns_total': 3724, 'NoAns_exact': 79.88225399495374, 'NoAns_f1': 79.88225399495374, 'NoAns_total': 5945, 'best_exact': 67.53542248422795, 'best_exact_thresh': 0.0, 'best_f1': 71.36229705426837, 'best_f1_thresh': 0.0}

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2",
    tokenizer="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2"
)

qa_pipeline({
    'context': "Amsterdam is de hoofdstad en de dichtstbevolkte stad van Nederland.",
    'question': "Wat is de hoofdstad van Nederland?"})

```

# Output:

```json
{
  "score": 0.83,
  "start": 0, 
  "end": 9,
  "answer": "Amsterdam"
}
```

## Contact

Please do not hesitate to contact me via [LinkedIn](https://www.linkedin.com/in/henryk-borzymowski-0755a2167/) if you want to discuss or get access to the Dutch version of SQuAD.