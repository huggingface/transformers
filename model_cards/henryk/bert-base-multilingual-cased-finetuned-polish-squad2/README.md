---
language: polish
---

# Multilingual + Polish SQuAD2.0

This model is the multilingual model provided by the Google research team with a fine-tuned polish Q&A downstream task.

## Details of the language model

Language model ([**bert-base-multilingual-cased**](https://github.com/google-research/bert/blob/master/multilingual.md)):
12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased text in the top 104 languages with the largest Wikipedias.

## Details of the downstream task
Using the `mtranslate` Python module, [**SQuAD2.0**](https://rajpurkar.github.io/SQuAD-explorer/) was machine-translated. In order to find the start tokens, the direct translations of the answers were searched in the corresponding paragraphs. Due to the different translations depending on the context (missing context in the pure answer), the answer could not always be found in the text, and thus a loss of question-answer examples occurred. This is a potential problem where errors can occur in the data set.

| Dataset                | # Q&A |
| ---------------------- | ----- |
| SQuAD2.0 Train         | 130 K |
| Polish SQuAD2.0 Train   | 83.1 K |
| SQuAD2.0 Dev           |  12 K |
| Polish SQuAD2.0 Dev     | 8.5  K |


## Model benchmark

| Model                | EM/F1 |HasAns (EM/F1) | NoAns |
| ---------------------- | ----- | ----- | ----- |
| [SlavicBERT](https://huggingface.co/DeepPavlov/bert-base-bg-cs-pl-ru-cased)   | 69.35/71.51  | 47.02/54.09 | 79.20 |
| [polBERT](https://huggingface.co/dkleczek/bert-base-polish-uncased-v1)   | 67.33/69.80| 45.73/53.80  | 76.87 |
| [multiBERT](https://huggingface.co/bert-base-multilingual-cased) | **70.76**/**72.92**  |45.00/52.04 | 82.13 |

## Model training

The model was trained on a **Tesla V100** GPU with the following command:

```python
export SQUAD_DIR=path/to/pl_squad

python run_squad.py 
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --train_file $SQUAD_DIR/pl_squadv2_train.json \
  --predict_file $SQUAD_DIR/pl_squadv2_dev.json \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps=8000 \
  --output_dir ../../output \
  --overwrite_cache \
  --overwrite_output_dir
```

**Results**:

{'exact': 70.76671723655035, 'f1': 72.92156947155917, 'total': 8569, 'HasAns_exact': 45.00762195121951, 'HasAns_f1': 52.04456128116991, 'HasAns_total': 2624, 'NoAns_exact': 82.13624894869638, '
NoAns_f1': 82.13624894869638, 'NoAns_total': 5945, 'best_exact': 71.72365503559342, 'best_exact_thresh': 0.0, 'best_f1': 73.62662512059369, 'best_f1_thresh': 0.0}


## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="henryk/bert-base-multilingual-cased-finetuned-polish-squad2",
    tokenizer="henryk/bert-base-multilingual-cased-finetuned-polish-squad2"
)

qa_pipeline({
    'context': "Warszawa jest największym miastem w Polsce pod względem liczby ludności i powierzchni",
    'question': "Jakie jest największe miasto w Polsce?"})

```

# Output:

```json
{
  "score": 0.9986,
  "start": 0, 
  "end": 8,
  "answer": "Warszawa"
}
```

## Contact

Please do not hesitate to contact me via [LinkedIn](https://www.linkedin.com/in/henryk-borzymowski-0755a2167/) if you want to discuss or get access to the Polish version of SQuAD.