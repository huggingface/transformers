---
language: polish
---

# Multilingual + Polish SQuAD1.1

This model is the multilingual model provided by the Google research team with a fine-tuned polish Q&A downstream task.

## Details of the language model

Language model ([**bert-base-multilingual-cased**](https://github.com/google-research/bert/blob/master/multilingual.md)):
12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased text in the top 104 languages with the largest Wikipedias.

## Details of the downstream task
Using the `mtranslate` Python module, [**SQuAD1.1**](https://rajpurkar.github.io/SQuAD-explorer/) was machine-translated. In order to find the start tokens, the direct translations of the answers were searched in the corresponding paragraphs. Due to the different translations depending on the context (missing context in the pure answer), the answer could not always be found in the text, and thus a loss of question-answer examples occurred. This is a potential problem where errors can occur in the data set.

| Dataset                | # Q&A |
| ---------------------- | ----- |
| SQuAD1.1 Train         | 87.7 K |
| Polish SQuAD1.1 Train   | 39.5 K |
| SQuAD1.1 Dev           |  10.6 K |
| Polish SQuAD1.1 Dev     |  2.6 K |


## Model benchmark

| Model                | EM | F1 |
| ---------------------- | ----- | ----- |
| [SlavicBERT](https://huggingface.co/DeepPavlov/bert-base-bg-cs-pl-ru-cased)   | **60.89** | 71.68 |
| [polBERT](https://huggingface.co/dkleczek/bert-base-polish-uncased-v1)   | 57.46 | 68.87 |
| [multiBERT](https://huggingface.co/bert-base-multilingual-cased) | 60.67 | **71.89** |
| [xlm](https://huggingface.co/xlm-mlm-100-1280)     | 47.98 | 59.42 |
## Model training

The model was trained on a **Tesla V100** GPU with the following command:

```python
export SQUAD_DIR=path/to/pl_squad

python run_squad.py 
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/pl_squadv1_train_clean.json \
  --predict_file $SQUAD_DIR/pl_squadv1_dev_clean.json \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps=8000 \
  --output_dir ../../output \
  --overwrite_cache \
  --overwrite_output_dir
```

**Results**:

{'exact': 60.670731707317074, 'f1': 71.8952193697293, 'total': 2624, 'HasAns_exact': 60.670731707317074, 'HasAns_f1': 71.8952193697293,
'HasAns_total': 2624, 'best_exact': 60.670731707317074, 'best_exact_thresh': 0.0, 'best_f1': 71.8952193697293, 'best_f1_thresh': 0.0}

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="henryk/bert-base-multilingual-cased-finetuned-polish-squad1",
    tokenizer="henryk/bert-base-multilingual-cased-finetuned-polish-squad1"
)

qa_pipeline({
    'context': "Warszawa jest największym miastem w Polsce pod względem liczby ludności i powierzchni",
    'question': "Jakie jest największe miasto w Polsce?"})

```

# Output:

```json
{
  "score": 0.9988,
  "start": 0, 
  "end": 8,
  "answer": "Warszawa"
}
```

## Contact

Please do not hesitate to contact me via [LinkedIn](https://www.linkedin.com/in/henryk-borzymowski-0755a2167/) if you want to discuss or get access to the Polish version of SQuAD.