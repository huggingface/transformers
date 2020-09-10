# BERT-Small fine-tuned on CORD-19 QA dataset

[bert-small-cord19-squad model](https://huggingface.co/NeuML/bert-small-cord19-squad2) fine-tuned on the [CORD-19 QA dataset](https://www.kaggle.com/davidmezzetti/cord19-qa?select=cord19-qa.json).

## CORD-19 QA dataset
The CORD-19 QA dataset is a SQuAD 2.0 formatted list of question, context, answer combinations covering the [CORD-19 dataset](https://www.semanticscholar.org/cord19).

## Building the model

```bash
python run_squad.py \
    --model_type bert \
    --model_name_or_path bert-small-cord19-squad \
    --do_train \
    --do_lower_case \
    --version_2_with_negative \
    --train_file cord19-qa.json \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir bert-small-cord19qa \
    --save_steps 0 \
    --threads 8 \
    --overwrite_cache \
    --overwrite_output_dir
```

## Testing the model

Example usage below:

```python
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="NeuML/bert-small-cord19qa",
    tokenizer="NeuML/bert-small-cord19qa"
)

qa({
    "question": "What is the median incubation period?",
    "context": "The incubation period is around 5 days (range: 4-7 days) with a maximum of 12-13 day"
})

qa({
    "question": "What is the incubation period range?",
    "context": "The incubation period is around 5 days (range: 4-7 days) with a maximum of 12-13 day"
})

qa({
    "question": "What type of surfaces does it persist?",
    "context": "The virus can survive on surfaces for up to 72 hours such as plastic and stainless steel ."
})
```

```json
{"score": 0.5970273583242793, "start": 32, "end": 38, "answer": "5 days"}
{"score": 0.999555868193891, "start": 39, "end": 56, "answer": "(range: 4-7 days)"}
{"score": 0.9992726505196998, "start": 61, "end": 88, "answer": "plastic and stainless steel"}
```
