---
language: dutch
---

# Multilingual + Dutch SQuAD2.0

This model is the multilingual model provided by the Google research team with a fine-tuned dutch Q&A downstream task.

## Details of the language model(bert-base-multilingual-cased)

Language model ([**bert-base-multilingual-cased**](https://github.com/google-research/bert/blob/master/multilingual.md)):
12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased text in the top 104 languages with the largest Wikipedias.

## Details of the downstream task - Dataset
Using the `mtranslate` Python module, [**SQuAD2.0**](https://rajpurkar.github.io/SQuAD-explorer/) was machine-translated. In order to find the start tokens the direct translations of the answers were searched in the corresponding paragraphs. Since the answer could not always be found in the text, due to the different translations depending on the context (missing context in the pure answer), a loss of question-answer examples occurred. This is a potential problem where errors can occur in the data set (but in the end it was a quick and dirty solution that worked well enough for my task).

| Dataset                | # Q&A |
| ---------------------- | ----- |
| SQuAD2.0 Train         | 130 K |
| Dutch SQuAD2.0 Train   | 99  K |
| SQuAD2.0 Dev           | 12  K |
| Dutch SQuAD2.0 Dev     | 10  K |

## Model training

The model was trained on a Tesla V100 GPU with the following command:

```python
export SQUAD_DIR=path/to/nl_squad

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --version_2_with_negative \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/train_nl-v2.0.json \
  --predict_file $SQUAD_DIR/dev_nl-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/output_dir/
```

**Results**:

{'exact': **67.38**, 'f1': **71.36**} 