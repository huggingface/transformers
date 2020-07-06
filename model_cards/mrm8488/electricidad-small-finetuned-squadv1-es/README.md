---
language: spanish
thumbnail: https://imgur.com/uxAvBfh
---

# Electricidad small + Spanish SQuAD v1 ⚡❓

[Electricidad-small-discriminator](https://huggingface.co/mrm8488/electricidad-small-discriminator) fine-tuned on [Spanish SQUAD v1.1 dataset](https://github.com/ccasimiro88/TranslateAlignRetrieve/tree/master/SQuAD-es-v1.1) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Dataset 📚

[SQuAD-es-v1.1](https://github.com/ccasimiro88/TranslateAlignRetrieve/tree/master/SQuAD-es-v1.1)

| Dataset split | # Samples |
| ------------- | --------- |
| Train         | 130 K     |
| Test          | 11 K      |

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python /content/transformers/examples/question-answering/run_squad.py \
  --model_type electra \
  --model_name_or_path 'mrm8488/electricidad-small-discriminator' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file '/content/dataset/train-v1.1-es.json' \
  --predict_file '/content/dataset/dev-v1.1-es.json' \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir '/content/electricidad-small-finetuned-squadv1-es' \
  --overwrite_output_dir \
  --save_steps 1000
```

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **46.82** |
| **F1** | **64.79** |

```json
{
'exact': 46.82119205298013,
'f1': 64.79435260021918,
'total': 10570,
'HasAns_exact': 46.82119205298013,
HasAns_f1': 64.79435260021918,
'HasAns_total': 10570,
'best_exact': 46.82119205298013,
'best_exact_thresh': 0.0,
'best_f1': 64.79435260021918,
'best_f1_thresh': 0.0
}
```

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/electricidad-small-finetuned-squadv1-es",
    tokenizer="mrm8488/electricidad-small-finetuned-squadv1-es"
)

context = "Manuel ha creado una versión del modelo Electra small en español que alcanza una puntuación F1 de 65 en el dataset SQUAD-es y sólo pesa 50 MB"

q1 = "Cuál es su marcador F1?"
q2 = "¿Cuál es el tamaño del modelo?"
q3 = "¿Quién lo ha creado?"
q4 = "¿Que es lo que ha hecho Manuel?"


questions = [q1, q2, q3, q4]

for question in questions:
  result = qa_pipeline({
    'context': context,
    'question': question})
  print(result)

# Output:
{'score': 0.14836778166355025, 'start': 98, 'end': 100, 'answer': '65'}
{'score': 0.32219420810758237, 'start': 136, 'end': 140, 'answer': '50 MB'}
{'score': 0.9672326951118713, 'start': 0, 'end': 6, 'answer': 'Manuel'}
{'score': 0.23552458113848118, 'start': 10, 'end': 53, 'answer': 'creado una versión del modelo Electra small'}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
