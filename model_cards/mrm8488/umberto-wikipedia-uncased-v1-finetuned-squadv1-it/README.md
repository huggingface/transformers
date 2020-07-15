---
language: it
---

# UmBERTo Wikipedia Uncased +  italian SQuAD v1 📚 🧐 ❓

[UmBERTo-Wikipedia-Uncased](https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1) fine-tuned on [Italian SQUAD v1 dataset](https://github.com/crux82/squad-it) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Model 🧠

[UmBERTo](https://github.com/musixmatchresearch/umberto) is a Roberta-based Language Model trained on large Italian Corpora and uses two innovative approaches: SentencePiece and Whole Word Masking.
UmBERTo-Wikipedia-Uncased Training is trained on a relative small corpus (~7GB) extracted from Wikipedia-ITA.


## Details of the downstream task (Q&A) - Dataset 📚

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) [Rajpurkar et al. 2016] is a large scale dataset for training of question answering systems on factoid questions. It contains more than 100,000 question-answer pairs about passages from 536 articles chosen from various domains of Wikipedia.

**SQuAD-it** is derived from the SQuAD dataset and it is obtained through semi-automatic translation of the SQuAD dataset into Italian. It represents a large-scale dataset for open question answering processes on factoid questions in Italian. The dataset contains more than 60,000 question/answer pairs derived from the original English dataset. 

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python transformers/examples/question-answering/run_squad.py \
  --model_type bert \
  --model_name_or_path 'Musixmatch/umberto-wikipedia-uncased-v1' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file '/content/dataset/SQuAD_it-train.json' \
  --predict_file '/content/dataset/SQuAD_it-test.json' \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/drive/My\ Drive/umberto-uncased-finetuned-squadv1-it \
  --overwrite_output_dir \
  --save_steps 1000
```
With 10 epochs the model overfits the train dataset so I evaluated the different checkpoints created during training (every 1000 steps) and chose the best (In this case the one created at 17000 steps).

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **60.50** |
| **F1** | **72.41** |



```json
{
'exact': 60.50729399395453,
'f1': 72.4141113348361,
'total': 7609,
'HasAns_exact': 60.50729399395453,
'HasAns_f1': 72.4141113348361,
'HasAns_total': 7609,
'best_exact': 60.50729399395453,
'best_exact_thresh': 0.0,
'best_f1': 72.4141113348361,
'best_f1_thresh': 0.0
}
```

## Comparison ⚖️

| Model                                                                                                                            | EM        | F1 score  |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- |
| [DrQA-it trained on SQuAD-it ](https://github.com/crux82/squad-it/blob/master/README.md#evaluating-a-neural-model-over-squad-it) | 56.1      | 65.9      |
| This one                                                                                                                         |60.50      |72.41      |
| [bert-italian-finedtuned-squadv1-it-alfa](https://huggingface.co/mrm8488/bert-italian-finedtuned-squadv1-it-alfa)                |**62.51**  |**74.16**  |                                                                                                                  | **62.51** | **74.16** |


### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline

QnA_pipeline = pipeline('question-answering', model='mrm8488/umberto-wikipedia-uncased-v1-finetuned-squadv1-it')

QnA_pipeline({
    'context': 'Marco Aurelio era un imperatore romano che praticava lo stoicismo come filosofia di vita .',
    'question': 'Quale filosofia seguì Marco Aurelio ?'
})
# Output:
{'answer': 'stoicismo', 'end': 65, 'score': 0.9477770241566028, 'start': 56}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)
> Made with <span style="color: #e25555;">&hearts;</span> in Spain
