---
language: en
thumbnail: 
license: mit
tags:
- question-answering
- mobilebert
datasets:
- squad
metrics:
- squad
widget:
- text: "Which name is also used to describe the Amazon rainforest in English?"
  context: "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
- text: "How many square kilometers of rainforest is covered in the basin?"
  context: "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
---

## MobileBERT fine-tuned on SQuAD v1

[MobileBERT](https://arxiv.org/abs/2004.02984) is a thin version of BERT_LARGE, while equipped with bottleneck structures and a carefully designed balance
between self-attentions and feed-forward networks.

This model was fine-tuned from the HuggingFace checkpoint `google/mobilebert-uncased` on [SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer).

## Details

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD1.1 | train | 90.6K      |
| SQuAD1.1 | eval  | 11.1k     |


### Fine-tuning
- Python: `3.7.5`

- Machine specs: 

  `CPU: Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz`
  
  `Memory: 32 GiB`

  `GPUs: 2 GeForce GTX 1070, each with 8GiB memory`
  
  `GPU driver: 418.87.01, CUDA: 10.1`

- script:

  ```shell
  # after install https://github.com/huggingface/transformers

  cd examples/question-answering
  mkdir -p data

  wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

  wget -O data/dev-v1.1.json  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

  export SQUAD_DIR=`pwd`/data

  python run_squad.py \
    --model_type mobilebert \
    --model_name_or_path google/mobilebert-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --learning_rate 4e-5 \
    --num_train_epochs 5.0 \
    --max_seq_length 320 \
    --doc_stride 128 \
    --warmup_steps 1400 \
    --output_dir $SQUAD_DIR/mobilebert-uncased-warmup-squad_v1 2>&1 | tee train-mobilebert-warmup-squad_v1.log
  ```

It took about 3 hours to finish.

### Results

**Model size**: `95M`

| Metric | # Value   | # Original ([Table 5](https://arxiv.org/pdf/2004.02984.pdf))|
| ------ | --------- | --------- |
| **EM** | **82.6** | **82.9** |
| **F1** | **90.0** | **90.0** |

Note that the above results didn't involve any hyperparameter search.

## Example Usage


```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="csarron/mobilebert-uncased-squad-v1",
    tokenizer="csarron/mobilebert-uncased-squad-v1"
)

predictions = qa_pipeline({
    'context': "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
    'question': "What day was the game played on?"
})

print(predictions)
# output:
# {'score': 0.7754058241844177, 'start': 23, 'end': 39, 'answer': 'February 7, 2016'}
```

> Created by [Qingqing Cao](https://awk.ai/) | [GitHub](https://github.com/csarron) | [Twitter](https://twitter.com/sysnlp) 

> Made with ❤️ in New York.