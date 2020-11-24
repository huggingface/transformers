

## SQuAD

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py).

#### Fine-tuning BERT on SQuAD1.0

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB. The data for SQuAD can be downloaded with the following links and should be saved in a
$SQUAD_DIR directory.

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

And for SQuAD2.0, you need to download:

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 88.52
exact_match = 81.22
```

#### Distributed training


Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD1.1:

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 93.15
exact_match = 86.91
```

This fine-tuned model is available as a checkpoint under the reference
[`bert-large-uncased-whole-word-masking-finetuned-squad`](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad).

#### Fine-tuning XLNet on SQuAD

This example code fine-tunes XLNet on both SQuAD1.0 and SQuAD2.0 dataset. See above to download the data for SQuAD .

##### Command for SQuAD1.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --save_steps 5000
```

##### Command for SQuAD2.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=2  \
    --per_gpu_train_batch_size=2   \
    --save_steps 5000
```

Larger batch size may improve the performance while costing more memory.

##### Results for SQuAD1.0 with the previously defined hyper-parameters:

```python
{
"exact": 85.45884578997162,
"f1": 92.5974600601065,
"total": 10570,
"HasAns_exact": 85.45884578997162,
"HasAns_f1": 92.59746006010651,
"HasAns_total": 10570
}
```

##### Results for SQuAD2.0 with the previously defined hyper-parameters:

```python
{
"exact": 80.4177545691906,
"f1": 84.07154997729623,
"total": 11873,
"HasAns_exact": 76.73751686909581,
"HasAns_f1": 84.05558584352873,
"HasAns_total": 5928,
"NoAns_exact": 84.0874684608915,
"NoAns_f1": 84.0874684608915,
"NoAns_total": 5945
}
```

#### Fine-tuning BERT on SQuAD1.0 with relative position embeddings

The following examples show how to fine-tune BERT models with different relative position embeddings. The BERT model 
`bert-base-uncased` was pre-trained with default absolute position embeddings. We provide the following pre-trained 
models which were pre-trained on the same training data (BooksCorpus and English Wikipedia) as in the BERT model 
training, but with different relative position embeddings. 

* `zhiheng-huang/bert-base-uncased-embedding-relative-key`, trained from scratch with relative embedding proposed by 
Shaw et al., [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* `zhiheng-huang/bert-base-uncased-embedding-relative-key-query`, trained from scratch with relative embedding method 4 
in Huang et al. [Improve Transformer Models with Better Relative Position Embeddings](https://arxiv.org/abs/2009.13658)
* `zhiheng-huang/bert-large-uncased-whole-word-masking-embedding-relative-key-query`, fine-tuned from model 
`bert-large-uncased-whole-word-masking` with 3 additional epochs with relative embedding method 4 in Huang et al. 
[Improve Transformer Models with Better Relative Position Embeddings](https://arxiv.org/abs/2009.13658)


##### Base models fine-tuning

```bash
export SQUAD_DIR=/path/to/SQUAD
output_dir=relative_squad
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 ./examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path zhiheng-huang/bert-base-uncased-embedding-relative-key-query \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ${output_dir} \
    --per_gpu_eval_batch_size=60 \
    --per_gpu_train_batch_size=6
```
Training with the above command leads to the following results. It boosts the BERT default from f1 score of 88.52 to 90.54.

```bash
'exact': 83.6802270577105, 'f1': 90.54772098174814
```

The change of `max_seq_length` from 512 to 384 in the above command leads to the f1 score of 90.34. Replacing the above 
model `zhiheng-huang/bert-base-uncased-embedding-relative-key-query` with 
`zhiheng-huang/bert-base-uncased-embedding-relative-key` leads to the f1 score of 89.51. The changing of 8 gpus to one 
gpu training leads to the f1 score of 90.71.

##### Large models fine-tuning

```bash
export SQUAD_DIR=/path/to/SQUAD
output_dir=relative_squad
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 ./examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path zhiheng-huang/bert-large-uncased-whole-word-masking-embedding-relative-key-query \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ${output_dir} \
    --per_gpu_eval_batch_size=6 \
    --per_gpu_train_batch_size=2 \
    --gradient_accumulation_steps 3
```
Training with the above command leads to the f1 score of 93.52, which is slightly better than the f1 score of 93.15 for 
`bert-large-uncased-whole-word-masking`.

## SQuAD with the Tensorflow Trainer

```bash
python run_tf_squad.py \
    --model_name_or_path bert-base-uncased \
    --output_dir model \
    --max_seq_length 384 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --do_train \
    --logging_dir logs \    
    --logging_steps 10 \
    --learning_rate 3e-5 \
    --doc_stride 128    
```


For the moment evaluation is not available in the Tensorflow Trainer only the training.
