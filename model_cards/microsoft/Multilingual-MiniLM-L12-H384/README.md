---
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
tags:
- text-classification
license: mit
---

## MiniLM: Small and Fast Pre-trained Models for Language Understanding and Generation

MiniLM is a distilled model from the paper "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)".

Please find the information about preprocessing, training and full details of the MiniLM in the [original MiniLM repository](https://github.com/microsoft/unilm/blob/master/minilm/).

Please note: This checkpoint uses `BertModel` with `XLMRobertaTokenizer` so `AutoTokenizer` won't work with this checkpoint!

### Multilingual Pretrained Model
- Multilingual-MiniLMv1-L12-H384: 12-layer, 384-hidden, 12-heads, 21M Transformer parameters, 96M embedding parameters

Multilingual MiniLM uses the same tokenizer as XLM-R. But the Transformer architecture of our model is the same as BERT. We provide the fine-tuning code on XNLI based on [huggingface/transformers](https://github.com/huggingface/transformers). Please replace `run_xnli.py` in transformers with [ours](https://github.com/microsoft/unilm/blob/master/minilm/examples/run_xnli.py) to fine-tune multilingual MiniLM.  

We evaluate the multilingual MiniLM on cross-lingual natural language inference benchmark (XNLI) and cross-lingual question answering benchmark (MLQA).

#### Cross-Lingual Natural Language Inference - [XNLI](https://arxiv.org/abs/1809.05053)

We evaluate our model on cross-lingual transfer from English to other languages. Following [Conneau et al. (2019)](https://arxiv.org/abs/1911.02116), we select the best single model on the joint dev set of all the languages.

| Model                                                                                       | #Layers | #Hidden | #Transformer Parameters | Average | en   | fr   | es   | de   | el   | bg   | ru   | tr   | ar   | vi   | th   | zh   | hi   | sw   | ur   |
|---------------------------------------------------------------------------------------------|---------|---------|-------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| [mBERT](https://github.com/google-research/bert)                                            | 12      | 768     | 85M                     | 66.3    | 82.1 | 73.8 | 74.3 | 71.1 | 66.4 | 68.9 | 69.0 | 61.6 | 64.9 | 69.5 | 55.8 | 69.3 | 60.0 | 50.4 | 58.0 |
| [XLM-100](https://github.com/facebookresearch/XLM#pretrained-cross-lingual-language-models) | 16      | 1280    | 315M                    | 70.7    | 83.2 | 76.7 | 77.7 | 74.0 | 72.7 | 74.1 | 72.7 | 68.7 | 68.6 | 72.9 | 68.9 | 72.5 | 65.6 | 58.2 | 62.4 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116)                                              | 12      | 768     | 85M                     | 74.5    | 84.6 | 78.4 | 78.9 | 76.8 | 75.9 | 77.3 | 75.4 | 73.2 | 71.5 | 75.4 | 72.5 | 74.9 | 71.1 | 65.2 | 66.5 |
| **mMiniLM-L12xH384**                                                                        | 12      | 384     | 21M                     | 71.1    | 81.5 | 74.8 | 75.7 | 72.9 | 73.0 | 74.5 | 71.3 | 69.7 | 68.8 | 72.1 | 67.8 | 70.0 | 66.2 | 63.3 | 64.2 |

This example code fine-tunes **12**-layer multilingual MiniLM on XNLI.

```bash
# run fine-tuning on XNLI
DATA_DIR=/{path_of_data}/
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_PATH=/{path_of_pre-trained_model}/

python ./examples/run_xnli.py --model_type minilm \
 --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR} \
 --model_name_or_path microsoft/Multilingual-MiniLM-L12-H384 \
 --tokenizer_name xlm-roberta-base \
 --config_name ${MODEL_PATH}/multilingual-minilm-l12-h384-config.json \
 --do_train \
 --do_eval \
 --max_seq_length 128 \
 --per_gpu_train_batch_size 128 \
 --learning_rate 5e-5 \
 --num_train_epochs 5 \
 --per_gpu_eval_batch_size 32 \
 --weight_decay 0.001 \
 --warmup_steps 500 \
 --save_steps 1500 \
 --logging_steps 1500 \
 --eval_all_checkpoints \
 --language en \
 --fp16 \
 --fp16_opt_level O2
```

#### Cross-Lingual Question Answering - [MLQA](https://arxiv.org/abs/1910.07475)

Following [Lewis et al. (2019b)](https://arxiv.org/abs/1910.07475), we adopt SQuAD 1.1 as training data and use MLQA English development data for early stopping.

| Model F1 Score                                                                             | #Layers | #Hidden | #Transformer Parameters | Average | en   | es   | de   | ar   | hi   | vi   | zh   |
|--------------------------------------------------------------------------------------------|---------|---------|-------------------------|---------|------|------|------|------|------|------|------|
| [mBERT](https://github.com/google-research/bert)                                           | 12      | 768     | 85M                     | 57.7    | 77.7 | 64.3 | 57.9 | 45.7 | 43.8 | 57.1 | 57.5 |
| [XLM-15](https://github.com/facebookresearch/XLM#pretrained-cross-lingual-language-models) | 12      | 1024    | 151M                    | 61.6    | 74.9 | 68.0 | 62.2 | 54.8 | 48.8 | 61.4 | 61.1 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116) (Reported)                                  | 12      | 768     | 85M                     | 62.9    | 77.8 | 67.2 | 60.8 | 53.0 | 57.9 | 63.1 | 60.2 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116) (Our fine-tuned)                            | 12      | 768     | 85M                     | 64.9    | 80.3 | 67.0 | 62.7 | 55.0 | 60.4 | 66.5 | 62.3 |
| **mMiniLM-L12xH384**                                                                       | 12      | 384     | 21M                     | 63.2    | 79.4 | 66.1 | 61.2 | 54.9 | 58.5 | 63.1 | 59.0 |

### Citation

If you find MiniLM useful in your research, please cite the following paper:

``` latex
@misc{wang2020minilm,
    title={MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
    author={Wenhui Wang and Furu Wei and Li Dong and Hangbo Bao and Nan Yang and Ming Zhou},
    year={2020},
    eprint={2002.10957},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
