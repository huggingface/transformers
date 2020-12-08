## Sequence to Sequence Training and Evaluation

This directory contains examples for finetuning and evaluating transformers on summarization and translation tasks.
Please tag @patil-suraj with any issues/unexpected behaviors, or send a PR!
For deprecated `bertabs` instructions, see [`bertabs/README.md`](bertabs/README.md).

### Supported Architectures

- `BartForConditionalGeneration` (and anything that inherits from it)
- `MarianMTModel`
- `PegasusForConditionalGeneration`
- `MBartForConditionalGeneration`
- `FSMTForConditionalGeneration`
- `T5ForConditionalGeneration`

## Datasets

#### XSUM

```bash
cd examples/contrib/pytorch-lightning/seq2seq
wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
export XSUM_DIR=${PWD}/xsum
```
this should make a directory called `xsum/` with files like `test.source`.
To use your own data, copy that files format. Each article to be summarized is on its own line.

#### CNN/DailyMail

```bash
cd examples/contrib/pytorch-lightning/seq2seq
wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz
tar -xzvf cnn_dm_v2.tgz  # empty lines removed
mv cnn_cln cnn_dm
export CNN_DIR=${PWD}/cnn_dm
```
this should make a directory called `cnn_dm/` with 6 files.

#### WMT16 English-Romanian Translation Data

download with this command:
```bash
wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz
export ENRO_DIR=${PWD}/wmt_en_ro
```
this should make a directory called `wmt_en_ro/` with 6 files.

#### WMT English-German

```bash
wget https://cdn-datasets.huggingface.co/translation/wmt_en_de.tgz
tar -xzvf wmt_en_de.tgz
export DATA_DIR=${PWD}/wmt_en_de
```

#### FSMT datasets (wmt)

Refer to the scripts starting with `eval_` under:
https://github.com/huggingface/transformers/tree/master/scripts/fsmt

#### Pegasus (multiple datasets)

Multiple eval datasets are available for download from: 
https://github.com/stas00/porting/tree/master/datasets/pegasus


#### Your Data

If you are using your own data, it must be formatted as one directory with 6 files:
```
train.source
train.target
val.source
val.target
test.source
test.target
```
The `.source` files are the input, the `.target` files are the desired output.

### Potential issues

- native AMP (`--fp16` and no apex) may lead to a huge memory leak and require 10x gpu memory. This has been fixed in pytorch-nightly and the minimal official version to have this fix will be pytorch-1.8. Until then if you have to use mixed precision please use AMP only with pytorch-nightly or NVIDIA's apex. Reference: https://github.com/huggingface/transformers/issues/8403


### Tips and Tricks

General Tips:
- since you need to run from this folder, and likely need to modify code, the easiest workflow is fork transformers, clone your fork, and run `pip install -e .` before you get started.
- try `--freeze_encoder` or `--freeze_embeds` for faster training/larger batch size.  (3hr per epoch with bs=8, see the "xsum_shared_task" command below)
- `fp16_opt_level=O1` (the default works best).
- In addition to the pytorch-lightning .ckpt checkpoint, a transformers checkpoint will be saved.
Load it with `BartForConditionalGeneration.from_pretrained(f'{output_dir}/best_tfmr)`.
- At the moment, `--do_predict` does not work in a multi-gpu setting. You need to use `evaluate_checkpoint` or the `run_eval.py` code.
- This warning can be safely ignored:
    > "Some weights of BartForConditionalGeneration were not initialized from the model checkpoint at facebook/bart-large-xsum and are newly initialized: ['final_logits_bias']"
- Both finetuning and eval are 30% faster with `--fp16`. For that you need to [install apex](https://github.com/NVIDIA/apex#quick-start).
- Read scripts before you run them!

Summarization Tips:
- (summ) 1 epoch at batch size 1 for bart-large takes 24 hours and requires 13GB GPU RAM with fp16 on an NVIDIA-V100.
- If you want to run experiments on improving the summarization finetuning process, try the XSUM Shared Task (below). It's faster to train than CNNDM because the summaries are shorter.
- For CNN/DailyMail, the default `val_max_target_length` and `test_max_target_length` will truncate the ground truth labels, resulting in slightly higher rouge scores. To get accurate rouge scores, you should rerun calculate_rouge on the `{output_dir}/test_generations.txt` file saved by `trainer.test()`
- `--max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 ` is a reasonable setting for XSUM.
- `wandb` can be used by specifying `--logger_name wandb`. It is useful for reproducibility. Specify the environment variable `WANDB_PROJECT='hf_xsum'` to do the XSUM shared task.
- If you are finetuning on your own dataset, start from `distilbart-cnn-12-6` if you want long summaries and `distilbart-xsum-12-6` if you want short summaries.
(It rarely makes sense to start from `bart-large` unless you are a researching finetuning methods).

**Update 2018-07-18**
Datasets: `LegacySeq2SeqDataset` will be used for all tokenizers without a `prepare_seq2seq_batch` method. Otherwise, `Seq2SeqDataset` will be used.
Future work/help wanted: A new dataset to support multilingual tasks.


### Finetuning Scripts
All finetuning bash scripts call finetune.py (or distillation.py) with reasonable command line arguments. They usually require extra command line arguments to work.

To see all the possible command line options, run:

```bash
./finetune.py --help
```

### Finetuning Training Params

To override the pretrained model's training params, you can pass them to `./finetune.sh`:

```bash
./finetune.sh \
    [...]
    --encoder_layerdrop 0.1 \
    --decoder_layerdrop 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
```

### Summarization Finetuning
Run/modify `finetune.sh`

The following command should work on a 16GB GPU:
```bash
./finetune.sh \
    --data_dir $XSUM_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=xsum_results \
    --num_train_epochs 6 \
    --model_name_or_path facebook/bart-large
```

There is a starter finetuning script for pegasus at `finetune_pegasus_xsum.sh`.

### Translation Finetuning

First, follow the wmt_en_ro download instructions.
Then you can finetune mbart_cc25 on english-romanian with the following command.
**Recommendation:** Read and potentially modify the fairly opinionated defaults in `train_mbart_cc25_enro.sh` script before running it.

Best performing command:
```bash
# optionally
export ENRO_DIR='wmt_en_ro' # Download instructions above
# export WANDB_PROJECT="MT" # optional
export MAX_LEN=128
export BS=4
./train_mbart_cc25_enro.sh --output_dir enro_finetune_baseline --label_smoothing 0.1 --fp16_opt_level=O1 --logger_name wandb --sortish_sampler
```
This should take < 6h/epoch on a 16GB v100 and achieve test BLEU above 26
To get results in line with fairseq, you need to do some postprocessing. (see `romanian_postprocessing.md`)

MultiGPU command
(using 8 GPUS as an example)
```bash
export ENRO_DIR='wmt_en_ro' # Download instructions above
 # export WANDB_PROJECT="MT" # optional
export MAX_LEN=128
export BS=4
./train_mbart_cc25_enro.sh --output_dir enro_finetune_baseline --gpus 8 --logger_name wandb
```
### Finetuning Outputs
As you train, `output_dir` will be filled with files, that look kind of like this (comments are mine).
Some of them are metrics, some of them are checkpoints, some of them are metadata. Here is a quick tour:

```bash
output_dir
├── best_tfmr  # this is a huggingface checkpoint generated by save_pretrained. It is the same model as the PL .ckpt file below
│   ├── config.json
│   ├── merges.txt
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── git_log.json   # repo, branch, and commit hash
├── val_avg_rouge2=0.1984-step_count=11.ckpt  # this is a pytorch lightning checkpoint associated with the best val score. (it will be called BLEU for MT)
├── metrics.json  # new validation metrics will continually be appended to this
├── student  # this is a huggingface checkpoint generated by SummarizationDistiller. It is the student before it gets finetuned.
│   ├── config.json
│   └── pytorch_model.bin
├── test_generations.txt
# ^^ are the summaries or translations produced by your best checkpoint on the test data. Populated when training is done
├── test_results.txt  # a convenience file with the test set metrics. This data is also in metrics.json['test']
├── hparams.pkl  # the command line args passed after some light preprocessing. Should be saved fairly quickly.
```
After training, you can recover the best checkpoint by running
```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(f'{output_dir}/best_tfmr')
```

### Converting pytorch-lightning checkpoints
pytorch lightning ``-do_predict`` often fails, after you are done training, the best way to evaluate your model is to convert it.

This should be done for you, with a file called `{save_dir}/best_tfmr`. 

If that file doesn't exist but you have a lightning `.ckpt` file, you can run
```bash
python convert_pl_checkpoint_to_hf.py PATH_TO_CKPT  randomly_initialized_hf_model_path save_dir/best_tfmr
```
Then either `run_eval` or `run_distributed_eval` with `save_dir/best_tfmr` (see previous sections)


# Experimental Features 
These features are harder to use and not always useful.

###  Dynamic Batch Size for MT
`finetune.py` has a command line arg `--max_tokens_per_batch` that allows batches to be dynamically sized.
This feature can only be used:
- with fairseq installed
- on 1 GPU
- without sortish sampler
- after calling `./save_len_file.py $tok $data_dir`

For example, 
```bash
./save_len_file.py Helsinki-NLP/opus-mt-en-ro  wmt_en_ro
./dynamic_bs_example.sh --max_tokens_per_batch=2000 --output_dir benchmark_dynamic_bs
```
splits `wmt_en_ro/train` into 11,197 uneven lengthed batches and can finish 1 epoch in 8 minutes on a v100.

For comparison,
```bash
./dynamic_bs_example.sh --sortish_sampler --train_batch_size 48
```
uses 12,723 batches of length 48 and takes slightly more time 9.5 minutes.

The feature is still experimental, because:
+ we can make it much more robust if we have memory mapped/preprocessed datasets.
+ The speedup over sortish sampler is not that large at the moment.

