## Sequence to Sequence

This directory contains examples for finetuning and evaluating transformers on summarization and translation tasks.
Summarization support is more mature than translation support.
Please tag @sshleifer with any issues/unexpected behaviors, or send a PR!
For `bertabs` instructions, see [`bertabs/README.md`](bertabs/README.md).


### Data
XSUM Data:
```bash
cd examples/seq2seq
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
export XSUM_DIR=${PWD}/xsum
```
this should make a directory called `xsum/` with files like `test.source`.
To use your own data, copy that files format. Each article to be summarized is on its own line.

CNN/DailyMail data
```bash
cd examples/seq2seq
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz
tar -xzvf cnn_dm.tgz
export CNN_DIR=${PWD}/cnn_dm
this should make a directory called `cnn_dm/` with files like `test.source`.
```

WMT16 English-Romanian Translation Data:
download with this command:
```bash
wget https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz
export ENRO_DIR=${PWD}/wmt_en_ro
this should make a directory called `wmt_en_ro/` with files like `test.source`.
```

If you are using your own data, it must be formatted as one directory with 6 files: train.source, train.target, val.source, val.target, test.source, test.target.
The `.source` files are the input, the `.target` files are the desired output.


### Tips and Tricks

General Tips:
- since you need to run from `examples/seq2seq`, and likely need to modify code, the easiest workflow is fork transformers, clone your fork, and run `pip install -e .` before you get started.
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
Datasets: `Seq2SeqDataset` should be used for all tokenizers without a `prepare_translation_batch` method. For those who do (like Marian, MBart), `TranslationDataset` should be used.**
A new dataset is needed to support multilingual tasks.


### Command Line Options

To see all the possible command line options, run:

```bash
./finetune.sh --help  # this calls python finetune.py --help
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
    --num_train_epochs 1 \
    --model_name_or_path facebook/bart-large
```

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

### Evaluation Commands

To create summaries for each article in dataset, we use `run_eval.py`, here are a few commands that run eval for different tasks and models.
If 'translation' is in your task name, the computed metric will be BLEU. Otherwise, ROUGE will be used.

For t5, you need to specify --task translation_{src}_to_{tgt} as follows:
```bash
export DATA_DIR=wmt_en_ro
python run_eval.py t5-base \
    $DATA_DIR/val.source t5_val_generations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path enro_bleu.json \
    --task translation_en_to_ro \
    --n_obs 100 \
    --device cuda \
    --fp16 \
    --bs 32
```

This command works for MBART, although the BLEU score is suspiciously low.
```bash
export DATA_DIR=wmt_en_ro
python run_eval.py facebook/mbart-large-en-ro $DATA_DIR/val.source mbart_val_generations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path enro_bleu.json \
    --task translation \
    --n_obs 100 \
    --device cuda \
    --fp16 \
    --bs 32
```

Summarization (xsum will be very similar):
```bash
export DATA_DIR=cnn_dm
python run_eval.py sshleifer/distilbart-cnn-12-6 $DATA_DIR/val.source dbart_val_generations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path cnn_rouge.json \
    --task summarization \
    --n_obs 100 \
    --device cuda \
    --max_source_length 1024 \
    --max_target_length 56 \
    --fp16 \
    --bs 32
```


### DistilBART
![DBART](https://huggingface.co/front/thumbnails/distilbart_large.png)

For the CNN/DailyMail dataset, (relatively longer, more extractive summaries), we found a simple technique that works:
you just copy alternating layers from `bart-large-cnn` and finetune more on the same data.

For the XSUM dataset, that didn’t work as well so we used that same initialization strategy followed by a combination of Distillbert’s ce_loss and the hidden states MSE loss used in the tinybert paper.

You can see the performance tradeoffs of model sizes [here](https://docs.google.com/spreadsheets/d/1EkhDMwVO02m8jCD1cG3RoFPLicpcL1GQHTQjfvDYgIM/edit#gid=0).
and more granular timing results [here](https://docs.google.com/spreadsheets/d/1EkhDMwVO02m8jCD1cG3RoFPLicpcL1GQHTQjfvDYgIM/edit#gid=1753259047&range=B2:I23).

#### No Teacher Distillation
To run the simpler distilbart-cnn style distillation all you need is data, a GPU, and a properly initialized student.
You don't even need `distillation.py`.

Some [un-finetuned students](https://huggingface.co/models?search=sshleifer%2Fstudent) are available for replication purposes.
They are initialized by copying layers from the associated `bart-large-{cnn|xsum}` teacher using `--init_strategy alternate`. (You can read about that in `initialization_utils.py`)
The command that produced `sshleifer/distilbart-cnn-12-6` is
```bash
./train_distilbart_cnn.sh
```
runtime: 6H on NVIDIA RTX 24GB GPU

*Note*: You can get the same simple distillation logic by using `./run_distiller.sh --no_teacher` followed by identical arguments as the ones in `train_distilbart_cnn.sh`.
If you are using `wandb` and comparing the two distillation methods, using this entry point will make your logs consistent,
because you will have the same hyperparameters logged in every run.

#### With a teacher
*Note* only BART variants are supported

In this method, we use try to enforce that the student and teacher produce similar encoder_outputs, logits, and hidden_states using `BartSummarizationDistiller`.
This is how `sshleifer/distilbart-xsum*` checkpoints were produced.

The command that produced `sshleifer/distilbart-xsum-12-6` is:

```bash
./train_distilbart_xsum.sh
```

runtime: 13H on V-100 16GB GPU.

### Contributing
- follow the standard contributing guidelines and code of conduct.
- add tests to `test_seq2seq_examples.py`
- To run only the seq2seq tests, you must be in the root of the repository and run:
```bash
pytest examples/seq2seq/
```
