This directory contains examples for finetuning and evaluating transformers on summarization and translation tasks.
Summarization support is more mature than translation support.
Please tag @sshleifer with any issues/unexpected behaviors, or send a PR!
For `bertabs` instructions, see `bertabs/README.md`. 

### Data

CNN/DailyMail data
```bash
cd examples/summarization
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz
tar -xzvf cnn_dm.tgz

export CNN_DIR=${PWD}/cnn_dm
```

this should make a directory called cnn_dm/ with files like `test.source`.
To use your own data, copy that files format. Each article to be summarized is on its own line.

XSUM Data:
```bash
cd examples/summarization
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
export XSUM_DIR=${PWD}/xsum
```


WMT16 English-Romanian Translation Data:
```bash
cd examples/summarization
wget https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz
export ENRO_DIR=${PWD}/wmt_en_ro
```

If you are using your own data, it must be formatted as one directory with 6 files: train.source, train.target, val.source, val.target, test.source, test.target
The source files are the input, the target files are the desired output.

### Evaluation

To create summaries for each article in dataset, run:
```bash
python run_eval.py <path_to_test.source> test_generations.txt <model-name>  --score_path rouge_scores.txt
```
The default batch size, 4, fits in 16GB GPU memory, but may need to be adjusted to fit your system.


### Summarization Finetuning
Run/modify `finetune.sh`

The following command should work on a 16GB GPU:
```bash
./finetune.sh \
    --data_dir $XSUM_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=xsum_results \
    --num_train_epochs 1
    --model_name_or_path facebook/bart-large
```

*Note*: The following tips mostly apply to summarization finetuning.

Tips:
- 1 epoch at batch size 1 for bart-large takes 24 hours and requires 13GB GPU RAM with fp16 on an NVIDIA-V100. 
- try `bart-base`, `--freeze_encoder` or `--freeze_embeds` for faster training/larger batch size.  (3hr/epoch with bs=8, see below)
- `fp16_opt_level=O1` (the default works best).
- If you are finetuning on your own dataset, start from `distilbart-cnn-12-6` if you want long summaries and `distilbart-xsum-12-6` if you want short summaries.
(It rarely makes sense to start from `bart-large` unless you are a researching finetuning methods).
- In addition to the pytorch-lightning .ckpt checkpoint, a transformers checkpoint will be saved.
Load it with `BartForConditionalGeneration.from_pretrained(f'{output_dir}/best_tfmr)`.
- At the moment, `--do_predict` does not work in a multi-gpu setting. You need to use `evaluate_checkpoint` or the `run_eval.py` code.
- If you want to run experiments on improving the summarization finetuning process, try the XSUM Shared Task (below). It's faster to train than CNNDM because the summaries are shorter.
- For CNN/DailyMail, the default `val_max_target_length` and `test_max_target_length` will truncate the ground truth labels, resulting in slightly higher rouge scores. To get accurate rouge scores, you should rerun calculate_rouge on the `{output_dir}/test_generations.txt` file saved by `trainer.test()`
- `--max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 ` is a reasonable setting for XSUM.
- `wandb` can be used by specifying `--logger wandb_shared` or `--logger wandb`. It is useful for reproducibility. 

### XSUM Shared Task
Compare XSUM results with others by using `--logger wandb_shared`. This requires `wandb` registration.
Here is an example command
```bash
./finetune.sh \
    --data_dir $XSUM_DIR \
    --output_dir xsum_frozen_embs \
    --model_name_or_path facebook/bart-large \
    --logger wandb_shared \
    --train_batch_size 16 --eval_batch_size 16 --freeze_embeds --freeze_encoder \
    --num_train_epochs 6 
```

Results can be viewed [here](https://app.wandb.ai/sshleifer/hf_summarization/table?workspace=user-)


### Distillation
 
#### No Teacher Distillation
To run the simpler distilbart-cnn style distillation all you need is data, a GPU, and a properly initialized student.
You don't even need `distillation.py`.

Some [un-finetuned students](https://huggingface.co/models?search=sshleifer%2Fstudent) are available for replication purposes.
They are initialized by copying layers from the associated `bart-large-{cnn|xsum}` teacher using `--init_strategy alternate`. (You can read about that in `initialization_utils.py)
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

runtime: 17H on NVIDIA RTX 24GB GPU 

### Contributing
- follow the standard contributing guidelines and code of conduct.
- add tests to `test_seq2seq_examples.py`
- To run only the seq2seq tests, you must be in the root of the repository and run:
```bash
pytest examples/seq2seq/  
```
