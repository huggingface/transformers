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


### Evaluation

To create summaries for each article in dataset, run:
```bash
python run_eval.py <path_to_test.source> test_generations.txt <model-name>  --score_path rouge_scores.txt
```
The default batch size, 4, fits in 16GB GPU memory, but may need to be adjusted to fit your system.


### Training
Run/modify `finetune.sh`

The following command should work on a 16GB GPU:
```bash
export me=`git config user.name`
./finetune.sh \
    --data_dir $XSUM_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir="$me"_xsum_results \
    --num_train_epochs 1
```

Tips:
- 1 epoch at batch size 1 for bart-large takes 24 hours, requires 13GB GPU RAM with fp16 on an NVIDIA-V100. 
- try `bart-base`, `--freeze_encoder` or `--freeze_embeds` for faster training/larger batch size.  (3hr/epoch with bs=8, see below)
- `fp16_opt_level=O1` (the default works best).
- If you are finetuning on your own dataset, start from `bart-large-cnn` if you want long summaries and `bart-large-xsum` if you want short summaries.
(It rarely makes sense to start from `bart-large` unless you are a researching finetuning methods).
- In addition to the pytorch-lightning .ckpt checkpoint, a transformers checkpoint will be saved.
Load it with `BartForConditionalGeneration.from_pretrained(f'{output_dir}/best_tfmr)`.
- At the moment, `--do_predict` does not work in a multi-gpu setting. You need to use `evaluate_checkpoint` or the `run_eval.py` code.
- If you want to run experiments on improving the summarization finetuning process, try the XSUM Shared Task (below). It's faster to train than CNNDM because the summaries are shorter.    

### XSUM Shared Task
Compare XSUM results with others by using `--logger wandb_shared`. This requires `wandb` registration.
Here is an example command
```bash
export me=`git config user.name`
./finetune.sh \
    --data_dir $XSUM_DIR \
    --output_dir "$me"_xsum_frozen_embs \
    --logger wandb_shared \
    --train_batch_size 16 --eval_batch_size 16 --freeze_embeds --freeze_encoder \
    --num_train_epochs 6
```

Results can be viewed [here](https://app.wandb.ai/sshleifer/hf_summarization/table?workspace=user-)
