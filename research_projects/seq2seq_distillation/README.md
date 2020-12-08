# DistilBART
<!---It should be called distilling bart and pegasus, but I don't want to break the link in the paper.-->
This section describes all code and artifacts from our [Paper](http://arxiv.org/abs/2010.13002)

![DBART](https://huggingface.co/front/thumbnails/distilbart_large.png)

+ For the CNN/DailyMail dataset, (relatively longer, more extractive summaries), we found a simple technique that works, which we call "Shrink and Fine-tune", or SFT.
you just copy alternating layers from `facebook/bart-large-cnn` and fine-tune more on the cnn/dm data. `sshleifer/distill-pegasus-cnn-16-4`, `sshleifer/distilbart-cnn-12-6` and all other checkpoints under `sshleifer` that start with `distilbart-cnn` were trained this way. 
+ For the XSUM dataset, training on pseudo-labels worked best for Pegasus (`sshleifer/distill-pegasus-16-4`), while training with KD worked best for `distilbart-xsum-12-6`
+ For `sshleifer/dbart-xsum-12-3`
+ We ran 100s experiments, and didn't want to document 100s of commands. If you want a command to replicate a figure from the paper that is not documented below, feel free to ask on the [forums](https://discuss.huggingface.co/t/seq2seq-distillation-methodology-questions/1270) and tag `@sshleifer`. 
+ You can see the performance tradeoffs of model sizes [here](https://docs.google.com/spreadsheets/d/1EkhDMwVO02m8jCD1cG3RoFPLicpcL1GQHTQjfvDYgIM/edit#gid=0).
and more granular timing results [here](https://docs.google.com/spreadsheets/d/1EkhDMwVO02m8jCD1cG3RoFPLicpcL1GQHTQjfvDYgIM/edit#gid=1753259047&range=B2:I23).

### Evaluation

use [run_distributed_eval](./run_distributed_eval.py), with the following convenient alias
```bash
deval () {
	proc=$1
	m=$2
	dd=$3
	sd=$4
	shift
	shift
	shift
	shift
	python -m torch.distributed.launch --nproc_per_node=$proc  run_distributed_eval.py \
		--model_name $m  --save_dir $sd --data_dir $dd $@
}
```
On a 1 GPU system, here are four commands (that assume `xsum`, `cnn_dm` are downloaded, cmd-F for those links in this file).

`distilBART`:
```bash
deval 1 sshleifer/distilbart-xsum-12-3 xsum dbart_12_3_xsum_eval --fp16  # --help for more choices.
deval 1 sshleifer/distilbart-cnn_dm-12-6 cnn_dm dbart_12_6_cnn_eval --fp16
```

`distill-pegasus`:
```bash
deval 1 sshleifer/distill-pegasus-cnn-16-4 cnn_dm dpx_cnn_eval
deval 1 sshleifer/distill-pegasus-xsum-16-4 xsum dpx_xsum_eval
```

### Distillation
+ For all of the following commands, you can get roughly equivalent result and faster run times by passing `--num_beams=4`. That's not what we did for the paper.
+ Besides the KD section, you can also run commands with the built-in transformers trainer. See, for example, [builtin_trainer/train_distilbart_cnn.sh](./builtin_trainer/train_distilbart_cnn.sh).
+ Large performance deviations (> 5X slower or more than 0.5 Rouge-2 worse), should be reported.
+ Multi-gpu (controlled with `--gpus` should work, but might require more epochs).

#### Recommended Workflow
+ Get your dataset in the right format. (see 6 files above).
+ Find a teacher model [Pegasus](https://huggingface.co/models?search=pegasus) (slower, better ROUGE) or `facebook/bart-large-xsum`/`facebook/bart-large-cnn` (faster, slightly lower.).
Choose the checkpoint where the corresponding dataset is most similar (or identical to) your dataset.
+ Follow the sections in order below. You can stop after SFT if you are satisfied, or move on to pseudo-labeling if you want more performance.
+ student size: If you want a close to free 50% speedup, cut the decoder in half. If you want a larger speedup, cut it in 4. 
+ If your SFT run starts at a validation ROUGE-2 that is more than 10 pts below the teacher's validation ROUGE-2,  you have a bug. Switching to a more expensive technique will not help. Try setting a breakpoint and looking at generation and truncation defaults/hyper-parameters, and share your experience on the forums!

  
#### Initialization
We use [make_student.py](./make_student.py) to copy alternating layers from the teacher, and save the resulting model to disk
```bash
python make_student.py facebook/bart-large-xsum --save_path dbart_xsum_12_3  -e 12 -d 3
```
or for `pegasus-xsum`
```bash
python make_student.py google/pegasus-xsum --save_path dpx_xsum_16_4  --e 16 --d 4
```
we now have an initialized student saved to  `dbart_xsum_12_3`, which we will use for the following commands.
+ Extension: To replicate more complicated initialize experiments in section 6.1, or try your own. Use the `create_student_by_copying_alternating_layers` function.

#### Pegasus 
+ The following commands are written for BART and will require, at minimum, the following modifications
+ reduce batch size, and increase gradient accumulation steps so that the product `gpus * batch size * gradient_accumulation_steps = 256`. We used `--learning-rate` = 1e-4 * gradient accumulation steps.
+ don't use fp16
+ `--tokenizer_name google/pegasus-large`

### SFT (No Teacher Distillation)
You don't need `distillation.py`, you can just run:

```bash
python finetune.py \
  --data_dir xsum \
  --freeze_encoder --freeze_embeds \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --fp16_opt_level=O1 \
  --val_check_interval 0.1 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path dbart_xsum_12_3 \
  --train_batch_size=64 --eval_batch_size=64 \
  --sortish_sampler \
  --num_train_epochs=6 \
  --warmup_steps 500 \
  --output_dir distilbart_xsum_sft_12_3 --gpus 1
```

+ Note: The command that produced `sshleifer/distilbart-cnn-12-6` is at [train_distilbart_cnn.sh](./[train_distilbart_cnn.sh)

```bash
./train_distilbart_cnn.sh
```
<!--- runtime: 6H on NVIDIA RTX 24GB GPU -->
+ Tip: You can get the same simple distillation logic by using `distillation.py --no_teacher ` followed by identical arguments as the ones in `train_distilbart_cnn.sh`.
If you are using `wandb` and comparing the two distillation methods, using this entry point will make your logs consistent,
because you will have the same hyper-parameters logged in every run.

### Pseudo-Labeling
+ You don't need `distillation.py`.
+ Instructions to generate pseudo-labels and use pre-computed pseudo-labels can be found [here](./precomputed_pseudo_labels.md).
Simply run `finetune.py` with one of those pseudo-label datasets as `--data_dir` (`DATA`, below).

```bash
python finetune.py \
  --teacher facebook/bart-large-xsum --data_dir DATA \
  --freeze_encoder --freeze_embeds \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --fp16_opt_level=O1 \
  --val_check_interval 0.1 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path dbart_xsum_12_3 \
  --train_batch_size=32 --eval_batch_size=32 \
  --sortish_sampler \
  --num_train_epochs=5 \
  --warmup_steps 500 \
  --output_dir dbart_xsum_12_3_PL --gpus 1 --logger_name wandb
```

 

To combine datasets, as in Section 6.2, try something like:
```bash
curl -S https://cdn-datasets.huggingface.co/pseudo/xsum/bart_xsum_pl.tgz | tar -xvz -C .
curl -S https://cdn-datasets.huggingface.co/pseudo/xsum/pegasus_xsum.tgz | tar -xvz -C .
curl -S https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz | tar -xvz -C .
mkdir all_pl
cat bart_xsum_pl/train.source pegasus_xsum/train.source xsum/train.source > all_pl/train.source
cat bart_xsum_pl/train.target pegasus_xsum/train.target xsum/train.target > all_pl/train.target
cp xsum/val* all_pl
cp xsum/test* all_pl
```
then use `all_pl` as DATA in the command above.

#### Direct Knowledge Distillation (KD)
+ In this method, we use try to enforce that the student and teacher produce similar encoder_outputs, logits, and hidden_states using `SummarizationDistiller`.
+ This method was used for `sshleifer/distilbart-xsum-12-6`, `6-6`, and `9-6` checkpoints were produced.
+ You must use [`distillation.py`](./distillation.py). Note that this command initializes the student for you.

The command that produced `sshleifer/distilbart-xsum-12-6` is at [./train_distilbart_xsum.sh](train_distilbart_xsum.sh)
```bash
./train_distilbart_xsum.sh --logger_name wandb --gpus 1
```

+ Expected ROUGE-2 between 21.3 and 21.6, run time ~13H.
+ direct KD + Pegasus is VERY slow and works best with `--supervise_forward --normalize_hidden`.

<!--- runtime: 13H on V-100 16GB GPU. -->

### Citation

```bibtex
@misc{shleifer2020pretrained,
      title={Pre-trained Summarization Distillation}, 
      author={Sam Shleifer and Alexander M. Rush},
      year={2020},
      eprint={2010.13002},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```

This is the end of the distillation section, the rest of this doc pertains to general seq2seq commands.
