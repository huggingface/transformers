# RoBERTa (large) fine-tuned on Winograd Schema Challenge (WSC) data

Step from its original [repo](https://github.com/pytorch/fairseq/blob/master/examples/roberta/wsc/README.md)

The following instructions can be used to finetune RoBERTa on the WSC training
data provided by [SuperGLUE](https://super.gluebenchmark.com/).

Note that there is high variance in the results. For our GLUE/SuperGLUE
submission we swept over the learning rate (1e-5, 2e-5, 3e-5), batch size (16,
32, 64) and total number of updates (500, 1000, 2000, 3000), as well as the
random seed. Out of ~100 runs we chose the best 7 models and ensembled them.

**Approach:** The instructions below use a slightly different loss function than
what's described in the original RoBERTa arXiv paper. In particular,
[Kocijan et al. (2019)](https://arxiv.org/abs/1905.06290) introduce a margin
ranking loss between `(query, candidate)` pairs with tunable hyperparameters
alpha and beta. This is supported in our code as well with the `--wsc-alpha` and
`--wsc-beta` arguments. However, we achieved slightly better (and more robust)
results on the development set by instead using a single cross entropy loss term
over the log-probabilities for the query and all mined candidates. **The
candidates are mined using spaCy from each input sentence in isolation, so the
approach remains strictly pointwise.** This reduces the number of
hyperparameters and our best model achieved 92.3% development set accuracy,
compared to ~90% accuracy for the margin loss. Later versions of the RoBERTa
arXiv paper will describe this updated formulation.

### 1) Download the WSC data from the SuperGLUE website:
```bash
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip
unzip WSC.zip

# we also need to copy the RoBERTa dictionary into the same directory
wget -O WSC/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
```

### 2) Finetune over the provided training data:
```bash
TOTAL_NUM_UPDATES=2000  # Total number of training steps.
WARMUP_UPDATES=250      # Linearly increase LR over this many steps.
LR=2e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16        # Batch size per GPU.
SEED=1                  # Random seed.
ROBERTA_PATH=/path/to/roberta/model.pt

# we use the --user-dir option to load the task and criterion
# from the examples/roberta/wsc directory:
FAIRSEQ_PATH=/path/to/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/wsc

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train WSC/ \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --valid-subset val \
    --fp16 --ddp-backend no_c10d \
    --user-dir $FAIRSEQ_USER_DIR \
    --task wsc --criterion wsc --wsc-cross-entropy \
    --arch roberta_large --bpe gpt2 --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --max-update $TOTAL_NUM_UPDATES \
    --log-format simple --log-interval 100 \
    --seed $SEED
```

The above command assumes training on 4 GPUs, but you can achieve the same
results on a single GPU by adding `--update-freq=4`.

### 3) Evaluate
```python
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
roberta.cuda()
nsamples, ncorrect = 0, 0
for sentence, label in wsc_utils.jsonl_iterator('WSC/val.jsonl', eval=True):
    pred = roberta.disambiguate_pronoun(sentence)
    nsamples += 1
    if pred == label:
        ncorrect += 1
print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.9230769230769231
```

## RoBERTa training on WinoGrande dataset
We have also provided `winogrande` task and criterion for finetuning on the
[WinoGrande](https://mosaic.allenai.org/projects/winogrande) like datasets
where there are always two candidates and one is correct.
It's more efficient implementation for such subcases.

```bash
TOTAL_NUM_UPDATES=23750 # Total number of training steps.
WARMUP_UPDATES=2375     # Linearly increase LR over this many steps.
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=32        # Batch size per GPU.
SEED=1                  # Random seed.
ROBERTA_PATH=/path/to/roberta/model.pt

# we use the --user-dir option to load the task and criterion
# from the examples/roberta/wsc directory:
FAIRSEQ_PATH=/path/to/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/wsc

cd fairseq
CUDA_VISIBLE_DEVICES=0 fairseq-train winogrande_1.0/ \
  --restore-file $ROBERTA_PATH \
  --reset-optimizer --reset-dataloader --reset-meters \
  --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --valid-subset val \
  --fp16 --ddp-backend no_c10d \
  --user-dir $FAIRSEQ_USER_DIR \
  --task winogrande --criterion winogrande \
  --wsc-margin-alpha 5.0 --wsc-margin-beta 0.4 \
  --arch roberta_large --bpe gpt2 --max-positions 512 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
  --lr-scheduler polynomial_decay --lr $LR \
  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
  --max-sentences $MAX_SENTENCES \
  --max-update $TOTAL_NUM_UPDATES \
  --log-format simple --log-interval 100
```
[Original repo](https://github.com/pytorch/fairseq/tree/master/examples/roberta/wsc)
