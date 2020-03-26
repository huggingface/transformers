# Named Entity Recognition

Based on the scripts [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/ner/run_ner.py) for Pytorch and
[`run_tf_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/ner/run_tf_ner.py) for Tensorflow 2.

## GermEval 2014

This example fine-tune Bert Multilingual on GermEval 2014 (German NER).
Details and results for the fine-tuning provided by @stefan-it.

### Data (Download and pre-processing steps)

Data can be obtained from the [GermEval 2014](https://sites.google.com/site/germeval2014ner/data) shared task page.

Here are the commands for downloading and pre-processing train, dev and test datasets. The original data format has four (tab-separated) columns, in a pre-processing step only the two relevant columns (token and outer span NER annotation) are extracted:

```bash
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp
```

The GermEval 2014 dataset contains some strange "control character" tokens like `'\x96', '\u200e', '\x95', '\xad' or '\x80'`. One problem with these tokens is, that `BertTokenizer` returns an empty token for them, resulting in misaligned `InputExample`s. I wrote a script that a) filters these tokens and b) splits longer sentences into smaller ones (once the max. subtoken length is reached).

```bash
wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
```
Let's define some variables that we need for further pre-processing steps and training the model:

```bash
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
```

Run the pre-processing script on training, dev and test datasets:

```bash
python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
```

The GermEval 2014 dataset has much more labels than CoNLL-2002/2003 datasets, so an own set of labels must be used:

```bash
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

### Prepare the run

Additional environment variables must be set:

```bash
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
```

### Run the Pytorch version

To start training, just run:

```bash
python3 run_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

If your GPU supports half-precision training, just add the `--fp16` flag. After training, the model will be both evaluated on development and test datasets.

#### Evaluation

Evaluation on development dataset outputs the following for our example:

```bash
10/04/2019 00:42:06 - INFO - __main__ -   ***** Eval results  *****
10/04/2019 00:42:06 - INFO - __main__ -     f1 = 0.8623348017621146
10/04/2019 00:42:06 - INFO - __main__ -     loss = 0.07183869666975543
10/04/2019 00:42:06 - INFO - __main__ -     precision = 0.8467916366258111
10/04/2019 00:42:06 - INFO - __main__ -     recall = 0.8784592370979806
```

On the test dataset the following results could be achieved:

```bash
10/04/2019 00:42:42 - INFO - __main__ -   ***** Eval results  *****
10/04/2019 00:42:42 - INFO - __main__ -     f1 = 0.8614389652384803
10/04/2019 00:42:42 - INFO - __main__ -     loss = 0.07064602487454782
10/04/2019 00:42:42 - INFO - __main__ -     precision = 0.8604651162790697
10/04/2019 00:42:42 - INFO - __main__ -     recall = 0.8624150210424085
```

#### Comparing BERT (large, cased), RoBERTa (large, cased) and DistilBERT (base, uncased)

Here is a small comparison between BERT (large, cased), RoBERTa (large, cased) and DistilBERT (base, uncased) with the same hyperparameters as specified in the [example documentation](https://huggingface.co/transformers/examples.html#named-entity-recognition) (one run):

| Model | F-Score Dev | F-Score Test
| --------------------------------- | ------- | --------
| `bert-large-cased`            | 95.59 | 91.70
| `roberta-large`                  | 95.96 | 91.87
| `distilbert-base-uncased` | 94.34 | 90.32

#### Run PyTorch version using PyTorch-Lightning

Run `bash run_pl.sh` from the `ner` directory. This would also install `pytorch-lightning` and the `examples/requirements.txt`. It is a shell pipeline which would automatically download, pre-process the data and run the models in `germeval-model` directory. Logs are saved in `lightning_logs` directory.

Pass `--n_gpu` flag to change the number of GPUs. Default uses 1. At the end, the expected results are: `TEST RESULTS {'val_loss': tensor(0.0707), 'precision': 0.852427800698191, 'recall': 0.869537067011978, 'f1': 0.8608974358974358}`


### Run the Tensorflow 2 version

To start training, just run:

```bash
python3 run_tf_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

Such as the Pytorch version, if your GPU supports half-precision training, just add the `--fp16` flag. After training, the model will be both evaluated on development and test datasets.

#### Evaluation

Evaluation on development dataset outputs the following for our example:
```
           precision    recall  f1-score   support

 LOCderiv     0.7619    0.6154    0.6809        52
  PERpart     0.8724    0.8997    0.8858      4057
  OTHpart     0.9360    0.9466    0.9413       711
  ORGpart     0.7015    0.6989    0.7002       269
  LOCpart     0.7668    0.8488    0.8057       496
      LOC     0.8745    0.9191    0.8963       235
 ORGderiv     0.7723    0.8571    0.8125        91
 OTHderiv     0.4800    0.6667    0.5581        18
      OTH     0.5789    0.6875    0.6286        16
 PERderiv     0.5385    0.3889    0.4516        18
      PER     0.5000    0.5000    0.5000         2
      ORG     0.0000    0.0000    0.0000         3

micro avg     0.8574    0.8862    0.8715      5968
macro avg     0.8575    0.8862    0.8713      5968
```

On the test dataset the following results could be achieved:
```
           precision    recall  f1-score   support

  PERpart     0.8847    0.8944    0.8896      9397
  OTHpart     0.9376    0.9353    0.9365      1639
  ORGpart     0.7307    0.7044    0.7173       697
      LOC     0.9133    0.9394    0.9262       561
  LOCpart     0.8058    0.8157    0.8107      1150
      ORG     0.0000    0.0000    0.0000         8
 OTHderiv     0.5882    0.4762    0.5263        42
 PERderiv     0.6571    0.5227    0.5823        44
      OTH     0.4906    0.6667    0.5652        39
 ORGderiv     0.7016    0.7791    0.7383       172
 LOCderiv     0.8256    0.6514    0.7282       109
      PER     0.0000    0.0000    0.0000        11

micro avg     0.8722    0.8774    0.8748     13869
macro avg     0.8712    0.8774    0.8740     13869
```

## FiNER

This example fine-tune `bert-base-finnish-cased-v1` and `bert-base-finnish-uncased-v1` on FiNER (Finnish NER). Details and results for the fine-tuning provided by @bmichele

This example is based on the example given above for the GermEval2014 shared task. As such, to avoid repetitions, we limit ourselves to underline the differences with the procedure already described in the previous section.

### Data (Download and pre-processing steps)

Data can be obtained, for example, from the [mpsilfve/finer-data](https://github.com/mpsilfve/finer-data) repository by running the commands:
```bash
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2014.train.csv
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2014.dev.csv
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2015.test.csv
```

The original data format has three (tab-separated) columns, in a pre-processing step only the two relevant columns (token and outer span NER annotation) are extracted. This can be done by running the script `preprocess_fi.py` present in this folder. The script performs also a clean-up of the output dataset, removing unlabeled tokens such as `<HEADLINE>` and `<INGRESS>` that are present in the raw data files.
```bash
python preprocess_fi.py digitoday.2014.train.csv train.txt.tmp
python preprocess_fi.py digitoday.2014.dev.csv dev.txt.tmp
python preprocess_fi.py digitoday.2015.test.csv test.txt.tmp
```

Afterwards, we can set the max sequence length and tokenize the data using the script implemented by @stefan-it.
As stated above, the script removes "control character" tokens like `'\x96', '\u200e', '\x95', '\xad' or '\x80'` (to avoid misaligned `InputExample`s) and splits longer sentences into smaller ones (see germEval section for more details).

```bash
wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
export MAX_LENGTH=128
export BERT_MODEL=bert-base-finnish-cased-v1
python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
```

Finally, we store the labels into a file:

```bash
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

### Prepare the run

Additional environment variables must be set:

```bash
export OUTPUT_DIR=finer-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
```

### Run the Pytorch version

To start training, just run:

```bash
python3 run_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

Keep in mind that:
 * If your GPU supports half-precision training, just add the `--fp16` flag.
 * If you use an uncased model, add the flag `--do_lower_case`.

For convenience, all the steps illustrated above are implemented in the script `run_fi.sh` in this folder.

#### Evaluation - `bert-base-finnish-cased-v1`

Evaluation on development dataset outputs the following for our example:

```bash
03/23/2020 18:31:41 - INFO - __main__ -   ***** Eval results  *****
03/23/2020 18:31:41 - INFO - __main__ -     f1 = 0.9308227589029882
03/23/2020 18:31:41 - INFO - __main__ -     loss = 0.03493800052064539
03/23/2020 18:31:41 - INFO - __main__ -     precision = 0.9319672131147541
03/23/2020 18:31:41 - INFO - __main__ -     recall = 0.9296811120196239
```

On the test dataset the following results could be achieved:

```bash
03/23/2020 18:32:49 - INFO - __main__ -   ***** Eval results  *****
03/23/2020 18:32:49 - INFO - __main__ -     f1 = 0.9166069295101553
03/23/2020 18:32:49 - INFO - __main__ -     loss = 0.06830704841334574
03/23/2020 18:32:49 - INFO - __main__ -     precision = 0.9036513545347468
03/23/2020 18:32:49 - INFO - __main__ -     recall = 0.929939393939394
```

#### Evaluation - `bert-base-finnish-uncased-v1`

To reproduce the following results, add the flag `--do_lower_case` when launching the script `run_ner.py` and set the variable `$BERT_MODEL` as follows:

```bash
export BERT_MODEL=bert-base-finnish-uncased-v1
```

Evaluation on development dataset outputs the following for our example:

```
03/25/2020 11:41:21 - INFO - __main__ -   ***** Eval results  *****
03/25/2020 11:41:21 - INFO - __main__ -     f1 = 0.9222542163718633
03/25/2020 11:41:21 - INFO - __main__ -     loss = 0.04647757345699046
03/25/2020 11:41:21 - INFO - __main__ -     precision = 0.9279801324503312
03/25/2020 11:41:21 - INFO - __main__ -     recall = 0.9165985282093213
```

On the test dataset the following results could be achieved:

```
03/25/2020 11:42:28 - INFO - __main__ -   ***** Eval results  *****
03/25/2020 11:42:28 - INFO - __main__ -     f1 = 0.9099820681410639
03/25/2020 11:42:28 - INFO - __main__ -     loss = 0.07058404185008013
03/25/2020 11:42:28 - INFO - __main__ -     precision = 0.8976415094339623
03/25/2020 11:42:28 - INFO - __main__ -     recall = 0.9226666666666666
```

### Run the Tensorflow 2 version

To start training, just run:

```bash
python3 run_tf_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

Again, add flags `--fp16` if your GPU supports half-precision training and `--do_lower_case` if you use uncased pre-trained models.
For convenience, all the steps illustrated above are implemented in the script `run_fi_tf.sh` in this folder.

#### Evaluation - bert-base-finnish-cased-v1

Evaluation on development dataset outputs the following for our example:

```
           precision    recall  f1-score   support

      PRO     0.7571    0.7477    0.7524      1847
      LOC     0.9326    0.9412    0.9369       544
      ORG     0.9624    0.9676    0.9650       185
      PER     0.8547    0.8497    0.8522       173
    EVENT     0.9551    0.9515    0.9533       268
     DATE     0.0000    0.0000    0.0000         2

micro avg     0.8249    0.8195    0.8222      3019
macro avg     0.8240    0.8195    0.8217      3019
```

On the test dataset the following results could be achieved:

```
           precision    recall  f1-score   support

      LOC     0.9270    0.9526    0.9396      1879
    EVENT     0.9463    0.9648    0.9554       511
      PRO     0.7427    0.7403    0.7415      6203
      ORG     0.8733    0.9680    0.9182       406
      PER     0.8235    0.8220    0.8228      1073
     DATE     0.8889    0.8889    0.8889        18

micro avg     0.8026    0.8093    0.8060     10090
macro avg     0.8014    0.8093    0.8052     10090
```

#### Evaluation - bert-base-finnish-uncased-v1

To reproduce the following results, add the flag `--do_lower_case` when launching the script `run_tf_ner.py` and set the variable `$BERT_MODEL` as follows:

```bash
export BERT_MODEL=bert-base-finnish-uncased-v1
```

Evaluation on development dataset outputs the following for our example:

```
           precision    recall  f1-score   support

      LOC     0.9231    0.9265    0.9248       544
      PRO     0.9592    0.9551    0.9571      1847
      ORG     0.9626    0.9730    0.9677       185
    EVENT     0.9700    0.9664    0.9682       268
      PER     0.8696    0.8092    0.8383       173
     DATE     0.0000    0.0000    0.0000         2

micro avg     0.9487    0.9430    0.9458      3019
macro avg     0.9481    0.9430    0.9455      3019
```

On the test dataset the following results could be achieved:

```
           precision    recall  f1-score   support

      LOC     0.9211    0.9564    0.9384      1879
      PRO     0.9314    0.9457    0.9385      6204
    EVENT     0.9486    0.9393    0.9440       511
      PER     0.7933    0.7940    0.7937      1073
      ORG     0.8568    0.9433    0.8980       406
     DATE     0.8000    0.8889    0.8421        18

micro avg     0.9124    0.9310    0.9216     10091
macro avg     0.9124    0.9310    0.9215     10091
```

# TODOs

Before PR in huggingface:

 * remove `preprocess.py`
 * move `preprocess_fi.py` and add step to download it (?)

