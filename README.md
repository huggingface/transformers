# BERT

## Introduction

**BERT**, or **B**idirectional **E**mbedding **R**epresentations from
**T**ransformers, is a new method of pre-training language representations which
obtains state-of-the-art results on a wide array of Natural Language Processing
(NLP) tasks.

Our academic paper which describes BERT in detail and provides full results on a
number of tasks can be found here:
[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

To give a few numbers, here are the results on the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) question answering
task:

SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1
------------------------------------- | :------: | :------:
1st Place Ensemble - BERT             | **87.4** | **93.2**
2nd Place Ensemble - nlnet            | 86.0     | 91.7
1st Place Single Model - BERT         | **85.1** | **91.8**
2nd Place Single Model - nlnet        | 83.5     | 90.1

And several natural language inference tasks:

System                  | MultiNLI | Question NLI | SWAG
----------------------- | :------: | :----------: | :------:
BERT                    | **86.7** | **91.1**     | **86.3**
OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0

Plus many other tasks.

Moreover, these results were all obtained with almost no task-specific neural
network architecture design.

If you already know what BERT is and you just want to get started, you can
[download the pre-trained models](#pre-trained-models) and
[run a state-of-the-art fine-tuning](#fine-tuning-with-bert) in only a few
minutes.

## What is BERT?

BERT is method of pre-training language representations, meaning that we train a
general-purpose "language understanding" model on a large text corpus (like
Wikipedia), and then use that model for downstream NLP tasks that we are about
(like question answering). BERT outperforms previous methods because it is the
first *unsupervised*, *deeply bidirectional* system for pre-training NLP.

*Unsupervised* means that BERT was trained using only a plain text corpus, which
is important because an enormous amount of plain text data is publicly available
on the web in many languages.

Pre-trained representations can also either be *context-free* or *contextual*,
and contextual representations can further be *unidirectional* or
*bidirectional*. Context-free models such as
[word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) or
[GloVe](https://nlp.stanford.edu/projects/glove/) generate a single "word
embedding" representation for each word in the vocabulary, so `bank` would have
the same representation in `bank deposit` and `river bank`. Contextual models
instead generate a representation of each word that is based on the other words
in the sentence.

BERT was built upon recent work in pre-training contextual representations —
including [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432),
[Generative Pre-Training](https://blog.openai.com/language-unsupervised/),
[ELMo](https://allennlp.org/elmo), and
[ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
— but crucially these models are all *unidirectional* or *shallowly
bidirectional*. This means that each word is only contextualized using the words
to its left (or right). For example, in the sentence `I made a bank deposit` the
unidirectional representation of `bank` is only based on `I made a` but not
`deposit`. Some previous work does combine the representations from separate
left-context and right-context models, but only in a "shallow" manner. BERT
represents "bank" using both its left and right context — `I made a ... deposit`
— starting from the very bottom of a deep neural network, so it is *deeply
bidirectional*.

BERT uses a simple approach for this: We mask out 15% of the words in the input,
run the entire sequence through a deep bidirectional
[Transformer](https://arxiv.org/abs/1706.03762) encoder, and then predict only
the masked words. For example:

```
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

In order to learn relationships between sentences, we also train on a simple
task which can be generated from any monolingual corpus: Given two sentences `A`
and `B`, is `B` the actual next sentence that comes after `A`, or just a random
sentence from the corpus?

```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
```

```
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```

We then train a large model (12-layer to 24-layer Transformer) on a large corpus
(Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb)) for a long time (1M
update steps), and that's BERT.

Using BERT has two stages: *Pre-training* and *fine-tuning*.

**Pre-training** is fairly expensive (four days on 4 to 16 Cloud TPUs), but is a
one-time procedure for each language (current models are English-only, but
multilingual models will be released in the near future). We are releasing a
number of pre-trained models from the paper which were pre-trained at Google.
Most NLP researchers will never need to pre-train their own model from scratch.

**Fine-tuning** is inexpensive. All of the results in the paper can be
replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU,
starting from the exact same pre-trained model. SQuAD, for example, can be
trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of
91.0%, which is the single system state-of-the-art.

The other important aspect of BERT is that it can be adapted to many types of
NLP tasks very easily. In the paper, we demonstrate state-of-the-art results on
sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level
(e.g., NER), and span-level (e.g., SQuAD) tasks with almost no task-specific
modifications.

## What has been released in this repository?

We are releasing the following:

*   TensorFlow code for the BERT model architecture (which is mostly a standard
    [Transformer](https://arxiv.org/abs/1706.03762) architecture).
*   Pre-trained checkpoints for both the lowercase and cased version of
    `BERT-Base` and `BERT-Large` from the paper.
*   TensorFlow code for push-button replication of the most important
    fine-tuning experiments from the paper, including SQuAD, MultiNLI, and MRPC.

All of the code in this repository works out-of-the-box with CPU, GPU, and Cloud
TPU.

## Pre-trained models

We are releasing the `BERT-Base` and `BERT-Large` models from the paper.
`Uncased` means that the text has been lowercased before WordPiece tokenization,
e.g., `John Smith` becomes `john smith`. The `Uncased` model also strips out any
accent markers. `Cased` means that the true case and accent markers are
preserved. Typically, the `Uncased` model is better unless you know that case
information is important for your task (e.g., Named Entity Recognition or
Part-of-Speech tagging).

These models are all released under the same license as the source code (Apache
2.0).

The links to the models are here (right-cick, 'Save link as...' on the name):

*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads , 110M parameters
*   **`BERT-Large, Cased`**: 24-layer, 1024-hidden, 16-heads, 340M parameters
    (Not available yet. Needs to be re-generated).

Each .zip file contains three items:

*   A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained
    weights (which is actually 3 files).
*   A vocab file (`vocab.txt`) to map WordPiece to word id.
*   A config file (`bert_config.json`) which specifies the hyperparameters of
    the model.

## Fine-tuning with BERT

**Important**: All results on the paper were fine-tuned on a single Cloud TPU,
which has 64GB of RAM. It is currently not possible to re-produce most of the
`BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because
the maximum batch size that can fit in memory is too small. We are working on
adding code to this repository which allows for much larger effective batch size
on the GPU. See the section on [out-of-memory issues](#out-of-memory-issues) for
more details.

This code was tested with TensorFlow 1.11.0. It was tested with Python2 and
Python3 (but more thoroughly with Python2, since this is what's used internally
in Google).

The fine-tuning examples which use `BERT-Base` should be able to run on a GPU
that has at least 12GB of RAM using the hyperparameters given.

### Fine-tuning with Cloud TPUs

Most of the examples below assumes that you will be running training/evaluation
on your local machine, using a GPU like a Titan X or GTX 1080.

However, if you have access to a Cloud TPU that you want to train on, just add
the following flags to `run_classifier.py` or `run_squad.py`:

```
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

Please see the
[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs.

On Cloud TPUs, the pretrained model and the output directory will need to be on
Google Cloud Storage. For example, if you have a bucket named `some_bucket`, you
might use the following flags instead:

```
  --output_dir=gs://some_bucket/my_output_dir/
```

The unzipped pre-trained model files can also be found in the Google Cloud
Storage folder `gs://bert_models/2018_10_18`. For example:

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

### Sentence (and sentence-pair) classification tasks

Before running this example you must download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`. Next, download the `BERT-Base`
checkpoint and unzip it to some directory `$BERT_BASE_DIR`.

This example code fine-tunes `BERT-Base` on the Microsoft Research Paraphrase
Corpus (MRPC) corpus, which only contains 3,600 examples and can fine-tune in a
few minutes on most GPUs.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

You should see output like this:

```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

This means that the Dev set accuracy was 84.55%. Small sets like MRPC have a
high variance in the Dev set accuracy, even when starting from the same
pre-training checkpoint. If you re-run multiple times (making sure to point to
different `output_dir`), you should see results between 84% and 88%.

A few other pre-trained models are implemented off-the-shelf in
`run_classifier.py`, so it should be straightforward to follow those examples to
use BERT for any single-sentence or sentence-pair classification task.

Note: You might see a message `Running train on CPU`. This really just means
that it's running on something other than a Cloud TPU, which includes a GPU.

### SQuAD

The Stanford Question Answering Dataset (SQuAD) is a popular question answering
benchmark dataset. BERT (at the time of the release) obtains state-of-the-art
results on SQuAD with almost no task-specific network architecture modifications
or data augmentation. However, it does require semi-complex data pre-processing
and post-processing to deal with (a) the variable-length nature of SQuAD context
paragraphs, and (b) the character-level answer annotations which are used for
SQuAD training. This processing is implemented and documented in `run_squad.py`.

To run on SQuAD, you will first need to download the dataset. The
[SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to
link to the v1.1 datasets any longer, but the necessary files can be found here:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Download these to some directory `$SQUAD_DIR`.

The state-of-the-art SQuAD results from the paper currently cannot be reproduced
on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does
not seem to fit on a 12GB GPU using `BERT-Large`). However, a reasonably strong
`BERT-Base` model can be trained on the GPU with these hyperparameters:

```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

The dev set predictions will be saved into a file called `predictions.json` in
the `output_dir`:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

Which should produce an output like this:

```shell
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

You should see a result similar to the 88.5% reported in the paper for
`BERT-Base`.

If you have access to a Cloud TPU, you can train with `BERT-Large`. Here is a
set of hyperparameters (slightly different than the paper) which consistently
obtain around 90.5%-91.0% F1 single-system trained only on SQuAD:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=48 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

For example, one random run with these parameters produces the following Dev
scores:

```shell
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

If you fine-tune for one epoch on
[TriviaQA](http://nlp.cs.washington.edu/triviaqa/) before this the results will
be even better, but you will need to convert TriviaQA into the SQuAD json
format.

### Out-of-memory issues

All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of
device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely
to encounter out-of-memory issues if you use the same hyperparameters described
in the paper.

The factors that affect memory usage are:

*   **`max_seq_length`**: The released models were trained with sequence lengths
    up to 512, but you can fine-tune with a shorter max sequence length to save
    substantial memory. This is controlled by the `max_seq_length` flag in our
    example code.

*   **`train_batch_size`**: The memory usage is also directly proportional to
    the batch size.

*   **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model
    requires significantly more memory than `BERT-Base`.

*   **Optimizer**: The default optimizer for BERT is Adam, which requires a lot
    of extra memory to store the `m` and `v` vectors. Switching to a more memory
    efficient optimizer can reduce memory usage, but can also affect the
    results. We have not experimented with other optimizers for fine-tuning.

Using the default training scripts (`run_classifier.py` and `run_squad.py`), we
benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with
TensorFlow 1.11.0:

System       | Seq Length | Max Batch Size
------------ | ---------- | --------------
`BERT-Base`  | 64         | 64
...          | 128        | 32
...          | 256        | 16
...          | 320        | 14
...          | 384        | 12
...          | 512        | 6
`BERT-Large` | 64         | 12
...          | 128        | 6
...          | 256        | 2
...          | 320        | 1
...          | 384        | 0
...          | 512        | 0

Unfortunately, these max batch sizes for `BERT-Large` are so small that they
will actually harm the model accuracy, regardless of the learning rate used. We
are working on adding code to this repository which will allow much larger
effective batch sizes to be used on the GPU. The code will be based on one (or
both) of the following techniques:

*   **Gradient accumulation**: The samples in a minibatch are typically
    independent with respect to gradient computation (excluding batch
    normalization, which is not used here). This means that the gradients of
    multiple smaller minibatches can be accumulated before performing the weight
    update, and this will be exactly equivalent to a single larger update.

*   [**Gradient checkpointing**](https://github.com/openai/gradient-checkpointing):
    The major use of GPU/TPU memory during DNN training is caching the
    intermediate activations in the forward pass that are necessary for
    efficient computation in the backward pass. "Gradient checkpointing" trades
    memory for compute time by re-computing the activations in an intelligent
    way.

**However, this is not implemented in the current release.**

## Using BERT to extract fixed feature vectors (like ELMo)

In certain cases, rather than fine-tuning the entire pre-trained model
end-to-end, it can be beneficial to obtained *pre-trained contextual
embeddings*, which are fixed contextual representations of each input token
generated from the hidden layers of the pre-trained model. This should also
mitigate most of the out-of-memory issues.

As an example, we include the script `extract_features.py` which can be used
like this:

```shell
# Sentence A and Sentence B are separated by the ||| delimiter.
# For single sentence inputs, don't use the delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

This will create a JSON file (one line per line of input) containing the BERT
activations from each Transformer layer specified by `layers` (-1 is the final
hidden layer of the Transformer, etc.)

Note that this script will produce very large output files (by default, around
15kb for every input token).

If you need to maintain alignment between the original and tokenized words (for
projecting training labels), see the [Tokenization](#tokenization) section
below.

## Tokenization

For sentence-level tasks (or sentence-pair) tasks, tokenization is very simple.
Just follow the example code in `run_classifier.py` and `extract_features.py`.
The basic procedure for sentence-level tasks is:

1.  Instantiate an instance of `tokenizer = tokenization.FullTokenizer`

2.  Tokenize the raw text with `tokens = tokenizer.tokenize(raw_text)`.

3.  Truncate to the maximum sequence length. (You can use up to 512, but you
    probably want to use shorter if possible for memory and speed reasons.)

4.  Add the `[CLS]` and `[SEP]` tokens in the right place.

Word-level and span-level tasks (e.g., SQuAD and NER) are more complex, since
you need to maintain alignment between your input text and output text so that
you can project your training labels. SQuAD is a particularly complex example
because the input labels are *character*-based, and SQuAD paragraphs are often
longer than our maximum sequence length. See the code in `run_squad.py` to show
how we handle this.

Before we describe the general recipe for handling word-level tasks, it's
important to understand what exactly our tokenizer is doing. It has three main
steps:

1.  **Text normalization**: Convert all whitespace characters to spaces, and
    (for the `Uncased` model) lowercase the input and strip out accent markers.
    E.g., `John Johanson's, → john johanson's,`.

2.  **Punctuation splitting**: Split *all* punctuation characters on both sides
    (i.e., add whitespace around all punctuation characters). Punctuation
    characters are defined as (a) Anything with a `P*` Unicode class, (b) any
    non-letter/number/space ASCII character (e.g., characters like `$` which are
    technically not punctuation). E.g., `john johanson's, → john johanson ' s ,`

3.  **WordPiece tokenization**: Apply whitespace tokenization to the output of
    the above procedure, and apply
    [WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py)
    tokenization to each token separately. (Our implementation is directly based
    on the one from `tensor2tensor`, which is linked). E.g., `john johanson ' s
    , → john johan ##son ' s ,`

The advantage of this scheme is that it is "compatible" with most existing
English tokenizers. For example, imagine that you have a part-of-speech tagging
task which looks like this:

```
Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
```

The tokenized output will look like this:

```
Tokens: john johan ##son ' s house
```

Crucially, this would be the same output as if the raw text were `John
Johanson's house` (with no space before the `'s`).

If you have a pre-tokenized representation with word-level annotations, you can
simply tokenize each input word independently, and deterministically maintain an
original-to-tokenized alignment:

```python
### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```

Now `orig_to_tok_map` can be used to project `labels` to the tokenized
representation.

There are common English tokenization schemes which will cause a slight mismatch
between how BERT was pre-trained. For example, if your input tokenization splits
off contractions like `do n't`, this will cause a mismatch. If it is possible to
do so, you should pre-process your data to convert these back to raw-looking
text, but if it's not possible, this mismatch is likely not a big deal.

## Pre-training with BERT

We are releasing code to do "masked LM" and "next sentence prediction" on an
arbitrary text corpus. Note that this is *not* the exact code that was used for
the paper (the original code was written in C++, and had some additional
complexity), but this code does generate pre-training data as described in the
paper.

Here's how to run the data generation. The input is a plain text file, with one
sentence per line. (It is important that these be actual sentences for the "next
sentence prediction" task). Documents are delimited by empty lines. The output
is a set of `tf.train.Example`s serialized into `TFRecord` file format.

This script stores all of the examples for the entire input file in memory, so
for large data files you should shard the input file and call the script
multiple times. (You can pass in a file glob to `run_pretraining.py`, e.g.,
`tf_examples.tf_record*`.)

The `max_predictions_per_seq` is the maximum number of masked LM predictions per
sequence. You should set this to around `max_seq_length` * `masked_lm_prob` (the
script doesn't do that automatically because the exact value needs to be passed
to both scripts).

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

Here's how to run the pre-training. Do not include `init_checkpoint` if you are
pre-training from scratch. The model configuration (including vocab size) is
specified in `bert_config_file`. This demo code only pre-trains for a small
number of steps (20), but in practice you will probably want to set
`num_train_steps` to 10000 steps or more. The `max_seq_length` and
`max_predictions_per_seq` parameters passed to `run_pretraining.py` must be the
same as `create_pretraining_data.py`.

```shell
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

This will produce an output like this:

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

Note that since our `sample_text.txt` file is very small, this example training
will overfit that data in only a few steps and produce unrealistically high
accuracy numbers.

### Pre-training tips and caveats

*   If your task has a large domain-specific corpus available (e.g., "movie
    reviews" or "scientific papers"), it will likely be beneficial to run
    additional steps of pre-training on your corpus, starting from the BERT
    checkpoint.
*   The learning rate we used in the paper was 1e-4. However, if you are doing
    additional steps of pre-training starting from an existing BERT checkpoint,
    you should use a smaller learning rate (e.g., 2e-5).
*   Current BERT models are English-only, but we do plan to release a
    multilingual model which has been pre-trained on a lot of languages in the
    near future (hopefully by the end of November 2018).
*   Longer sequences are disproportionately expensive because attention is
    quadratic to the sequence length. In other words, a batch of 64 sequences of
    length 512 is much more expensive than a batch of 256 sequences of
    length 128. The fully-connected/convolutional cost is the same, but the
    attention cost is far greater for the 512-length sequences. Therefore, one
    good recipe is to pre-train for, say, 90,000 steps with a sequence length of
    128 and then for 10,000 additional steps with a sequence length of 512. The
    very long sequences are mostly needed to learn positional embeddings, which
    can be learned fairly quickly. Note that this does require generating the
    data twice with different values of `max_seq_length`.
*   If you are pre-training from scratch, be prepared that pre-training is
    computationally expensive, especially on GPUs. If you are pre-training from
    scratch, our recommended recipe is to pre-train a `BERT-Base` on a single
    [preemptable Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing), which
    takes about 2 weeks at a cost of about $500 USD (based on the pricing in
    October 2018). You will have to scale down the batch size when only training
    on a single Cloud TPU, compared to what was used in the paper. It is
    recommended to use the largest batch size that fits into TPU memory.

### Pre-training data

We will **not** be able to release the pre-processed datasets used in the paper.
For Wikipedia, the recommended pre-processing is to download
[the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),
extract the text with
[`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply
any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the
[BookCorpus](http://yknzhu.wixsite.com/mbweb) no longer have it available for
public download. The
[Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
is a somewhat smaller (200M word) collection of older books that are public
domain.

[Common Crawl](http://commoncrawl.org/) is another very large collection of
text, but you will likely have to do substantial pre-processing and cleanup to
extract a usuable corpus for pre-training BERT.

### Learning a new WordPiece vocabulary

This repository does not include code for *learning* a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with
dependencies on Google's internal libraries. For English, it is almost always
better to just start with our vocabulary and pre-trained models. For learning
vocabularies of other languages, there are a number of open source options
available. However, keep in mind that these are not compatible with our
`tokenization.py` library:

*   [Google's SentencePiece library](https://github.com/google/sentencepiece)

*   [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

*   [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)

## Using BERT in Colab

If you want to use BERT with [Colab](https://colab.sandbox.google.com), you can
get started with the notebook
"[BERT FineTuning with Cloud TPUs](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".
**At the time of this writing (October 31st, 2018), Colab users can access a
Cloud TPU completely for free.** Note: One per user, availability limited,
requires a Google Cloud Platform account with storage (although storage may be
purchased with free credit for signing up with GCP), and this capability may not
longer be available in the future. Click on the BERT Colab that was just linked
for more information.

## FAQ

#### Is this code compatible with Cloud TPUs? What about GPUs?

Yes, all of the code in this repository works out-of-the-box with CPU, GPU, and
Cloud TPU. However, GPU training is single-GPU only.

#### I am getting out-of-memory errors, what is wrong?

See the section on [out-of-memory issues](#out-of-memory-issues) for more
information.

#### Is there a PyTorch version available?

There is no official PyTorch implementation. If someone creates a line-for-line
PyTorch reimplementation so that our pre-trained checkpoints can be directly
converted, we would be happy to link to that PyTorch version here.

#### Will models in other languages be released?

Yes, we plan to release a multi-lingual BERT model in the near future. We cannot
make promises about exactly which languages will be included, but it will likely
be a single model which includes *most* of the languages which have a
significantly-sized Wikipedia.

#### Will models larger than `BERT-Large` be released?

So far we have not attempted to train anything larger than `BERT-Large`. It is
possible that we will release larger models if we are able to obtain significant
improvements.

#### What license is this library released under?

All code *and* models are released under the Apache 2.0 license. See the
`LICENSE` file for more information.

#### How do I cite BERT?

For now, cite [the Arxiv paper](https://arxiv.org/abs/1810.04805):

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

If we submit the paper to a conference or journal, we will update the BibTeX.

## Disclaimer

This is not an official Google product.

## Contact information

For help or issues using BERT, please submit a GitHub issue.

For personal communication related to BERT, please contact Jacob Devlin
(`jacobdevlin@google.com`), Ming-Wei Chang (`mingweichang@google.com`), or
Kenton Lee (`kentonl@google.com`).
