# 👾 PyTorch-Transformers

[![CircleCI](https://circleci.com/gh/huggingface/pytorch-transformers.svg?style=svg)](https://circleci.com/gh/huggingface/pytorch-transformers)

PyTorch-Transformers (formerly known as `pytorch-pretrained-bert`) is a library of state-of-the-art pre-trained models for Natural Language Processing (NLP).

The library currently contains PyTorch implementations, pre-trained model weights, usage scripts and conversion utilities for the following models:

1. **[BERT](https://github.com/google-research/bert)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[GPT](https://github.com/openai/finetune-transformer-lm)** (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. **[GPT-2](https://blog.openai.com/better-language-models/)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. **[Transformer-XL](https://github.com/kimiyoung/transformer-xl)** (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. **[XLM](https://github.com/facebookresearch/XLM/)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. **[RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. **[DistilBERT](https://github.com/huggingface/pytorch-transformers/tree/master/examples/distillation)** (from HuggingFace), released together with the blogpost [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5
) by Victor Sanh, Lysandre Debut and Thomas Wolf.

These implementations have been tested on several datasets (see the example scripts) and should match the performances of the original implementations (e.g. ~93 F1 on SQuAD for BERT Whole-Word-Masking, ~88 F1 on RocStories for OpenAI GPT, ~18.3 perplexity on WikiText 103 for Transformer-XL, ~0.916 Peason R coefficient on STS-B for XLNet). You can find more details on the performances in the Examples section of the [documentation](https://huggingface.co/pytorch-transformers/examples.html).

| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Online demo](#online-demo) | Experimenting with this repo’s text generation capabilities |
| [Quick tour: Usage](#quick-tour) | Tokenizers & models usage: Bert and GPT-2 |
| [Quick tour: Fine-tuning/usage scripts](#quick-tour-of-the-fine-tuningusage-scripts) | Using provided scripts: GLUE, SQuAD and Text generation |
| [Migrating from pytorch-pretrained-bert to pytorch-transformers](#Migrating-from-pytorch-pretrained-bert-to-pytorch-transformers) | Migrating your code from pytorch-pretrained-bert to pytorch-transformers |
| [Documentation](https://huggingface.co/pytorch-transformers/) | Full API documentation and more |

## Installation

This repo is tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 1.0.0+

### With pip

PyTorch-Transformers can be installed by pip as follows:

```bash
pip install pytorch-transformers
```

### From source

Clone the repository and run:

```bash
pip install [--editable] .
```

### Tests

A series of tests is included for the library and the example scripts. Library tests can be found in the [tests folder](https://github.com/huggingface/pytorch-transformers/tree/master/pytorch_transformers/tests) and examples tests in the [examples folder](https://github.com/huggingface/pytorch-transformers/tree/master/examples).

These tests can be run using `pytest` (install pytest if needed with `pip install pytest`).

You can run the tests from the root of the cloned repository with the commands:

```bash
python -m pytest -sv ./pytorch_transformers/tests/
python -m pytest -sv ./examples/
```

### Do you want to run a Transformer model on a mobile device?

You should check out our [`swift-coreml-transformers`](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains an example of a conversion script from a Pytorch trained Transformer model (here, `GPT-2`) to a CoreML model that runs on iOS devices.

At some point in the future, you'll be able to seamlessly move from pre-training or fine-tuning models in PyTorch to productizing them in CoreML,
or prototype a model or an app in CoreML then research its hyperparameters or architecture from PyTorch. Super exciting!

## Online demo

**[Write With Transformer](https://transformer.huggingface.co)**, built by the Hugging Face team at transformer.huggingface.co, is the official demo of this repo’s text generation capabilities.
You can use it to experiment with completions generated by `GPT2Model`, `TransfoXLModel`, and `XLNetModel`.

> “🦄 Write with transformer is to writing what calculators are to calculus.”

![write_with_transformer](https://transformer.huggingface.co/front/assets/thumbnail-large.png)

## Quick tour

Let's do a very quick overview of PyTorch-Transformers. Detailed examples for each model architecture (Bert, GPT, GPT-2, Transformer-XL, XLNet and XLM) can be found in the [full documentation](https://huggingface.co/pytorch-transformers/).

```python
import torch
from pytorch_transformers import *

# PyTorch-Transformers has a unified API
# for 7 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
          (RobertaModel,    RobertaTokenizer,   'roberta-base')]

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
                      BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for model_class in BERT_MODEL_CLASSES:
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained('bert-base-uncased')

# Models can return full list of hidden-states & attentions weights at each layer
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
all_hidden_states, all_attentions = model(input_ids)[-2:]

# Models are compatible with Torchscript
model = model_class.from_pretrained(pretrained_weights, torchscript=True)
traced_model = torch.jit.trace(model, (input_ids,))

# Simple serialization for models and tokenizers
model.save_pretrained('./directory/to/save/')  # save
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer.save_pretrained('./directory/to/save/')  # save
tokenizer = tokenizer_class.from_pretrained('./directory/to/save/')  # re-load

# SOTA examples for GLUE, SQUAD, text generation...
```

## Quick tour of the fine-tuning/usage scripts

The library comprises several example scripts with SOTA performances for NLU and NLG tasks:

- `run_glue.py`: an example fine-tuning Bert, XLNet and XLM on nine different GLUE tasks (*sequence-level classification*)
- `run_squad.py`: an example fine-tuning Bert, XLNet and XLM on the question answering dataset SQuAD 2.0 (*token-level classification*)
- `run_generation.py`: an example using GPT, GPT-2, Transformer-XL and XLNet for conditional language generation
- other model-specific examples (see the documentation).

Here are three quick usage examples for these scripts:

### `run_glue.py`: Fine-tuning on GLUE tasks for sequence classification

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

You should also install the additional packages required by the examples:

```shell
pip install -r ./examples/requirements.txt
```

```shell
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

#### Fine-tuning XLNet model on the STS-B regression task

This example code fine-tunes XLNet on the STS-B corpus using parallel training on a server with 4 V100 GPUs.
Parallel training is a simple way to use several GPUs (but is slower and less flexible than distributed training, see below).

```shell
export GLUE_DIR=/path/to/glue

python ./examples/run_glue.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train  \
    --do_eval   \
    --task_name=sts-b     \
    --data_dir=${GLUE_DIR}/STS-B  \
    --output_dir=./proc_data/sts-b-110   \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --gradient_accumulation_steps=1 \
    --max_steps=1200  \
    --model_name=xlnet-large-cased   \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120
```

On this machine we thus have a batch size of 32, please increase `gradient_accumulation_steps` to reach the same batch size if you have a smaller machine. These hyper-parameters should result in a Pearson correlation coefficient of `+0.917` on the development set.

#### Fine-tuning Bert model on the MRPC classification task

This example code fine-tunes the Bert Whole Word Masking model on the Microsoft Research Paraphrase Corpus (MRPC) corpus using distributed training on 8 V100 GPUs to reach a F1 > 92.

```bash
python -m torch.distributed.launch --nproc_per_node 8 ./examples/run_glue.py   \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name MRPC \
    --do_train   \
    --do_eval   \
    --do_lower_case   \
    --data_dir $GLUE_DIR/MRPC/   \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir /tmp/mrpc_output/ \
    --overwrite_output_dir   \
    --overwrite_cache \
```

Training with these hyper-parameters gave us the following results:

```bash
  acc = 0.8823529411764706
  acc_and_f1 = 0.901702786377709
  eval_loss = 0.3418912578906332
  f1 = 0.9210526315789473
  global_step = 174
  loss = 0.07231863956341798
```

### `run_squad.py`: Fine-tuning on SQuAD for question-answering

This example code fine-tunes BERT on the SQuAD dataset using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD:

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./examples/run_squad.py \
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
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \
```

Training with these hyper-parameters gave us the following results:

```bash
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
{"exact_match": 86.91579943235573, "f1": 93.1532499015869}
```

This is the model provided as `bert-large-uncased-whole-word-masking-finetuned-squad`.

### `run_generation.py`: Text generation with GPT, GPT-2, Transformer-XL and XLNet

A conditional generation script is also included to generate text from a prompt.
The generation script includes the [tricks](https://github.com/rusiaaman/XLNet-gen#methodology) proposed by by Aman Rusia to get high quality generation with memory models like Transformer-XL and XLNet (include a predefined text to make short inputs longer).

Here is how to run the script with the small version of OpenAI GPT-2 model:

```shell
python ./examples/run_generation.py \
    --model_type=gpt2 \
    --length=20 \
    --model_name_or_path=gpt2 \
```

## Migrating from pytorch-pretrained-bert to pytorch-transformers

Here is a quick summary of what you should take care of when migrating from `pytorch-pretrained-bert` to `pytorch-transformers`

### Models always output `tuples`

The main breaking change when migrating from `pytorch-pretrained-bert` to `pytorch-transformers` is that the models forward method always outputs a `tuple` with various elements depending on the model and the configuration parameters.

The exact content of the tuples for each model are detailed in the models' docstrings and the [documentation](https://huggingface.co/pytorch-transformers/).

In pretty much every case, you will be fine by taking the first element of the output as the output you previously used in `pytorch-pretrained-bert`.

Here is a `pytorch-pretrained-bert` to `pytorch-transformers` conversion example for a `BertForSequenceClassification` classification model:

```python
# Let's load our model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# If you used to have this line in pytorch-pretrained-bert:
loss = model(input_ids, labels=labels)

# Now just use this line in pytorch-transformers to extract the loss from the output tuple:
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# In pytorch-transformers you can also have access to the logits:
loss, logits = outputs[:2]

# And even the attention weights if you configure the model to output them (and other outputs too, see the docstrings and documentation)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs
```

### Serialization

Breaking change in the `from_pretrained()`method:

1. Models are now set in evaluation mode by default when instantiated with the `from_pretrained()` method. To train them don't forget to set them back in training mode (`model.train()`) to activate the dropout modules.

2. The additional `*input` and `**kwargs` arguments supplied to the `from_pretrained()` method used to be directly passed to the underlying model's class `__init__()` method. They are now used to update the model configuration attribute instead which can break derived model classes build based on the previous `BertForSequenceClassification` examples. We are working on a way to mitigate this breaking change in [#866](https://github.com/huggingface/pytorch-transformers/pull/866) by forwarding the the model `__init__()` method (i) the provided positional arguments and (ii) the keyword arguments which do not match any configuration class attributes.

Also, while not a breaking change, the serialization methods have been standardized and you probably should switch to the new method `save_pretrained(save_directory)` if you were using any other serialization method before.

Here is an example:

```python
### Let's load a model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

### Do some stuff to our model and tokenizer
# Ex: add new tokens to the vocabulary and embeddings of our model
tokenizer.add_tokens(['[SPECIAL_TOKEN_1]', '[SPECIAL_TOKEN_2]'])
model.resize_token_embeddings(len(tokenizer))
# Train our model
train(model)

### Now let's save our model and tokenizer to a directory
model.save_pretrained('./my_saved_model_directory/')
tokenizer.save_pretrained('./my_saved_model_directory/')

### Reload the model and the tokenizer
model = BertForSequenceClassification.from_pretrained('./my_saved_model_directory/')
tokenizer = BertTokenizer.from_pretrained('./my_saved_model_directory/')
```

### Optimizers: BertAdam & OpenAIAdam are now AdamW, schedules are standard PyTorch schedules

The two optimizers previously included, `BertAdam` and `OpenAIAdam`, have been replaced by a single `AdamW` optimizer which has a few differences:

- it only implements weights decay correction,
- schedules are now externals (see below),
- gradient clipping is now also external (see below).

The new optimizer `AdamW` matches PyTorch `Adam` optimizer API and let you use standard PyTorch or apex methods for the schedule and clipping.

The schedules are now standard [PyTorch learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) and not part of the optimizer anymore.

Here is a conversion examples from `BertAdam` with a linear warmup and decay schedule to `AdamW` and the same schedule:

```python
# Parameters:
lr = 1e-3
max_grad_norm = 1.0
num_total_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

### Previously BertAdam optimizer was instantiated like this:
optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, t_total=num_total_steps)
### and used like this:
for batch in train_data:
    loss = model(batch)
    loss.backward()
    optimizer.step()

### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
### and used like this:
for batch in train_data:
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## Citation

At the moment, there is no paper associated to PyTorch-Transformers but we are working on preparing one. In the meantime, please include a mention of the library and a link to the present repository if you use this work in a published or open-source project.
