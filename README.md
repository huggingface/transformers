<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
</h3>

🤗 Transformers (formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability between PyTorch & TensorFlow 2.0.

### Recent contributors
[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/0)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/0)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/1)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/1)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/2)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/2)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/3)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/3)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/4)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/4)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/5)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/5)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/6)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/6)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/7)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/7)

### Features
- High performance on NLU and NLG tasks
- Low barrier to entry for educators and practitioners

State-of-the-art NLP for everyone
- Deep learning researchers
- Hands-on practitioners
- AI/ML/NLP teachers and educators

Lower compute costs, smaller carbon footprint
- Researchers can share trained models instead of always retraining
- Practitioners can reduce compute time and production costs
- Dozens of architectures with over 1,000 pretrained models, some in more than 100 languages

Choose the right framework for every part of a model's lifetime
- Train state-of-the-art models in 3 lines of code
- Deep interoperability between TensorFlow 2.0 and PyTorch models
- Move a single model between TF2.0/PyTorch frameworks at will
- Seamlessly pick the right framework for training, evaluation, production


| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Model architectures](#model-architectures) | Architectures (with pretrained weights) |
| [Online demo](#online-demo) | Experimenting with this repo’s text generation capabilities |
| [Quick tour: Usage](#quick-tour) | Tokenizers & models usage: Bert and GPT-2 |
| [Quick tour: TF 2.0 and PyTorch ](#Quick-tour-TF-20-training-and-PyTorch-interoperability) | Train a TF 2.0 model in 10 lines of code, load it in PyTorch |
| [Quick tour: pipelines](#quick-tour-of-pipelines) | Using Pipelines: Wrapper around tokenizer and models to use finetuned models |
| [Quick tour: Fine-tuning/usage scripts](#quick-tour-of-the-fine-tuningusage-scripts) | Using provided scripts: GLUE, SQuAD and Text generation |
| [Quick tour: Share your models ](#Quick-tour-of-model-sharing) | Upload and share your fine-tuned models with the community |
| [Migrating from pytorch-transformers to transformers](#Migrating-from-pytorch-transformers-to-transformers) | Migrating your code from pytorch-transformers to transformers |
| [Migrating from pytorch-pretrained-bert to pytorch-transformers](#Migrating-from-pytorch-pretrained-bert-to-transformers) | Migrating your code from pytorch-pretrained-bert to transformers |
| [Documentation](https://huggingface.co/transformers/) | Full API documentation and more |

## Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples) and TensorFlow 2.0.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Create a virtual environment with the version of Python you're going to use and activate it.

Now, if you want to use 🤗 Transformers, you can install it with pip. If you'd like to play with the examples, you must install it from source.

### With pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

### From source

Here also, you first need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

When you update the repository, you should upgrade the transformers installation and its dependencies as follows:

```bash
git pull
pip install --upgrade .
```

### Run the examples

Examples are included in the repository but are not shipped with the library.

Therefore, in order to run the latest versions of the examples, you need to install from source, as described above.

Look at the [README](https://github.com/huggingface/transformers/blob/master/examples/README.md) for how to run examples.

### Tests

A series of tests are included for the library and for some example scripts. Library tests can be found in the [tests folder](https://github.com/huggingface/transformers/tree/master/tests) and examples tests in the [examples folder](https://github.com/huggingface/transformers/tree/master/examples).

Depending on which framework is installed (TensorFlow 2.0 and/or PyTorch), the irrelevant tests will be skipped. Ensure that both frameworks are installed if you want to execute all tests.

Here's the easiest way to run tests for the library:

```bash
pip install -e ".[testing]"
make test
```

and for the examples:

```bash
pip install -e ".[testing]"
pip install -r examples/requirements.txt
make test-examples
```

For details, refer to the [contributing guide](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md#tests).

### Do you want to run a Transformer model on a mobile device?

You should check out our [`swift-coreml-transformers`](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains a set of tools to convert PyTorch or TensorFlow 2.0 trained Transformer models (currently contains `GPT-2`, `DistilGPT-2`, `BERT`, and `DistilBERT`) to CoreML models that run on iOS devices.

At some point in the future, you'll be able to seamlessly move from pre-training or fine-tuning models to productizing them in CoreML, or prototype a model or an app in CoreML then research its hyperparameters or architecture from TensorFlow 2.0 and/or PyTorch. Super exciting!

## Model architectures

🤗 Transformers currently provides the following NLU/NLG architectures:

1. **[BERT](https://huggingface.co/transformers/model_doc/bert.html)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[GPT](https://huggingface.co/transformers/model_doc/gpt.html)** (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. **[GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. **[Transformer-XL](https://huggingface.co/transformers/model_doc/transformerxl.html)** (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. **[XLNet](https://huggingface.co/transformers/model_doc/xlnet.html)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. **[XLM](https://huggingface.co/transformers/model_doc/xlm.html)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
9. **[CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)** (from Salesforce) released with the paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
10. **[CamemBERT](https://huggingface.co/transformers/model_doc/camembert.html)** (from Inria/Facebook/Sorbonne) released with the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
11. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
12. **[T5](https://huggingface.co/transformers/model_doc/t5.html)** (from Google AI) released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
13. **[XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)** (from Facebook AI), released together with the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
14. **[MMBT](https://github.com/facebookresearch/mmbt/)** (from Facebook), released together with the paper a [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/pdf/1909.02950.pdf) by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
15. **[FlauBERT](https://huggingface.co/transformers/model_doc/flaubert.html)** (from CNRS) released with the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
16. **[BART](https://huggingface.co/transformers/model_doc/bart.html)** (from Facebook) released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
17. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
18. **[DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html)** (from Microsoft Research) released with the paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
19. **[Reformer](https://huggingface.co/transformers/model_doc/reformer.html)** (from Google Research) released with the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
20. **[MarianMT](https://huggingface.co/transformers/model_doc/marian.html)** Machine translation models trained using [OPUS](http://opus.nlpl.eu/) data by Jörg Tiedemann. The [Marian Framework](https://marian-nmt.github.io/) is being developed by the Microsoft Translator Team.
21. **[Longformer](https://huggingface.co/transformers/model_doc/longformer.html)** (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
22. **[Other community models](https://huggingface.co/models)**, contributed by the [community](https://huggingface.co/users).
23. Want to contribute a new model? We have added a **detailed guide and templates** to guide you in the process of adding a new model. You can find them in the [`templates`](./templates) folder of the repository. Be sure to check the [contributing guidelines](./CONTRIBUTING.md) and contact the maintainers or open an issue to collect feedbacks before starting your PR.

These implementations have been tested on several datasets (see the example scripts) and should match the performances of the original implementations (e.g. ~93 F1 on SQuAD for BERT Whole-Word-Masking, ~88 F1 on RocStories for OpenAI GPT, ~18.3 perplexity on WikiText 103 for Transformer-XL, ~0.916 Pearson R coefficient on STS-B for XLNet). You can find more details on the performances in the Examples section of the [documentation](https://huggingface.co/transformers/examples.html).

## Online demo

**[Write With Transformer](https://transformer.huggingface.co)**, built by the Hugging Face team at transformer.huggingface.co, is the official demo of this repo’s text generation capabilities.
You can use it to experiment with completions generated by `GPT2Model`, `TransfoXLModel`, and `XLNetModel`.

> “🦄 Write with transformer is to writing what calculators are to calculus.”

![write_with_transformer](https://transformer.huggingface.co/front/assets/thumbnail-large.png)

## Quick tour

Let's do a very quick overview of the model architectures in 🤗 Transformers. Detailed examples for each model architecture (Bert, GPT, GPT-2, Transformer-XL, XLNet and XLM) can be found in the [full documentation](https://huggingface.co/transformers/).

```python
import torch
from transformers import *

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
         ]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

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
                      BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in BERT_MODEL_CLASSES:
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained(pretrained_weights)

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
    tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load

    # SOTA examples for GLUE, SQUAD, text generation...
```

## Quick tour TF 2.0 training and PyTorch interoperability

Let's do a quick example of how a TensorFlow 2.0 model can be trained in 12 lines of code with 🤗 Transformers and then loaded in PyTorch for fast inspection/tests.

```python
import tensorflow as tf
import tensorflow_datasets
from transformers import *

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
data = tensorflow_datasets.load('glue/mrpc')

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('./save/')
pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
```

## Quick tour of the fine-tuning/usage scripts

**Important**
Before running the fine-tuning scripts, please read the
[instructions](#run-the-examples) on how to
setup your environment to run the examples.

The library comprises several example scripts with SOTA performances for NLU and NLG tasks:

- `run_glue.py`: an example fine-tuning sequence classification models on nine different GLUE tasks (*sequence-level classification*)
- `run_squad.py`: an example fine-tuning question answering models on the question answering dataset SQuAD 2.0 (*token-level classification*)
- `run_ner.py`: an example fine-tuning token classification models on named entity recognition (*token-level classification*)
- `run_generation.py`: an example using GPT, GPT-2, CTRL, Transformer-XL and XLNet for conditional language generation
- other model-specific examples (see the documentation).

Here are three quick usage examples for these scripts:

### `run_glue.py`: Fine-tuning on GLUE tasks for sequence classification

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running any of these GLUE tasks you should download the
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

python ./examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
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

python ./examples/text-classification/run_glue.py \
    --model_name_or_path xlnet-large-cased \
    --do_train  \
    --do_eval   \
    --task_name=sts-b     \
    --data_dir=${GLUE_DIR}/STS-B  \
    --output_dir=./proc_data/sts-b-110   \
    --max_seq_length=128   \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
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
python -m torch.distributed.launch --nproc_per_node 8 ./examples/text-classification/run_glue.py   \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name MRPC \
    --do_train   \
    --do_eval   \
    --data_dir $GLUE_DIR/MRPC/   \
    --max_seq_length 128   \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
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
python -m torch.distributed.launch --nproc_per_node=8 ./examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --per_device_eval_batch_size=3   \
    --per_device_train_batch_size=3   \
```

Training with these hyper-parameters gave us the following results:

```bash
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
{"exact_match": 86.91579943235573, "f1": 93.1532499015869}
```

This is the model provided as `bert-large-uncased-whole-word-masking-finetuned-squad`.

### `run_generation.py`: Text generation with GPT, GPT-2, CTRL, Transformer-XL and XLNet

A conditional generation script is also included to generate text from a prompt.
The generation script includes the [tricks](https://github.com/rusiaaman/XLNet-gen#methodology) proposed by Aman Rusia to get high-quality generation with memory models like Transformer-XL and XLNet (include a predefined text to make short inputs longer).

Here is how to run the script with the small version of OpenAI GPT-2 model:

```shell
python ./examples/text-generation/run_generation.py \
    --model_type=gpt2 \
    --length=20 \
    --model_name_or_path=gpt2 \
```

and from the Salesforce CTRL model:
```shell
python ./examples/text-generation/run_generation.py \
    --model_type=ctrl \
    --length=20 \
    --model_name_or_path=ctrl \
    --temperature=0 \
    --repetition_penalty=1.2 \
```

## Quick tour of model sharing

Starting with `v2.2.2`, you can now upload and share your fine-tuned models with the community, using the <abbr title="Command-line interface">CLI</abbr> that's built-in to the library.

**First, create an account on [https://huggingface.co/join](https://huggingface.co/join)**. Optionally, join an existing organization or create a new one. Then:

```shell
transformers-cli login
# log in using the same credentials as on huggingface.co
```
Upload your model:
```shell
transformers-cli upload ./path/to/pretrained_model/

# ^^ Upload folder containing weights/tokenizer/config
# saved via `.save_pretrained()`

transformers-cli upload ./config.json [--filename folder/foobar.json]

# ^^ Upload a single file
# (you can optionally override its filename, which can be nested inside a folder)
```

If you want your model to be namespaced by your organization name rather than your username, add the following flag to any command:
```shell
--organization organization_name
```

Your model will then be accessible through its identifier, a concatenation of your username (or organization name) and the folder name above:
```python
"username/pretrained_model"
# or if an org:
"organization_name/pretrained_model"
```

**Please add a README.md model card** to the repo under `model_cards/` with: model description, training params (dataset, preprocessing, hardware used, hyperparameters), evaluation results, intended uses & limitations, etc.

Your model now has a page on huggingface.co/models 🔥

Anyone can load it from code:
```python
tokenizer = AutoTokenizer.from_pretrained("namespace/pretrained_model")
model = AutoModel.from_pretrained("namespace/pretrained_model")
```

List all your files on S3:
```shell
transformers-cli s3 ls
```

You can also delete unneeded files:

```shell
transformers-cli s3 rm …
```

## Quick tour of pipelines

New in version `v2.3`: `Pipeline` are high-level objects which automatically handle tokenization, running your data through a transformers model
and outputting the result in a structured object.

You can create `Pipeline` objects for the following down-stream tasks:

 - `feature-extraction`: Generates a tensor representation for the input sequence
 - `ner`: Generates named entity mapping for each word in the input sequence.
 - `sentiment-analysis`: Gives the polarity (positive / negative) of the whole input sequence.
 - `text-classification`: Initialize a `TextClassificationPipeline` directly, or see `sentiment-analysis` for an example.
 - `question-answering`: Provided some context and a question refering to the context, it will extract the answer to the question in the context.
 - `fill-mask`: Takes an input sequence containing a masked token (e.g. `<mask>`) and return list of most probable filled sequences, with their probabilities.
 - `summarization`
 - `translation_xx_to_yy`

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> nlp = pipeline('sentiment-analysis')
>>> nlp('We are very happy to include pipeline into the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9978193640708923}]

# Allocate a pipeline for question-answering
>>> nlp = pipeline('question-answering')
>>> nlp({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline have been included in the huggingface/transformers repository'
... })
{'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}

```

## Migrating from pytorch-transformers to transformers

Here is a quick summary of what you should take care of when migrating from `pytorch-transformers` to `transformers`.

### Positional order of some models' keywords inputs (`attention_mask`, `token_type_ids`...) changed

To be able to use Torchscript (see #1010, #1204 and #1195) the specific order of some models **keywords inputs** (`attention_mask`, `token_type_ids`...) has been changed.

If you used to call the models with keyword names for keyword arguments, e.g. `model(inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)`, this should not cause any change.

If you used to call the models with positional inputs for keyword arguments, e.g. `model(inputs_ids, attention_mask, token_type_ids)`, you may have to double check the exact order of input arguments.


## Migrating from pytorch-pretrained-bert to transformers

Here is a quick summary of what you should take care of when migrating from `pytorch-pretrained-bert` to `transformers`.

### Models always output `tuples`

The main breaking change when migrating from `pytorch-pretrained-bert` to `transformers` is that every model's forward method always outputs a `tuple` with various elements depending on the model and the configuration parameters.

The exact content of the tuples for each model is detailed in the models' docstrings and the [documentation](https://huggingface.co/transformers/).

In pretty much every case, you will be fine by taking the first element of the output as the output you previously used in `pytorch-pretrained-bert`.

Here is a `pytorch-pretrained-bert` to `transformers` conversion example for a `BertForSequenceClassification` classification model:

```python
# Let's load our model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# If you used to have this line in pytorch-pretrained-bert:
loss = model(input_ids, labels=labels)

# Now just use this line in transformers to extract the loss from the output tuple:
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# In transformers you can also have access to the logits:
loss, logits = outputs[:2]

# And even the attention weights if you configure the model to output them (and other outputs too, see the docstrings and documentation)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs
```

### Using hidden states

By enabling the configuration option `output_hidden_states`, it was possible to retrieve the last hidden states of the encoder. In `pytorch-transformers` as well as `transformers` the return value has changed slightly: `all_hidden_states` now also includes the hidden state of the embeddings in addition to those of the encoding layers. This allows users to easily access the embeddings final state.

### Serialization

Breaking change in the `from_pretrained()` method:

1. Models are now set in evaluation mode by default when instantiated with the `from_pretrained()` method. To train them, don't forget to set them back in training mode (`model.train()`) to activate the dropout modules.

2. The additional `*input` and `**kwargs` arguments supplied to the `from_pretrained()` method used to be directly passed to the underlying model's class `__init__()` method. They are now used to update the model configuration attribute instead, which can break derived model classes built based on the previous `BertForSequenceClassification` examples. We are working on a way to mitigate this breaking change in [#866](https://github.com/huggingface/transformers/pull/866) by forwarding the the model's `__init__()` method (i) the provided positional arguments and (ii) the keyword arguments which do not match any configuration class attributes.

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
num_training_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

### Previously BertAdam optimizer was instantiated like this:
optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, t_total=num_training_steps)
### and used like this:
for batch in train_data:
    loss = model(batch)
    loss.backward()
    optimizer.step()

### In Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
### and used like this:
for batch in train_data:
    model.train()
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## Citation

We now have a paper you can cite for the 🤗 Transformers library:
```bibtex
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```
