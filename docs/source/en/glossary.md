<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Glossary

This glossary defines general machine learning and 🤗 Transformers terms to help you better understand the
documentation.

## A

### attention mask

The attention mask is an optional argument used when batching sequences together.

<Youtube id="M6adb1j2jPI"/>

This argument indicates to the model which tokens should be attended to, and which should not.

For example, consider these two sequences:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence_a = "This is a short sequence."
>>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

>>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
>>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

The encoded versions have different lengths:

```python
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

Therefore, we can't put them together in the same tensor as-is. The first sequence needs to be padded up to the length
of the second one, or the second one needs to be truncated down to the length of the first one.

In the first case, the list of IDs will be extended by the padding indices. We can pass a list to the tokenizer and ask
it to pad like this:

```python
>>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```

We can see that 0s have been added on the right of the first sentence to make it the same length as the second one:

```python
>>> padded_sequences["input_ids"]
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

This can then be converted into a tensor in PyTorch or TensorFlow. The attention mask is a binary tensor indicating the
position of the padded indices so that the model does not attend to them. For the [`BertTokenizer`], `1` indicates a
value that should be attended to, while `0` indicates a padded value. This attention mask is in the dictionary returned
by the tokenizer under the key "attention_mask":

```python
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### autoencoding models

See [encoder models](#encoder-models) and [masked language modeling](#masked-language-modeling-mlm)

### autoregressive models

See [causal language modeling](#causal-language-modeling) and [decoder models](#decoder-models)

## B

### backbone

The backbone is the network (embeddings and layers) that outputs the raw hidden states or features. It is usually connected to a [head](#head) which accepts the features as its input to make a prediction. For example, [`ViTModel`] is a backbone without a specific head on top. Other models can also use [`VitModel`] as a backbone such as [DPT](model_doc/dpt).

## C

### causal language modeling

A pretraining task where the model reads the texts in order and has to predict the next word. It's usually done by
reading the whole sentence but using a mask inside the model to hide the future tokens at a certain timestep.

### channel

Color images are made up of some combination of values in three channels: red, green, and blue (RGB) and grayscale images only have one channel. In 🤗 Transformers, the channel can be the first or last dimension of an image's tensor: [`n_channels`, `height`, `width`] or [`height`, `width`, `n_channels`].

### connectionist temporal classification (CTC)

An algorithm which allows a model to learn without knowing exactly how the input and output are aligned; CTC calculates the distribution of all possible outputs for a given input and chooses the most likely output from it. CTC is commonly used in speech recognition tasks because speech doesn't always cleanly align with the transcript for a variety of reasons such as a speaker's different speech rates.

### convolution

A type of layer in a neural network where the input matrix is multiplied element-wise by a smaller matrix (kernel or filter) and the values are summed up in a new matrix. This is known as a convolutional operation which is repeated over the entire input matrix. Each operation is applied to a different segment of the input matrix. Convolutional neural networks (CNNs) are commonly used in computer vision.

## D

### DataParallel (DP)

Parallelism technique for training on multiple GPUs where the same setup is replicated multiple times, with each instance 
receiving a distinct data slice. The processing is done in parallel and all setups are synchronized at the end of each training step.

Learn more about how DataParallel works [here](perf_train_gpu_many#dataparallel-vs-distributeddataparallel).

### decoder input IDs

This input is specific to encoder-decoder models, and contains the input IDs that will be fed to the decoder. These
inputs should be used for sequence to sequence tasks, such as translation or summarization, and are usually built in a
way specific to each model.

Most encoder-decoder models (BART, T5) create their `decoder_input_ids` on their own from the `labels`. In such models,
passing the `labels` is the preferred way to handle training.

Please check each model's docs to see how they handle these input IDs for sequence to sequence training.

### decoder models

Also referred to as autoregressive models, decoder models involve a pretraining task (called causal language modeling) where the model reads the texts in order and has to predict the next word. It's usually done by
reading the whole sentence with a mask to hide future tokens at a certain timestep.

<Youtube id="d_ixlCubqQw"/>

### deep learning (DL)

Machine learning algorithms which uses neural networks with several layers.

## E

### encoder models

Also known as autoencoding models, encoder models take an input (such as text or images) and transform them into a condensed numerical representation called an embedding. Oftentimes, encoder models are pretrained using techniques like [masked language modeling](#masked-language-modeling-mlm), which masks parts of the input sequence and forces the model to create more meaningful representations.

<Youtube id="H39Z_720T5s"/>

## F

### feature extraction

The process of selecting and transforming raw data into a set of features that are more informative and useful for machine learning algorithms. Some examples of feature extraction include transforming raw text into word embeddings and extracting important features such as edges or shapes from image/video data.

### feed forward chunking

In each residual attention block in transformers the self-attention layer is usually followed by 2 feed forward layers.
The intermediate embedding size of the feed forward layers is often bigger than the hidden size of the model (e.g., for
`google-bert/bert-base-uncased`).

For an input of size `[batch_size, sequence_length]`, the memory required to store the intermediate feed forward
embeddings `[batch_size, sequence_length, config.intermediate_size]` can account for a large fraction of the memory
use. The authors of [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) noticed that since the
computation is independent of the `sequence_length` dimension, it is mathematically equivalent to compute the output
embeddings of both feed forward layers `[batch_size, config.hidden_size]_0, ..., [batch_size, config.hidden_size]_n`
individually and concat them afterward to `[batch_size, sequence_length, config.hidden_size]` with `n = sequence_length`, which trades increased computation time against reduced memory use, but yields a mathematically
**equivalent** result.

For models employing the function [`apply_chunking_to_forward`], the `chunk_size` defines the number of output
embeddings that are computed in parallel and thus defines the trade-off between memory and time complexity. If
`chunk_size` is set to 0, no feed forward chunking is done.

### finetuned models

Finetuning is a form of transfer learning which involves taking a pretrained model, freezing its weights, and replacing the output layer with a newly added [model head](#head). The model head is trained on your target dataset.

See the [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training) tutorial for more details, and learn how to fine-tune models with 🤗 Transformers.

## H

### head

The model head refers to the last layer of a neural network that accepts the raw hidden states and projects them onto a different dimension. There is a different model head for each task. For example:

  * [`GPT2ForSequenceClassification`] is a sequence classification head - a linear layer - on top of the base [`GPT2Model`].
  * [`ViTForImageClassification`] is an image classification head - a linear layer on top of the final hidden state of the `CLS` token - on top of the base [`ViTModel`].
  * [`Wav2Vec2ForCTC`] is a language modeling head with [CTC](#connectionist-temporal-classification-ctc) on top of the base [`Wav2Vec2Model`].

## I

### image patch

Vision-based Transformers models split an image into smaller patches which are linearly embedded, and then passed as a sequence to the model. You can find the `patch_size` - or resolution - of the model in its configuration.

### inference

Inference is the process of evaluating a model on new data after training is complete. See the [Pipeline for inference](https://huggingface.co/docs/transformers/pipeline_tutorial) tutorial to learn how to perform inference with 🤗 Transformers.

### input IDs

The input ids are often the only required parameters to be passed to the model as input. They are token indices,
numerical representations of tokens building the sequences that will be used as input by the model.

<Youtube id="VFp38yj8h3A"/>

Each tokenizer works differently but the underlying mechanism remains the same. Here's an example using the BERT
tokenizer, which is a [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) tokenizer:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence = "A Titan RTX has 24GB of VRAM"
```

The tokenizer takes care of splitting the sequence into tokens available in the tokenizer vocabulary.

```python
>>> tokenized_sequence = tokenizer.tokenize(sequence)
```

The tokens are either words or subwords. Here for instance, "VRAM" wasn't in the model vocabulary, so it's been split
in "V", "RA" and "M". To indicate those tokens are not separate words but parts of the same word, a double-hash prefix
is added for "RA" and "M":

```python
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

These tokens can then be converted into IDs which are understandable by the model. This can be done by directly feeding the sentence to the tokenizer, which leverages the Rust implementation of [🤗 Tokenizers](https://github.com/huggingface/tokenizers) for peak performance.

```python
>>> inputs = tokenizer(sequence)
```

The tokenizer returns a dictionary with all the arguments necessary for its corresponding model to work properly. The
token indices are under the key `input_ids`:

```python
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

Note that the tokenizer automatically adds "special tokens" (if the associated model relies on them) which are special
IDs the model sometimes uses.

If we decode the previous sequence of ids,

```python
>>> decoded_sequence = tokenizer.decode(encoded_sequence)
```

we will see

```python
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

because this is the way a [`BertModel`] is going to expect its inputs.

## L

### labels

The labels are an optional argument which can be passed in order for the model to compute the loss itself. These labels
should be the expected prediction of the model: it will use the standard loss in order to compute the loss between its
predictions and the expected value (the label).

These labels are different according to the model head, for example:

- For sequence classification models, ([`BertForSequenceClassification`]), the model expects a tensor of dimension
  `(batch_size)` with each value of the batch corresponding to the expected label of the entire sequence.
- For token classification models, ([`BertForTokenClassification`]), the model expects a tensor of dimension
  `(batch_size, seq_length)` with each value corresponding to the expected label of each individual token.
- For masked language modeling, ([`BertForMaskedLM`]), the model expects a tensor of dimension `(batch_size,
  seq_length)` with each value corresponding to the expected label of each individual token: the labels being the token
  ID for the masked token, and values to be ignored for the rest (usually -100).
- For sequence to sequence tasks, ([`BartForConditionalGeneration`], [`MBartForConditionalGeneration`]), the model
  expects a tensor of dimension `(batch_size, tgt_seq_length)` with each value corresponding to the target sequences
  associated with each input sequence. During training, both BART and T5 will make the appropriate
  `decoder_input_ids` and decoder attention masks internally. They usually do not need to be supplied. This does not
  apply to models leveraging the Encoder-Decoder framework.
- For image classification models, ([`ViTForImageClassification`]), the model expects a tensor of dimension
  `(batch_size)` with each value of the batch corresponding to the expected label of each individual image.
- For semantic segmentation models, ([`SegformerForSemanticSegmentation`]), the model expects a tensor of dimension
  `(batch_size, height, width)` with each value of the batch corresponding to the expected label of each individual pixel.
- For object detection models, ([`DetrForObjectDetection`]), the model expects a list of dictionaries with a
  `class_labels` and `boxes` key where each value of the batch corresponds to the expected label and number of bounding boxes of each individual image.
- For automatic speech recognition models, ([`Wav2Vec2ForCTC`]), the model expects a tensor of dimension `(batch_size,
  target_length)` with each value corresponding to the expected label of each individual token.
  
<Tip>

Each model's labels may be different, so be sure to always check the documentation of each model for more information
about their specific labels!

</Tip>

The base models ([`BertModel`]) do not accept labels, as these are the base transformer models, simply outputting
features.

### large language models (LLM)

A generic term that refers to transformer language models (GPT-3, BLOOM, OPT) that were trained on a large quantity of data. These models also tend to have a large number of learnable parameters (e.g. 175 billion for GPT-3).

## M

### masked language modeling (MLM)

A pretraining task where the model sees a corrupted version of the texts, usually done by
masking some tokens randomly, and has to predict the original text.

### multimodal

A task that combines texts with another kind of inputs (for instance images).

## N

### Natural language generation (NLG)

All tasks related to generating text (for instance, [Write With Transformers](https://transformer.huggingface.co/), translation).

### Natural language processing (NLP)

A generic way to say "deal with texts".

### Natural language understanding (NLU)

All tasks related to understanding what is in a text (for instance classifying the
whole text, individual words).

## P

### pipeline

A pipeline in 🤗 Transformers is an abstraction referring to a series of steps that are executed in a specific order to preprocess and transform data and return a prediction from a model. Some example stages found in a pipeline might be data preprocessing, feature extraction, and normalization.

For more details, see [Pipelines for inference](https://huggingface.co/docs/transformers/pipeline_tutorial).

### PipelineParallel (PP)

Parallelism technique in which the model is split up vertically (layer-level) across multiple GPUs, so that only one or 
several layers of the model are placed on a single GPU. Each GPU processes in parallel different stages of the pipeline 
and working on a small chunk of the batch. Learn more about how PipelineParallel works [here](perf_train_gpu_many#from-naive-model-parallelism-to-pipeline-parallelism).

### pixel values

A tensor of the numerical representations of an image that is passed to a model. The pixel values have a shape of [`batch_size`, `num_channels`, `height`, `width`], and are generated from an image processor.

### pooling

An operation that reduces a matrix into a smaller matrix, either by taking the maximum or average of the pooled dimension(s). Pooling layers are commonly found between convolutional layers to downsample the feature representation.

### position IDs

Contrary to RNNs that have the position of each token embedded within them, transformers are unaware of the position of
each token. Therefore, the position IDs (`position_ids`) are used by the model to identify each token's position in the
list of tokens.

They are an optional parameter. If no `position_ids` are passed to the model, the IDs are automatically created as
absolute positional embeddings.

Absolute positional embeddings are selected in the range `[0, config.max_position_embeddings - 1]`. Some models use
other types of positional embeddings, such as sinusoidal position embeddings or relative position embeddings.

### preprocessing

The task of preparing raw data into a format that can be easily consumed by machine learning models. For example, text is typically preprocessed by tokenization. To gain a better idea of what preprocessing looks like for other input types, check out the [Preprocess](https://huggingface.co/docs/transformers/preprocessing) tutorial.

### pretrained model

A model that has been pretrained on some data (for instance all of Wikipedia). Pretraining methods involve a
self-supervised objective, which can be reading the text and trying to predict the next word (see [causal language
modeling](#causal-language-modeling)) or masking some words and trying to predict them (see [masked language
modeling](#masked-language-modeling-mlm)). 

Speech and vision models have their own pretraining objectives. For example, Wav2Vec2 is a speech model pretrained on a contrastive task which requires the model to identify the "true" speech representation from a set of "false" speech representations. On the other hand, BEiT is a vision model pretrained on a masked image modeling task which masks some of the image patches and requires the model to predict the masked patches (similar to the masked language modeling objective).

## R

### recurrent neural network (RNN)

A type of model that uses a loop over a layer to process texts.

### representation learning

A subfield of machine learning which focuses on learning meaningful representations of raw data. Some examples of representation learning techniques include word embeddings, autoencoders, and Generative Adversarial Networks (GANs).

## S

### sampling rate

A measurement in hertz of the number of samples (the audio signal) taken per second. The sampling rate is a result of discretizing a continuous signal such as speech.

### self-attention

Each element of the input finds out which other elements of the input they should attend to.

### self-supervised learning 

A category of machine learning techniques in which a model creates its own learning objective from unlabeled data. It differs from [unsupervised learning](#unsupervised-learning) and [supervised learning](#supervised-learning) in that the learning process is supervised, but not explicitly from the user. 

One example of self-supervised learning is [masked language modeling](#masked-language-modeling-mlm), where a model is passed sentences with a proportion of its tokens removed and learns to predict the missing tokens.

### semi-supervised learning

A broad category of machine learning training techniques that leverages a small amount of labeled data with a larger quantity of unlabeled data to improve the accuracy of a model, unlike [supervised learning](#supervised-learning) and [unsupervised learning](#unsupervised-learning).

An example of a semi-supervised learning approach is "self-training", in which a model is trained on labeled data, and then used to make predictions on the unlabeled data. The portion of the unlabeled data that the model predicts with the most confidence gets added to the labeled dataset and used to retrain the model.

### sequence-to-sequence (seq2seq)

Models that generate a new sequence from an input, like translation models, or summarization models (such as
[Bart](model_doc/bart) or [T5](model_doc/t5)).

### Sharded DDP

Another name for the foundational [ZeRO](#zero-redundancy-optimizer-zero) concept as used by various other implementations of ZeRO.

### stride

In [convolution](#convolution) or [pooling](#pooling), the stride refers to the distance the kernel is moved over a matrix. A stride of 1 means the kernel is moved one pixel over at a time, and a stride of 2 means the kernel is moved two pixels over at a time.

### supervised learning

A form of model training that directly uses labeled data to correct and instruct model performance. Data is fed into the model being trained, and its predictions are compared to the known labels. The model updates its weights based on how incorrect its predictions were, and the process is repeated to optimize model performance.

## T

### Tensor Parallelism (TP)

Parallelism technique for training on multiple GPUs in which each tensor is split up into multiple chunks, so instead of 
having the whole tensor reside on a single GPU, each shard of the tensor resides on its designated GPU. Shards gets 
processed separately and in parallel on different GPUs and the results are synced at the end of the processing step. 
This is what is sometimes called horizontal parallelism, as the splitting happens on horizontal level.
Learn more about Tensor Parallelism [here](perf_train_gpu_many#tensor-parallelism).

### token

A part of a sentence, usually a word, but can also be a subword (non-common words are often split in subwords) or a
punctuation symbol.

### token Type IDs

Some models' purpose is to do classification on pairs of sentences or question answering.

<Youtube id="0u3ioSwev3s"/>

These require two different sequences to be joined in a single "input_ids" entry, which usually is performed with the
help of special tokens, such as the classifier (`[CLS]`) and separator (`[SEP]`) tokens. For example, the BERT model
builds its two sequence input as such:

```python
>>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

We can use our tokenizer to automatically generate such a sentence by passing the two sequences to `tokenizer` as two
arguments (and not a list, like before) like this:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
>>> decoded = tokenizer.decode(encoded_dict["input_ids"])
```

which will return:

```python
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

This is enough for some models to understand where one sequence ends and where another begins. However, other models,
such as BERT, also deploy token type IDs (also called segment IDs). They are represented as a binary mask identifying
the two types of sequence in the model.

The tokenizer returns this mask as the "token_type_ids" entry:

```python
>>> encoded_dict["token_type_ids"]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

The first sequence, the "context" used for the question, has all its tokens represented by a `0`, whereas the second
sequence, corresponding to the "question", has all its tokens represented by a `1`.

Some models, like [`XLNetModel`] use an additional token represented by a `2`.

### transfer learning

A technique that involves taking a pretrained model and adapting it to a dataset specific to your task. Instead of training a model from scratch, you can leverage knowledge obtained from an existing model as a starting point. This speeds up the learning process and reduces the amount of training data needed.

### transformer

Self-attention based deep learning model architecture.

## U

### unsupervised learning

A form of model training in which data provided to the model is not labeled. Unsupervised learning techniques leverage statistical information of the data distribution to find patterns useful for the task at hand.

## Z

### Zero Redundancy Optimizer (ZeRO)

Parallelism technique which performs sharding of the tensors somewhat similar to [TensorParallel](#tensor-parallelism-tp), 
except the whole tensor gets reconstructed in time for a forward or backward computation, therefore the model doesn't need 
to be modified. This method also supports various offloading techniques to compensate for limited GPU memory. 
Learn more about ZeRO [here](perf_train_gpu_many#zero-data-parallelism).