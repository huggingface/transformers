<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quick tour

[[open-in-colab]]

Get up and running with 🤗 Transformers! Whether you're a developer or an everyday user, this quick tour will help you get started and show you how to use the [`pipeline`] for inference, load a pretrained model and preprocessor with an [AutoClass](./model_doc/auto), and quickly train a model with PyTorch or TensorFlow. If you're a beginner, we recommend checking out our tutorials or [course](https://huggingface.co/course/chapter1/1) next for more in-depth explanations of the concepts introduced here.

Before you begin, make sure you have all the necessary libraries installed:

```bash
!pip install transformers datasets
```

You'll also need to install your preferred machine learning framework:

<frameworkcontent>
<pt>

```bash
pip install torch
```
</pt>
<tf>

```bash
pip install tensorflow
```
</tf>
</frameworkcontent>

## Pipeline

<Youtube id="tiZFewofSLM"/>

The [`pipeline`] is the easiest and fastest way to use a pretrained model for inference. You can use the [`pipeline`] out-of-the-box for many tasks across different modalities, some of which are shown in the table below:

<Tip>

For a complete list of available tasks, check out the [pipeline API reference](./main_classes/pipelines).

</Tip>

| **Task**                     | **Description**                                                                                              | **Modality**    | **Pipeline identifier**                       |
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| Text classification          | assign a label to a given sequence of text                                                                   | NLP             | pipeline(task=“sentiment-analysis”)           |
| Text generation              | generate text given a prompt                                                                                 | NLP             | pipeline(task=“text-generation”)              |
| Summarization                | generate a summary of a sequence of text or document                                                         | NLP             | pipeline(task=“summarization”)                |
| Image classification         | assign a label to an image                                                                                   | Computer vision | pipeline(task=“image-classification”)         |
| Image segmentation           | assign a label to each individual pixel of an image (supports semantic, panoptic, and instance segmentation) | Computer vision | pipeline(task=“image-segmentation”)           |
| Object detection             | predict the bounding boxes and classes of objects in an image                                                | Computer vision | pipeline(task=“object-detection”)             |
| Audio classification         | assign a label to some audio data                                                                            | Audio           | pipeline(task=“audio-classification”)         |
| Automatic speech recognition | transcribe speech into text                                                                                  | Audio           | pipeline(task=“automatic-speech-recognition”) |
| Visual question answering    | answer a question about the image, given an image and a question                                             | Multimodal      | pipeline(task=“vqa”)                          |
| Document question answering  | answer a question about the document, given a document and a question                                        | Multimodal      | pipeline(task="document-question-answering")  |
| Image captioning             | generate a caption for a given image                                                                         | Multimodal      | pipeline(task="image-to-text")                |

Start by creating an instance of [`pipeline`] and specifying a task you want to use it for. In this guide, you'll use the [`pipeline`] for sentiment analysis as an example:



The [`pipeline`] downloads and caches a default [pretrained model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) and tokenizer for sentiment analysis. Now you can use the `classifier` on your target text:



If you have more than one input, pass your inputs as a list to the [`pipeline`] to return a list of dictionaries:



The [`pipeline`] can also iterate over an entire dataset for any task you like. For this example, let's choose automatic speech recognition as our task:



Load an audio dataset (see the 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart#audio) for more details) you'd like to iterate over. For example, load the [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset:



You need to make sure the sampling rate of the dataset matches the sampling 
rate [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) was trained on:



The audio files are automatically loaded and resampled when calling the `"audio"` column.
Extract the raw waveform arrays from the first 4 samples and pass it as a list to the pipeline:



For larger datasets where the inputs are big (like in speech or vision), you'll want to pass a generator instead of a list to load all the inputs in memory. Take a look at the [pipeline API reference](./main_classes/pipelines) for more information.

### Use another model and tokenizer in the pipeline

The [`pipeline`] can accommodate any model from the [Hub](https://huggingface.co/models), making it easy to adapt the [`pipeline`] for other use-cases. For example, if you'd like a model capable of handling French text, use the tags on the Hub to filter for an appropriate model. The top filtered result returns a multilingual [BERT model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) finetuned for sentiment analysis you can use for French text:

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> model_name = "distilbert-base-uncased"
>>> model_name = "gpt2"
```

<frameworkcontent>
<pt>
Use [`AutoModelForSequenceClassification`] and [`AutoTokenizer`] to load the pretrained model and it's associated tokenizer (more on an `AutoClass` in the next section):


</pt>
<tf>
Use [`TFAutoModelForSequenceClassification`] and [`AutoTokenizer`] to load the pretrained model and it's associated tokenizer (more on an `TFAutoClass` in the next section):

```py
>>> 1 == 1
>>> # import tensorflow as tf
>>> # from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModel

>>> # model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> # model = TFAutoModel.from_pretrained(model_name)

```
</tf>
</frameworkcontent>

Specify the model and tokenizer in the [`pipeline`], and now you can apply the `classifier` on French text:



If you can't find a model for your use-case, you'll need to finetune a pretrained model on your data. Take a look at our [finetuning tutorial](./training) to learn how. Finally, after you've finetuned your pretrained model, please consider [sharing](./model_sharing) the model with the community on the Hub to democratize machine learning for everyone! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

Under the hood, the [`AutoModelForSequenceClassification`] and [`AutoTokenizer`] classes work together to power the [`pipeline`] you used above. An [AutoClass](./model_doc/auto) is a shortcut that automatically retrieves the architecture of a pretrained model from its name or path. You only need to select the appropriate `AutoClass` for your task and it's associated preprocessing class. 

Let's return to the example from the previous section and see how you can use the `AutoClass` to replicate the results of the [`pipeline`].

### AutoTokenizer

A tokenizer is responsible for preprocessing text into an array of numbers as inputs to a model. There are multiple rules that govern the tokenization process, including how to split a word and at what level words should be split (learn more about tokenization in the [tokenizer summary](./tokenizer_summary)). The most important thing to remember is you need to instantiate a tokenizer with the same model name to ensure you're using the same tokenization rules a model was pretrained with.

Load a tokenizer with [`AutoTokenizer`]:



Pass your text to the tokenizer:



The tokenizer returns a dictionary containing:

* [input_ids](./glossary#input-ids): numerical representations of your tokens.
* [attention_mask](.glossary#attention-mask): indicates which tokens should be attended to.

A tokenizer can also accept a list of inputs, and pad and truncate the text to return a batch with uniform length:

<frameworkcontent>
<pt>


</pt>
<tf>


</tf>
</frameworkcontent>

<Tip>

Check out the [preprocess](./preprocessing) tutorial for more details about tokenization, and how to use an [`AutoImageProcessor`], [`AutoFeatureExtractor`] and [`AutoProcessor`] to preprocess image, audio, and multimodal inputs.

</Tip>

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers provides a simple and unified way to load pretrained instances. This means you can load an [`AutoModel`] like you would load an [`AutoTokenizer`]. The only difference is selecting the correct [`AutoModel`] for the task. For text (or sequence) classification, you should load [`AutoModelForSequenceClassification`]:



<Tip>

See the [task summary](./task_summary) for tasks supported by an [`AutoModel`] class.

</Tip>

Now pass your preprocessed batch of inputs directly to the model. You just have to unpack the dictionary by adding `**`:



The model outputs the final activations in the `logits` attribute. Apply the softmax function to the `logits` to retrieve the probabilities:


</pt>
<tf>
🤗 Transformers provides a simple and unified way to load pretrained instances. This means you can load an [`TFAutoModel`] like you would load an [`AutoTokenizer`]. The only difference is selecting the correct [`TFAutoModel`] for the task. For text (or sequence) classification, you should load [`TFAutoModelForSequenceClassification`]:



<Tip>

See the [task summary](./task_summary) for tasks supported by an [`AutoModel`] class.

</Tip>

Now pass your preprocessed batch of inputs directly to the model. You can pass the tensors as-is:



The model outputs the final activations in the `logits` attribute. Apply the softmax function to the `logits` to retrieve the probabilities:


</tf>
</frameworkcontent>

<Tip>

All 🤗 Transformers models (PyTorch or TensorFlow) output the tensors *before* the final activation
function (like softmax) because the final activation function is often fused with the loss. Model outputs are special dataclasses so their attributes are autocompleted in an IDE. The model outputs behave like a tuple or a dictionary (you can index with an integer, a slice or a string) in which case, attributes that are None are ignored.

</Tip>

### Save a model

<frameworkcontent>
<pt>
Once your model is fine-tuned, you can save it with its tokenizer using [`PreTrainedModel.save_pretrained`]:



When you are ready to use the model again, reload it with [`PreTrainedModel.from_pretrained`]:


</pt>
<tf>
Once your model is fine-tuned, you can save it with its tokenizer using [`TFPreTrainedModel.save_pretrained`]:



When you are ready to use the model again, reload it with [`TFPreTrainedModel.from_pretrained`]:


</tf>
</frameworkcontent>

One particularly cool 🤗 Transformers feature is the ability to save a model and reload it as either a PyTorch or TensorFlow model. The `from_pt` or `from_tf` parameter can convert the model from one framework to the other:

<frameworkcontent>
<pt>


</pt>
<tf>


</tf>
</frameworkcontent>

## Custom model builds

You can modify the model's configuration class to change how a model is built. The configuration specifies a model's attributes, such as the number of hidden layers or attention heads. You start from scratch when you initialize a model from a custom configuration class. The model attributes are randomly initialized, and you'll need to train the model before you can use it to get meaningful results.

Start by importing [`AutoConfig`], and then load the pretrained model you want to modify. Within [`AutoConfig.from_pretrained`], you can specify the attribute you want to change, such as the number of attention heads:



<frameworkcontent>
<pt>
Create a model from your custom configuration with [`AutoModel.from_config`]:


</pt>
<tf>
Create a model from your custom configuration with [`TFAutoModel.from_config`]:


</tf>
</frameworkcontent>

Take a look at the [Create a custom architecture](./create_a_model) guide for more information about building custom configurations.

## Trainer - a PyTorch optimized training loop

All models are a standard [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) so you can use them in any typical training loop. While you can write your own training loop, 🤗 Transformers provides a [`Trainer`] class for PyTorch, which contains the basic training loop and adds additional functionality for features like distributed training, mixed precision, and more.

Depending on your task, you'll typically pass the following parameters to [`Trainer`]:

1. You'll start with a [`PreTrainedModel`] or a [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module):



2. [`TrainingArguments`] contains the model hyperparameters you can change like learning rate, batch size, and the number of epochs to train for. The default values are used if you don't specify any training arguments:



3. Load a preprocessing class like a tokenizer, image processor, feature extractor, or processor:



4. Load a dataset:



5. Create a function to tokenize the dataset:



   Then apply it over the entire dataset with [`~datasets.Dataset.map`]:



6. A [`DataCollatorWithPadding`] to create a batch of examples from your dataset:



Now gather all these classes in [`Trainer`]:



When you're ready, call [`~Trainer.train`] to start training:



<Tip>

For tasks - like translation or summarization - that use a sequence-to-sequence model, use the [`Seq2SeqTrainer`] and [`Seq2SeqTrainingArguments`] classes instead.

</Tip>

You can customize the training loop behavior by subclassing the methods inside [`Trainer`]. This allows you to customize features such as the loss function, optimizer, and scheduler. Take a look at the [`Trainer`] reference for which methods can be subclassed. 

The other way to customize the training loop is by using [Callbacks](./main_classes/callbacks). You can use callbacks to integrate with other libraries and inspect the training loop to report on progress or stop the training early. Callbacks do not modify anything in the training loop itself. To customize something like the loss function, you need to subclass the [`Trainer`] instead.

## Train with TensorFlow

All models are a standard [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) so they can be trained in TensorFlow with the [Keras](https://keras.io/) API. 🤗 Transformers provides the [`~TFPreTrainedModel.prepare_tf_dataset`] method to easily load your dataset as a `tf.data.Dataset` so you can start training right away with Keras' [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) and [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) methods.

1. You'll start with a [`TFPreTrainedModel`] or a [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model):



2. Load a preprocessing class like a tokenizer, image processor, feature extractor, or processor:



3. Create a function to tokenize the dataset:



4. Apply the tokenizer over the entire dataset with [`~datasets.Dataset.map`] and then pass the dataset and tokenizer to [`~TFPreTrainedModel.prepare_tf_dataset`]. You can also change the batch size and shuffle the dataset here if you'd like:



5. When you're ready, you can call `compile` and `fit` to start training. Note that Transformers models all have a default task-relevant loss function, so you don't need to specify one unless you want to:



## What's next?

Now that you've completed the 🤗 Transformers quick tour, check out our guides and learn how to do more specific things like writing a custom model, fine-tuning a model for a task, and how to train a model with a script. If you're interested in learning more about 🤗 Transformers core concepts, grab a cup of coffee and take a look at our Conceptual Guides!
