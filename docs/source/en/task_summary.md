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

# What 🤗 Transformers can do

🤗 Transformers is a library of pretrained state-of-the-art models for natural language processing (NLP), computer vision, and audio and speech processing tasks. Not only does the library contain Transformer models, but it also has non-Transformer models like modern convolutional networks for computer vision tasks. If you look at some of the most popular consumer products today, like smartphones, apps, and televisions, odds are that some kind of deep learning technology is behind it. Want to remove a background object from a picture taken by your smartphone? This is an example of a panoptic segmentation task (don't worry if you don't know what this means yet, we'll describe it in the following sections!). 

This page provides an overview of the different speech and audio, computer vision, and NLP tasks that can be solved with the 🤗 Transformers library in just three lines of code!

## Audio

Audio and speech processing tasks are a little different from the other modalities mainly because audio as an input is a continuous signal. Unlike text, a raw audio waveform can't be neatly split into discrete chunks the way a sentence can be divided into words. To get around this, the raw audio signal is typically sampled at regular intervals. If you take more samples within an interval, the sampling rate is higher, and the audio more closely resembles the original audio source.

Previous approaches preprocessed the audio to extract useful features from it. It is now more common to start audio and speech processing tasks by directly feeding the raw audio waveform to a feature encoder to extract an audio representation. This simplifies the preprocessing step and allows the model to learn the most essential features.

### Audio classification

Audio classification is a task that labels audio data from a predefined set of classes. It is a broad category with many specific applications, some of which include:

* acoustic scene classification: label audio with a scene label ("office", "beach", "stadium")
* acoustic event detection: label audio with a sound event label ("car horn", "whale calling", "glass breaking")
* tagging: label audio containing multiple sounds (birdsongs, speaker identification in a meeting)
* music classification: label music with a genre label ("metal", "hip-hop", "country")

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### Automatic speech recognition

Automatic speech recognition (ASR) transcribes speech into text. It is one of the most common audio tasks due partly to speech being such a natural form of human communication. Today, ASR systems are embedded in "smart" technology products like speakers, phones, and cars. We can ask our virtual assistants to play music, set reminders, and tell us the weather. 

But one of the key challenges Transformer architectures have helped with is in low-resource languages. By pretraining on large amounts of speech data, finetuning the model on only one hour of labeled speech data in a low-resource language can still produce high-quality results compared to previous ASR systems trained on 100x more labeled data.

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## Computer vision

One of the first and earliest successful computer vision tasks was recognizing images of zip code numbers using a [convolutional neural network (CNN)](glossary#convolution). An image is composed of pixels, and each pixel has a numerical value. This makes it easy to represent an image as a matrix of pixel values. Each particular combination of pixel values describes the colors of an image. 

Two general ways computer vision tasks can be solved are:

1. Use convolutions to learn the hierarchical features of an image from low-level features to high-level abstract things.
2. Split an image into patches and use a Transformer to gradually learn how each image patch is related to each other to form an image. Unlike the bottom-up approach favored by a CNN, this is kind of like starting out with a blurry image and then gradually bringing it into focus.

### Image classification

Image classification labels an entire image from a predefined set of classes. Like most classification tasks, there are many practical use cases for image classification, some of which include:

* healthcare: label medical images to detect disease or monitor patient health
* environment: label satellite images to monitor deforestation, inform wildland management or detect wildfires
* agriculture: label images of crops to monitor plant health or satellite images for land use monitoring 
* ecology: label images of animal or plant species to monitor wildlife populations or track endangered species

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### Object detection

Unlike image classification, object detection identifies multiple objects within an image and the objects' positions in an image (defined by the bounding box). Some example applications of object detection include:

* self-driving vehicles: detect everyday traffic objects such as other vehicles, pedestrians, and traffic lights
* remote sensing: disaster monitoring, urban planning, and weather forecasting
* defect detection: detect cracks or structural damage in buildings, and manufacturing defects

```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### Image segmentation

Image segmentation is a pixel-level task that assigns every pixel in an image to a class. It differs from object detection, which uses bounding boxes to label and predict objects in an image because segmentation is more granular. Segmentation can detect objects at a pixel-level. There are several types of image segmentation:

* instance segmentation: in addition to labeling the class of an object, it also labels each distinct instance of an object ("dog-1", "dog-2")
* panoptic segmentation: a combination of semantic and instance segmentation; it labels each pixel with a semantic class **and** each distinct instance of an object

Segmentation tasks are helpful in self-driving vehicles to create a pixel-level map of the world around them so they can navigate safely around pedestrians and other vehicles. It is also useful for medical imaging, where the task's finer granularity can help identify abnormal cells or organ features. Image segmentation can also be used in ecommerce to virtually try on clothes or create augmented reality experiences by overlaying objects in the real world through your camera.

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### Depth estimation

Depth estimation predicts the distance of each pixel in an image from the camera. This computer vision task is especially important for scene understanding and reconstruction. For example, in self-driving cars, vehicles need to understand how far objects like pedestrians, traffic signs, and other vehicles are to avoid obstacles and collisions. Depth information is also helpful for constructing 3D representations from 2D images and can be used to create high-quality 3D representations of biological structures or buildings.

There are two approaches to depth estimation:

* stereo: depths are estimated by comparing two images of the same image from slightly different angles
* monocular: depths are estimated from a single image

```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## Natural language processing

NLP tasks are among the most common types of tasks because text is such a natural way for us to communicate. To get text into a format recognized by a model, it needs to be tokenized. This means dividing a sequence of text into separate words or subwords (tokens) and then converting these tokens into numbers. As a result, you can represent a sequence of text as a sequence of numbers, and once you have a sequence of numbers, it can be input into a model to solve all sorts of NLP tasks!

### Text classification

Like classification tasks in any modality, text classification labels a sequence of text (it can be sentence-level, a paragraph, or a document) from a predefined set of classes. There are many practical applications for text classification, some of which include:

* sentiment analysis: label text according to some polarity like `positive` or `negative` which can inform and support decision-making in fields like politics, finance, and marketing
* content classification: label text according to some topic to help organize and filter information in news and social media feeds (`weather`, `sports`, `finance`, etc.)

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### Token classification

In any NLP task, text is preprocessed by separating the sequence of text into individual words or subwords. These are known as [tokens](glossary#token). Token classification assigns each token a label from a predefined set of classes. 

Two common types of token classification are:

* named entity recognition (NER): label a token according to an entity category like organization, person, location or date. NER is especially popular in biomedical settings, where it can label genes, proteins, and drug names.
* part-of-speech tagging (POS): label a token according to its part-of-speech like noun, verb, or adjective. POS is useful for helping translation systems understand how two identical words are grammatically different (bank as a noun versus bank as a verb).

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### Question answering

Question answering is another token-level task that returns an answer to a question, sometimes with context (open-domain) and other times without context (closed-domain). This task happens whenever we ask a virtual assistant something like whether a restaurant is open. It can also provide customer or technical support and help search engines retrieve the relevant information you're asking for. 

There are two common types of question answering:

* extractive: given a question and some context, the answer is a span of text from the context the model must extract
* abstractive: given a question and some context, the answer is generated from the context; this approach is handled by the [`Text2TextGenerationPipeline`] instead of the [`QuestionAnsweringPipeline`] shown below


```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### Summarization

Summarization creates a shorter version of a text from a longer one while trying to preserve most of the meaning of the original document. Summarization is a sequence-to-sequence task; it outputs a shorter text sequence than the input. There are a lot of long-form documents that can be summarized to help readers quickly understand the main points. Legislative bills, legal and financial documents, patents, and scientific papers are a few examples of documents that could be summarized to save readers time and serve as a reading aid.

Like question answering, there are two types of summarization:

* extractive: identify and extract the most important sentences from the original text
* abstractive: generate the target summary (which may include new words not in the input document) from the original text; the [`SummarizationPipeline`] uses the abstractive approach

```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### Translation

Translation converts a sequence of text in one language to another. It is important in helping people from different backgrounds communicate with each other, help translate content to reach wider audiences, and even be a learning tool to help people learn a new language. Along with summarization, translation is a sequence-to-sequence task, meaning the model receives an input sequence and returns a target output sequence. 

In the early days, translation models were mostly monolingual, but recently, there has been increasing interest in multilingual models that can translate between many pairs of languages.

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### Language modeling

Language modeling is a task that predicts a word in a sequence of text. It has become a very popular NLP task because a pretrained language model can be finetuned for many other downstream tasks. Lately, there has been a lot of interest in large language models (LLMs) which demonstrate zero- or few-shot learning. This means the model can solve tasks it wasn't explicitly trained to do! Language models can be used to generate fluent and convincing text, though you need to be careful since the text may not always be accurate.

There are two types of language modeling:

* causal: the model's objective is to predict the next token in a sequence, and future tokens are masked

    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

* masked: the model's objective is to predict a masked token in a sequence with full access to the tokens in the sequence
    
    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.224, 'token': 3944, 'token_str': ' tool', 'sequence': 'Hugging Face is a community-based open-source tool for machine learning.'}]
    ```

## Multimodal

Multimodal tasks require a model to process multiple data modalities (text, image, audio, video) to solve a particular problem. Image captioning is an example of a multimodal task where the model takes an image as input and outputs a sequence of text describing the image or some properties of the image. 

Although multimodal models work with different data types or modalities, internally, the preprocessing steps help the model convert all the data types into embeddings (vectors or list of numbers that holds meaningful information about the data). For a task like image captioning, the model learns relationships between image embeddings and text embeddings.

### Document question answering

Document question answering is a task that answers natural language questions from a document. Unlike a token-level question answering task which takes text as input, document question answering takes an image of a document as input along with a question about the document and returns an answer. Document question answering can be used to parse structured documents and extract key information from it. In the example below, the total amount and change due can be extracted from a receipt.

```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/2.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="What is the total amount?",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

Hopefully, this page has given you some more background information about all the types of tasks in each modality and the practical importance of each one. In the next [section](tasks_explained), you'll learn **how** 🤗 Transformers work to solve these tasks.