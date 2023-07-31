<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Visual Question Answering

[[open-in-colab]]

[//]: # (TODO: intro) 
Visual Question Answering is a task ...

This guide illustrates how to:

- Fine-tune [ViLT](../model_doc/vilt) on the [`Graphcore/vqa` dataset](https://huggingface.co/datasets/Graphcore/vqa).
- Use your fine-tuned model for inference.

[//]: # (TODO: add the automated Tip if there are other models that support this task)

[//]: # (TODO: couple of words about ViLT)

Before you begin, make sure you have all the necessary libraries installed. 

```bash
pip install -q transformers datasets
```

We encourage you to share your model with the community. Log in to your Hugging Face account to upload it to the 🤗 Hub.
When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

Let's define the model checkpoint as a global variable.

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## Load the data

In this guide we use a very small sample of the `Graphcore/vqa` dataset, which you can find on 🤗 Hub.

[//]: # (TODO: Explain where this dataset originally comes from? )

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:100]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 100
})
```

Let's take a look at an example to understand the dataset's features:

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

[//]: # (TODO: explain the features. different answers, and weights)
[//]: # (As we can see, the example contains several answers (collected by different human annotators). The answer to a question can be a bit subjective: for instance for the question "where is he looking?", some people annotated this with "down", others with "table", another one with "skateboard", etc. So there's a bit of disambiguity among the annotators :))

Let's see what the image for this example is: 

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

[//]: # (TODO: add the image)

<div class="flex justify-center">
     <img src="" alt="VQA Image Example"/>
</div>

[//]: # (TODO: create label to id mapping)

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()} 
```

Now that we have the mappings....

```python
# replace actual labels with their ids

>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'question_type': Value(dtype='string', id=None),
 'question_id': Value(dtype='int32', id=None),
 'image_id': Value(dtype='string', id=None),
 'answer_type': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## Preprocessing data

```py 
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']    

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k,v in encoding.items():
...           encoding[k] = v.squeeze()
    
...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...       target = torch.zeros(len(id2label))

...       for label, score in zip(labels, scores):
...         target[label] = score
      
...       targets.append(target)

...     encoding["labels"] = targets
    
...     return encoding

>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type', 'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 100
})
```

Data collator 

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## Train the model

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

Training arguments

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_100"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

Trainer

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     tokenizer=processor,
... )

>>> trainer.train() 
Step	Training Loss
50	22.692400
100	6.525900
150	5.436700
200	4.800800
250	4.475900
300	3.639400
350	3.137100
400	2.702400
450	2.366200
500	2.181200
```

Push to Hub
```py
>>> trainer.push_to_hub()
```

## Inference

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_100")
```

```py
example = dataset[0]
image = Image.open(example['image_id'])

question = example['question']

pipe(image, question, top_k=1)
```
