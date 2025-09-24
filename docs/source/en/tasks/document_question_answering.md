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

# Document Question Answering

[[open-in-colab]]

Document Question Answering, also referred to as Document Visual Question Answering, is a task that involves providing
answers to questions posed about document images. The input to models supporting this task is typically a combination of an image and
a question, and the output is an answer expressed in natural language. These models utilize multiple modalities, including
text, the positions of words (bounding boxes), and the image itself.

This guide illustrates how to:

- Fine-tune [LayoutLMv2](../model_doc/layoutlmv2) on the [DocVQA dataset](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut).
- Use your fine-tuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/image-to-text)

</Tip>

LayoutLMv2 solves the document question-answering task by adding a question-answering head on top of the final hidden
states of the tokens, to predict the positions of the start and end tokens of the
answer. In other words, the problem is treated as extractive question answering: given the context, extract which piece
of information answers the question. The context comes from the output of an OCR engine, here it is Google's Tesseract.

Before you begin, make sure you have all the necessary libraries installed. LayoutLMv2 depends on detectron2, torchvision and tesseract.

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

Once you have installed all of the dependencies, restart your runtime.

We encourage you to share your model with the community. Log in to your Hugging Face account to upload it to the 🤗 Hub.
When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

Let's define some global variables.

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## Load the data

In this guide we use a small sample of preprocessed DocVQA that you can find on 🤗 Hub. If you'd like to use the full
DocVQA dataset, you can register and download it on [DocVQA homepage](https://rrc.cvc.uab.es/?ch=17). If you do so, to
proceed with this guide check out [how to load files into a 🤗 dataset](https://huggingface.co/docs/datasets/loading#local-and-remote-files).

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 200
    })
})
```

As you can see, the dataset is split into train and test sets already. Take a look at a random example to familiarize
yourself with the features.

```py
>>> dataset["train"].features
```

Here's what the individual fields represent:
* `id`: the example's id
* `image`: a PIL.Image.Image object containing the document image
* `query`: the question string - natural language asked question, in several languages
* `answers`: a list of correct answers provided by human annotators
* `words` and `bounding_boxes`: the results of OCR, which we will not use here
* `answer`: an answer matched by a different model which we will not use here

Let's leave only English questions, and drop the `answer` feature which appears to contain predictions by another model.
We'll also take the first of the answers from the set provided by the annotators. Alternatively, you can randomly sample it.

```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

Note that the LayoutLMv2 checkpoint that we use in this guide has been trained with `max_position_embeddings = 512` (you can
find this information in the [checkpoint's `config.json` file](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18)).
We can truncate the examples but to avoid the situation where the answer might be at the end of a large document and end up truncated,
here we'll remove the few examples where the embedding is likely to end up longer than 512.
If most of the documents in your dataset are long, you can implement a sliding window strategy - check out [this notebook](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb) for details.

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

At this point let's also remove the OCR features from this dataset. These are a result of OCR for fine-tuning a different
model. They would still require some processing if we wanted to use them, as they do not match the input requirements
of the model we use in this guide. Instead, we can use the [`LayoutLMv2Processor`] on the original data for both OCR and
tokenization. This way we'll get the inputs that match model's expected input. If you want to process images manually,
check out the [`LayoutLMv2` model documentation](../model_doc/layoutlmv2) to learn what input format the model expects.

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

Finally, the data exploration won't be complete if we don't peek at an image example.

```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="DocVQA Image Example"/>
 </div>

## Preprocess the data

The Document Question Answering task is a multimodal task, and you need to make sure that the inputs from each modality
are preprocessed according to the model's expectations. Let's start by loading the [`LayoutLMv2Processor`], which internally combines an image processor that can handle image data and a tokenizer that can encode text data.

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### Preprocessing document images

First, let's prepare the document images for the model with the help of the `image_processor` from the processor.
By default, image processor resizes the images to 224x224, makes sure they have the correct order of color channels,
applies OCR with tesseract to get words and normalized bounding boxes. In this tutorial, all of these defaults are exactly what we need.
Write a function that applies the default image processing to a batch of images and returns the results of OCR.

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

To apply this preprocessing to the entire dataset in a fast way, use [`~datasets.Dataset.map`].

```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### Preprocessing text data

Once we have applied OCR to the images, we need to encode the text part of the dataset to prepare it for the model.
This involves converting the words and boxes that we got in the previous step to token-level `input_ids`, `attention_mask`,
`token_type_ids` and `bbox`. For preprocessing text, we'll need the `tokenizer` from the processor.

```py
>>> tokenizer = processor.tokenizer
```

On top of the preprocessing mentioned above, we also need to add the labels for the model. For `xxxForQuestionAnswering` models
in 🤗 Transformers, the labels consist of the `start_positions` and `end_positions`, indicating which token is at the
start and which token is at the end of the answer.

Let's start with that. Define a helper function that can find a sublist (the answer split into words) in a larger list (the words list).

This function will take two lists as input, `words_list` and `answer_list`. It will then iterate over the `words_list` and check
if the current word in the `words_list` (words_list[i]) is equal to the first word of answer_list (answer_list[0]) and if
the sublist of `words_list` starting from the current word and of the same length as `answer_list` is equal `to answer_list`.
If this condition is true, it means that a match has been found, and the function will record the match, its starting index (idx),
and its ending index (idx + len(answer_list) - 1). If more than one match was found, the function will return only the first one.
If no match is found, the function returns (`None`, 0, and 0).

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

To illustrate how this function finds the position of the answer, let's use it on an example:

```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  Who is in  cc in this letter?
Words: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorming', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast', 'buming', 'cigarette.', '«novel', 'cigarette', 'constructions', 'that', 'permit', 'a', 'significant', 'reduction', 'iretobacco', 'weight', 'while', 'maintaining', 'smoking', 'mechanics', 'and', 'visual', 'characteristics.', 'higher', 'basis', 'weight', 'paper:', 'potential', 'reduction', 'in', 'tobacco', 'weight.', '«more', 'rigid', 'tobacco', 'column;', 'stiffing', 'agent', 'for', 'tobacco;', 'e.g.', 'starch', '*colored', 'tow', 'and', 'cigarette', 'papers;', 'seasonal', 'promotions,', 'e.g.', 'pastel', 'colored', 'cigarettes', 'for', 'easter', 'or', 'in', 'an', 'ebony', 'and', 'ivory', 'brand', 'containing', 'a', 'mixture', 'of', 'all', 'black', '(black', 'paper', 'and', 'tow)', 'and', 'ail', 'white', 'cigarettes.', '499150498']
Answer:  T.F. Riehl
start_index 17
end_index 18
```

Once examples are encoded, however, they will look like this:

```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
[CLS] who is in cc in this letter? [SEP] wie baw brown & williamson tobacco corporation research & development ...
```

We'll need to find the position of the answer in the encoded input.
* `token_type_ids` tells us which tokens are part of the question, and which ones are part of the document's words.
* `tokenizer.cls_token_id` will help find the special token at the beginning of the input.
* `word_ids` will help match the answer found in the original `words` to the same answer in the full encoded input and determine
the start/end position of the answer in the encoded input.

With that in mind, let's create a function to encode a batch of examples in the dataset:

```py
>>> def encode_dataset(examples, max_length=512):
...     questions = examples["question"]
...     words = examples["words"]
...     boxes = examples["boxes"]
...     answers = examples["answer"]

...     # encode the batch of examples and initialize the start_positions and end_positions
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []

...     # loop through the examples in the batch
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)

...         # find the position of the answer in example's words
...         words_example = [word.lower() for word in words[i]]
...         answer = answers[i]
...         match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())

...         if match:
...             # if match is found, use `token_type_ids` to find where words start in the encoding
...             token_type_ids = encoding["token_type_ids"][i]
...             token_start_index = 0
...             while token_type_ids[token_start_index] != 1:
...                 token_start_index += 1

...             token_end_index = len(encoding["input_ids"][i]) - 1
...             while token_type_ids[token_end_index] != 1:
...                 token_end_index -= 1

...             word_ids = encoding.word_ids(i)[token_start_index : token_end_index + 1]
...             start_position = cls_index
...             end_position = cls_index

...             # loop over word_ids and increase `token_start_index` until it matches the answer position in words
...             # once it matches, save the `token_start_index` as the `start_position` of the answer in the encoding
...             for id in word_ids:
...                 if id == word_idx_start:
...                     start_position = token_start_index
...                 else:
...                     token_start_index += 1

...             # similarly loop over `word_ids` starting from the end to find the `end_position` of the answer
...             for id in word_ids[::-1]:
...                 if id == word_idx_end:
...                     end_position = token_end_index
...                 else:
...                     token_end_index -= 1

...             start_positions.append(start_position)
...             end_positions.append(end_position)

...         else:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)

...     encoding["image"] = examples["image"]
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions

...     return encoding
```

Now that we have this preprocessing function, we can encode the entire dataset:

```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```

Let's check what the features of the encoded dataset look like:

```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'start_positions': Value(dtype='int64', id=None),
 'end_positions': Value(dtype='int64', id=None)}
```

## Evaluation

Evaluation for document question answering requires a significant amount of postprocessing. To avoid taking up too much
of your time, this guide skips the evaluation step. The [`Trainer`] still calculates the evaluation loss during training so
you're not completely in the dark about your model's performance. Extractive question answering is typically evaluated using F1/exact match.
If you'd like to implement it yourself, check out the [Question Answering chapter](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing)
of the Hugging Face course for inspiration.

## Train

Congratulations! You've successfully navigated the toughest part of this guide and now you are ready to train your own model.
Training involves the following steps:
* Load the model with [`AutoModelForDocumentQuestionAnswering`] using the same checkpoint as in the preprocessing.
* Define your training hyperparameters in [`TrainingArguments`].
* Define a function to batch examples together, here the [`DefaultDataCollator`] will do just fine
* Pass the training arguments to [`Trainer`] along with the model, dataset, and data collator.
* Call [`~Trainer.train`] to finetune your model.

```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

In the [`TrainingArguments`] use `output_dir` to specify where to save your model, and configure hyperparameters as you see fit.
If you wish to share your model with the community, set `push_to_hub` to `True` (you must be signed in to Hugging Face to upload your model).
In this case the `output_dir` will also be the name of the repo where your model checkpoint will be pushed.

```py
>>> from transformers import TrainingArguments

>>> # REPLACE THIS WITH YOUR REPO ID
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     eval_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

Define a simple data collator to batch examples together.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

Finally, bring everything together, and call [`~Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     processing_class=processor,
... )

>>> trainer.train()
```

To add the final model to 🤗 Hub, create a model card and call `push_to_hub`:

```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## Inference

Now that you have finetuned a LayoutLMv2 model, and uploaded it to the 🤗 Hub, you can use it for inference. The simplest
way to try out your finetuned model for inference is to use it in a [`Pipeline`].

Let's take an example:

```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

Next, instantiate a pipeline for
document question answering with your model, and pass the image + question combination to it.

```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

You can also manually replicate the results of the pipeline if you'd like:
1. Take an image and a question, prepare them for the model using the processor from your model.
2. Forward the result or preprocessing through the model.
3. The model returns `start_logits` and `end_logits`, which indicate which token is at the start of the answer and
which token is at the end of the answer. Both have shape (batch_size, sequence_length).
4. Take an argmax on the last dimension of both the `start_logits` and `end_logits` to get the predicted `start_idx` and `end_idx`.
5. Decode the answer with the tokenizer.

```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```
