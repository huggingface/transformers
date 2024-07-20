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

# Image tasks with IDEFICS

[[open-in-colab]]

While individual tasks can be tackled by fine-tuning specialized models, an alternative approach 
that has recently emerged and gained popularity is to use large models for a diverse set of tasks without fine-tuning. 
For instance, large language models can handle such NLP tasks as summarization, translation, classification, and more. 
This approach is no longer limited to a single modality, such as text, and in this guide, we will illustrate how you can 
solve image-text tasks with a large multimodal model called IDEFICS. 

[IDEFICS](../model_doc/idefics) is an open-access vision and language model based on [Flamingo](https://huggingface.co/papers/2204.14198), 
a state-of-the-art visual language model initially developed by DeepMind. The model accepts arbitrary sequences of image 
and text inputs and generates coherent text as output. It can answer questions about images, describe visual content, 
create stories grounded in multiple images, and so on. IDEFICS comes in two variants - [80 billion parameters](https://huggingface.co/HuggingFaceM4/idefics-80b) 
and [9 billion parameters](https://huggingface.co/HuggingFaceM4/idefics-9b), both of which are available on the 🤗 Hub. For each variant, you can also find fine-tuned instructed 
versions of the model adapted for conversational use cases.

This model is exceptionally versatile and can be used for a wide range of image and multimodal tasks. However, 
being a large model means it requires significant computational resources and infrastructure. It is up to you to decide whether 
this approach suits your use case better than fine-tuning specialized models for each individual task. 

In this guide, you'll learn how to: 
- [Load IDEFICS](#loading-the-model) and [load the quantized version of the model](#quantized-model)
- Use IDEFICS for: 
  - [Image captioning](#image-captioning)
  - [Prompted image captioning](#prompted-image-captioning)
  - [Few-shot prompting](#few-shot-prompting)
  - [Visual question answering](#visual-question-answering)
  - [Image classification](#image-classification)
  - [Image-guided text generation](#image-guided-text-generation)
- [Run inference in batch mode](#running-inference-in-batch-mode)
- [Run IDEFICS instruct for conversational use](#idefics-instruct-for-conversational-use)

Before you begin, make sure you have all the necessary libraries installed. 

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
To run the following examples with a non-quantized version of the model checkpoint you will need at least 20GB of GPU memory.
</Tip>

## Loading the model

Let's start by loading the model's 9 billion parameters checkpoint: 

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

Just like for other Transformers models, you need to load a processor and the model itself from the checkpoint. 
The IDEFICS processor wraps a [`LlamaTokenizer`] and IDEFICS image processor into a single processor to take care of 
preparing text and image inputs for the model.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

Setting `device_map` to `"auto"` will automatically determine how to load and store the model weights in the most optimized 
manner given existing devices.

### Quantized model

If high-memory GPU availability is an issue, you can load the quantized version of the model. To load the model and the 
processor in 4bit precision, pass a `BitsAndBytesConfig` to the `from_pretrained` method and the model will be compressed 
on the fly while loading.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True,
...     bnb_4bit_compute_dtype=torch.float16,
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint,
...     quantization_config=quantization_config,
...     device_map="auto"
... )
```

Now that you have the model loaded in one of the suggested ways, let's move on to exploring tasks that you can use IDEFICS for.

## Image captioning
Image captioning is the task of predicting a caption for a given image. A common application is to aid visually impaired 
people navigate through different situations, for instance, explore image content online. 

To illustrate the task, get an image to be captioned, e.g.:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="Image of a puppy in a flower bed"/>
</div>

Photo by [Hendo Wang](https://unsplash.com/@hendoo). 

IDEFICS accepts text and image prompts. However, to caption an image, you do not have to provide a text prompt to the 
model, only the preprocessed input image. Without a text prompt, the model will start generating text from the 
BOS (beginning-of-sequence) token thus creating a caption.

As image input to the model, you can use either an image object (`PIL.Image`) or a url from which the image can be retrieved.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
A puppy in a flower bed
```

<Tip>

It is a good idea to include the `bad_words_ids` in the call to `generate` to avoid errors arising when increasing 
the `max_new_tokens`: the model will want to generate a new `<image>` or `<fake_token_around_image>` token when there 
is no image being generated by the model.
You can set it on-the-fly as in this guide, or store in the `GenerationConfig` as described in the [Text generation strategies](../generation_strategies) guide.
</Tip>

## Prompted image captioning

You can extend image captioning by providing a text prompt, which the model will continue given the image. Let's take 
another image to illustrate:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="Image of the Eiffel Tower at night"/>
</div>

Photo by [Denys Nevozhai](https://unsplash.com/@dnevozhai).
   
Textual and image prompts can be passed to the model's processor as a single list to create appropriate inputs.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...     "This is an image of ",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
This is an image of the Eiffel Tower in Paris, France.
```

## Few-shot prompting

While IDEFICS demonstrates great zero-shot results, your task may require a certain format of the caption, or come with 
other restrictions or requirements that increase task's complexity. Few-shot prompting can be used to enable in-context learning.
By providing examples in the prompt, you can steer the model to generate results that mimic the format of given examples. 

Let's use the previous image of the Eiffel Tower as an example for the model and build a prompt that demonstrates to the model 
that in addition to learning what the object in an image is, we would also like to get some interesting information about it. 
Then, let's see, if we can get the same response format for an image of the Statue of Liberty:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="Image of the Statue of Liberty"/>
</div>

Photo by [Juan Mayobre](https://unsplash.com/@jmayobres).
  
```py
>>> prompt = ["User:",
...            "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...            "Describe this image.\nAssistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building.\n",
...            "User:",
...            "https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80",
...            "Describe this image.\nAssistant:"
...            ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
User: Describe this image.
Assistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building. 
User: Describe this image.
Assistant: An image of the Statue of Liberty. Fun fact: the Statue of Liberty is 151 feet tall.
```

Notice that just from a single example (i.e., 1-shot) the model has learned how to perform the task. For more complex tasks, 
feel free to experiment with a larger number of examples (e.g., 3-shot, 5-shot, etc.).

## Visual question answering

Visual Question Answering (VQA) is the task of answering open-ended questions based on an image. Similar to image 
captioning it can be used in accessibility applications, but also in education (reasoning about visual materials), customer 
service (questions about products based on images), and image retrieval.

Let's get a new image for this task: 

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="Image of a couple having a picnic"/>
</div>

Photo by [Jarritos Mexican Soda](https://unsplash.com/@jarritos). 

You can steer the model from image captioning to visual question answering by prompting it with appropriate instructions: 

```py
>>> prompt = [
...     "Instruction: Provide an answer to the question. Use the image to answer.\n",
...     "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...     "Question: Where are these people and what's the weather like? Answer:"
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Provide an answer to the question. Use the image to answer.
 Question: Where are these people and what's the weather like? Answer: They're in a park in New York City, and it's a beautiful day.
```

## Image classification

IDEFICS is capable of classifying images into different categories without being explicitly trained on data containing 
labeled examples from those specific categories. Given a list of categories and using its image and text understanding 
capabilities, the model can infer which category the image likely belongs to. 

Say, we have this image of a vegetable stand: 

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="Image of a vegetable stand"/>
</div>

Photo by [Peter Wendt](https://unsplash.com/@peterwendt).

We can instruct the model to classify the image into one of the categories that we have:

```py
>>> categories = ['animals','vegetables', 'city landscape', 'cars', 'office']
>>> prompt = [f"Instruction: Classify the following image into a single category from the following list: {categories}.\n",
...     "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",    
...     "Category: "
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=6, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Classify the following image into a single category from the following list: ['animals', 'vegetables', 'city landscape', 'cars', 'office'].
Category: Vegetables
```  

In the example above we instruct the model to classify the image into a single category, however, you can also prompt the model to do rank classification.

## Image-guided text generation

For more creative applications, you can use image-guided text generation to generate text based on an image. This can be 
useful to create descriptions of products, ads, descriptions of a scene, etc. 

Let's prompt IDEFICS to write a story based on a simple image of a red door: 

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="Image of a red door with a pumpkin on the steps"/>
</div>

Photo by [Craig Tidball](https://unsplash.com/@devonshiremedia).
  
```py
>>> prompt = ["Instruction: Use the image to write a story. \n",
...     "https://images.unsplash.com/photo-1517086822157-2b0358e7684a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2203&q=80",
...     "Story: \n"]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, num_beams=2, max_new_tokens=200, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0]) 
Instruction: Use the image to write a story. 
 Story: 
Once upon a time, there was a little girl who lived in a house with a red door.  She loved her red door.  It was the prettiest door in the whole world.

One day, the little girl was playing in her yard when she noticed a man standing on her doorstep.  He was wearing a long black coat and a top hat.

The little girl ran inside and told her mother about the man.

Her mother said, “Don’t worry, honey.  He’s just a friendly ghost.”

The little girl wasn’t sure if she believed her mother, but she went outside anyway.

When she got to the door, the man was gone.

The next day, the little girl was playing in her yard again when she noticed the man standing on her doorstep.

He was wearing a long black coat and a top hat.

The little girl ran
```

Looks like IDEFICS noticed the pumpkin on the doorstep and went with a spooky Halloween story about a ghost.

<Tip>

For longer outputs like this, you will greatly benefit from tweaking the text generation strategy. This can help 
you significantly improve the quality of the generated output. Check out [Text generation strategies](../generation_strategies) 
to learn more. 
</Tip>

## Running inference in batch mode

All of the earlier sections illustrated IDEFICS for a single example. In a very similar fashion, you can run inference 
for a batch of examples by passing a list of prompts:

```py
>>> prompts = [
...     [   "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
... ]

>>> inputs = processor(prompts, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i,t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n") 
0:
This is an image of the Eiffel Tower in Paris, France.

1:
This is an image of a couple on a picnic blanket.

2:
This is an image of a vegetable stand.
```

## IDEFICS instruct for conversational use

For conversational use cases, you can find fine-tuned instructed versions of the model on the 🤗 Hub: 
`HuggingFaceM4/idefics-80b-instruct` and `HuggingFaceM4/idefics-9b-instruct`.

These checkpoints are the result of fine-tuning the respective base models on a mixture of supervised and instruction 
fine-tuning datasets, which boosts the downstream performance while making the models more usable in conversational settings.

The use and prompting for the conversational use is very similar to using the base models: 

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "User: What is in this image?",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

...         "\nUser:",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "And who is that?<end_of_utterance>",

...         "\nAssistant:",
...     ],
... ]

>>> # --batched mode
>>> inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
>>> # --single sample mode
>>> # inputs = processor(prompts[0], return_tensors="pt").to(device)

>>> # Generation args
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")
```
