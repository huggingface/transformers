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


# LLM prompting guide

[[open-in-colab]]

Large Language Models such as Falcon, LLaMA, etc. are pretrained transformer models initially trained to predict the 
next token given some input text. They typically have billions of parameters and have been trained on trillions of 
tokens for an extended period of time. As a result, these models become quite powerful and versatile, and you can use 
them to solve multiple NLP tasks out of the box by instructing the models with natural language prompts.

Designing such prompts to ensure the optimal output is often called "prompt engineering". Prompt engineering is an 
iterative process that requires a fair amount of experimentation. Natural languages are much more flexible and expressive 
than programming languages, however, they can also introduce some ambiguity. At the same time, prompts in natural language 
are quite sensitive to changes. Even minor modifications in prompts can lead to wildly different outputs.

While there is no exact recipe for creating prompts to match all cases, researchers have worked out a number of best 
practices that help to achieve optimal results more consistently. 

This guide covers the prompt engineering best practices to help you craft better LLM prompts and solve various NLP tasks. 
You'll learn:

- [Basics of prompting](#basics-of-prompting)
- [Best practices of LLM prompting](#best-practices-of-llm-prompting)
- [Advanced prompting techniques: few-shot prompting and chain-of-thought](#advanced-prompting-techniques)
- [When to fine-tune instead of prompting](#prompting-vs-fine-tuning)

<Tip>

Prompt engineering is only a part of the LLM output optimization process. Another essential component is choosing the 
optimal text generation strategy. You can customize how your LLM selects each of the subsequent tokens when generating 
the text without modifying any of the trainable parameters. By tweaking the text generation parameters, you can reduce 
repetition in the generated text and make it more coherent and human-sounding. 
Text generation strategies and parameters are out of scope for this guide, but you can learn more about these topics in 
the following guides: 
 
* [Generation with LLMs](../llm_tutorial)
* [Text generation strategies](../generation_strategies)

</Tip>

## Basics of prompting

### Types of models 

The majority of modern LLMs are decoder-only transformers. Some examples include: [LLaMA](../model_doc/llama), 
[Llama2](../model_doc/llama2), [Falcon](../model_doc/falcon), [GPT2](../model_doc/gpt2). However, you may encounter
encoder-decoder transformer LLMs as well, for instance, [Flan-T5](../model_doc/flan-t5) and [BART](../model_doc/bart).

Encoder-decoder-style models are typically used in generative tasks where the output **heavily** relies on the input, for 
example, in translation and summarization. The decoder-only models are used for all other types of generative tasks.

When using a pipeline to generate text with an LLM, it's important to know what type of LLM you are using, because 
they use different pipelines. 

Run inference with decoder-only models with the `text-generation` pipeline:

```python
>>> from transformers import pipeline
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT

>>> generator = pipeline('text-generation', model = 'openai-community/gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)
[{'generated_text': "Hello, I'm a language model programmer so you can use some of my stuff. But you also need some sort of a C program to run."}]
```

To run inference with an encoder-decoder, use the `text2text-generation` pipeline:

```python
>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)
[{'generated_text': 'Je suis très heureuse de vous rencontrer.'}]
```

### Base vs instruct/chat models

Most of the recent LLM checkpoints available on 🤗 Hub come in two versions: base and instruct (or chat). For example, 
[`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b) and [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct).

Base models are excellent at completing the text when given an initial prompt, however, they are not ideal for NLP tasks 
where they need to follow instructions, or for conversational use. This is where the instruct (chat) versions come in. 
These checkpoints are the result of further fine-tuning of the pre-trained base versions on instructions and conversational data. 
This additional fine-tuning makes them a better choice for many NLP tasks.  

Let's illustrate some simple prompts that you can use with [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct) 
to solve some common NLP tasks.

### NLP tasks 

First, let's set up the environment: 

```bash
pip install -q transformers accelerate
```

Next, let's load the model with the appropriate pipeline (`"text-generation"`): 

```python
>>> from transformers import pipeline, AutoTokenizer
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipe = pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
```

<Tip>

Note that Falcon models were trained using the `bfloat16` datatype, so we recommend you use the same. This requires a recent 
version of CUDA and works best on modern cards.

</Tip>

Now that we have the model loaded via the pipeline, let's explore how you can use prompts to solve NLP tasks.

#### Text classification

One of the most common forms of text classification is sentiment analysis, which assigns a label like "positive", "negative", 
or "neutral" to a sequence of text. Let's write a prompt that instructs the model to classify a given text (a movie review). 
We'll start by giving the instruction, and then specifying the text to classify. Note that instead of leaving it at that, we're 
also adding the beginning of the response - `"Sentiment: "`:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Classify the text into neutral, negative or positive. 
... Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
... Sentiment:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

As a result, the output contains a classification label from the list we have provided in the instructions, and it is a correct one!

<Tip>

You may notice that in addition to the prompt, we pass a `max_new_tokens` parameter. It controls the number of tokens the 
model shall generate, and it is one of the many text generation parameters that you can learn about 
in [Text generation strategies](../generation_strategies) guide.

</Tip>

#### Named Entity Recognition

Named Entity Recognition (NER) is a task of finding named entities in a piece of text, such as a person, location, or organization.
Let's modify the instructions in the prompt to make the LLM perform this task. Here, let's also set `return_full_text = False` 
so that output doesn't contain the prompt:

```python
>>> torch.manual_seed(1) # doctest: +IGNORE_RESULT
>>> prompt = """Return a list of named entities in the text.
... Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
... Named entities:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=15,
...     return_full_text = False,    
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
- Golden State Warriors
- San Francisco
```

As you can see, the model correctly identified two named entities from the given text.

#### Translation

Another task LLMs can perform is translation. You can choose to use encoder-decoder models for this task, however, here,
for the simplicity of the examples, we'll keep using Falcon-7b-instruct, which does a decent job. Once again, here's how 
you can write a basic prompt to instruct a model to translate a piece of text from English to Italian: 

```python
>>> torch.manual_seed(2) # doctest: +IGNORE_RESULT
>>> prompt = """Translate the English text to Italian.
... Text: Sometimes, I've believed as many as six impossible things before breakfast.
... Translation:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=20,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
A volte, ho creduto a sei impossibili cose prima di colazione.
```

Here we've added a `do_sample=True` and `top_k=10` to allow the model to be a bit more flexible when generating output.

#### Text summarization

Similar to the translation, text summarization is another generative task where the output **heavily** relies on the input, 
and encoder-decoder models can be a better choice. However, decoder-style models can be used for this task as well.
Previously, we have placed the instructions at the very beginning of the prompt. However, the very end of the prompt can 
also be a suitable location for instructions. Typically, it's better to place the instruction on one of the extreme ends.  

```python
>>> torch.manual_seed(3) # doctest: +IGNORE_RESULT
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
... Write a summary of the above text.
... Summary:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
Permaculture is an ecological design mimicking natural ecosystems to meet basic needs and prepare for climate change. It is based on traditional knowledge and scientific understanding.
```

#### Question answering

For question answering task we can structure the prompt into the following logical components: instructions, context, question, and 
the leading word or phrase (`"Answer:"`) to nudge the model to start generating the answer:

```python
>>> torch.manual_seed(4) # doctest: +IGNORE_RESULT
>>> prompt = """Answer the question using the context below.
... Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
... Question: What modern tool is used to make gazpacho?
... Answer:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Modern tools often used to make gazpacho include
```

#### Reasoning

Reasoning is one of the most difficult tasks for LLMs, and achieving good results often requires applying advanced prompting techniques, like 
[Chain-of-thought](#chain-of-thought).

Let's try if we can make a model reason about a simple arithmetics task with a basic prompt: 

```python
>>> torch.manual_seed(5) # doctest: +IGNORE_RESULT
>>> prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
There are a total of 5 groups, so there are 5 x 4=20 students in the class.
```

Correct! Let's increase the complexity a little and see if we can still get away with a basic prompt:

```python
>>> torch.manual_seed(6) # doctest: +IGNORE_RESULT
>>> prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
The total number of muffins now is 21
```

This is a wrong answer, it should be 12. In this case, this can be due to the prompt being too basic, or due to the choice 
of model, after all we've picked the smallest version of Falcon. Reasoning is difficult for models of all sizes, but larger 
models are likely to perform better. 

## Best practices of LLM prompting

In this section of the guide we have compiled a list of best practices that tend to improve the prompt results:

* When choosing the model to work with, the latest and most capable models are likely to perform better. 
* Start with a simple and short prompt, and iterate from there.
* Put the instructions at the beginning of the prompt, or at the very end. When working with large context, models apply various optimizations to prevent Attention complexity from scaling quadratically. This may make a model more attentive to the beginning or end of a prompt than the middle.
* Clearly separate instructions from the text they apply to - more on this in the next section. 
* Be specific and descriptive about the task and the desired outcome - its format, length, style, language, etc.
* Avoid ambiguous descriptions and instructions.
* Favor instructions that say "what to do" instead of those that say "what not to do".
* "Lead" the output in the right direction by writing the first word (or even begin the first sentence for the model).
* Use advanced techniques like [Few-shot prompting](#few-shot-prompting) and [Chain-of-thought](#chain-of-thought)
* Test your prompts with different models to assess their robustness. 
* Version and track the performance of your prompts. 

## Advanced prompting techniques

### Few-shot prompting

The basic prompts in the sections above are the examples of "zero-shot" prompts, meaning, the model has been given 
instructions and context, but no examples with solutions. LLMs that have been fine-tuned on instruction datasets, generally 
perform well on such "zero-shot" tasks. However, you may find that your task has more complexity or nuance, and, perhaps, 
you have some requirements for the output that the model doesn't catch on just from the instructions. In this case, you can 
try the technique called few-shot prompting. 

In few-shot prompting, we provide examples in the prompt giving the model more context to improve the performance. 
The examples condition the model to generate the output following the patterns in the examples.

Here's an example: 

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
... Date: 04/12/1961
... Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
... Date:"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=8,
...     do_sample=True,
...     top_k=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

In the above code snippet we used a single example to demonstrate the desired output to the model, so this can be called a 
"one-shot" prompting. However, depending on the task complexity you may need to use more than one example. 

Limitations of the few-shot prompting technique: 
- While LLMs can pick up on the patterns in the examples, these technique doesn't work well on complex reasoning tasks
- Few-shot prompting requires creating lengthy prompts. Prompts with large number of tokens can increase computation and latency. There's also a limit to the length of the prompts.  
- Sometimes when given a number of examples, models can learn patterns that you didn't intend them to learn, e.g. that the third movie review is always negative.

### Chain-of-thought

Chain-of-thought (CoT) prompting is a technique that nudges a model to produce intermediate reasoning steps thus improving 
the results on complex reasoning tasks. 

There are two ways of steering a model to producing the reasoning steps:
- few-shot prompting by illustrating examples with detailed answers to questions, showing the model how to work through a problem.
- by instructing the model to reason by adding phrases like "Let's think step by step" or "Take a deep breath and work through the problem step by step."

If we apply the CoT technique to the muffins example from the [reasoning section](#reasoning) and use a larger model, 
such as (`tiiuae/falcon-180B-chat`) which you can play with in the [HuggingChat](https://huggingface.co/chat/), 
we'll get a significant improvement on the reasoning result:

```text
Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
Therefore, you now have 12 muffins.
```

## Prompting vs fine-tuning

You can achieve great results by optimizing your prompts, however, you may still ponder whether fine-tuning a model 
would work better for your case. Here are some scenarios when fine-tuning a smaller model may be a preferred option:

- Your domain is wildly different from what LLMs were pre-trained on and extensive prompt optimization did not yield sufficient results. 
- You need your model to work well in a low-resource language.
- You need the model to be trained on sensitive data that is under strict regulations. 
- You have to use a small model due to cost, privacy, infrastructure or other limitations. 

In all of the above examples, you will need to make sure that you either already have or can easily obtain a large enough 
domain-specific dataset at a reasonable cost to fine-tune a model. You will also need to have enough time and resources 
to fine-tune a model.

If the above examples are not the case for you, optimizing prompts can prove to be more beneficial.   


