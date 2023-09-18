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

- [Basics of prompting](#basic-prompts)
- [Best practices of LLM prompting](#best-practices-of-llm-prompting)
- [Prompt formatting and structure](#prompt-formatting-and-structure)
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

>>> generator = pipeline('text-generation', model = 'gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)

[{'generated_text': "Hello, I'm a language model programmer.\n\nYou know what? I've got to explain my language concepts to a lot of people. What"}]
```

To run inference with an encoder-decoder, use the `text2text-generation` pipeline:

```python
>>> from transformers import pipeline

>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)

[{'generated_text': 'Je suis très heureuse de vous voir'}]
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
pip install -q transformers einops
```

Next, let's load the model with the appropriate pipeline (`"text-generation"`): 

```python
>>> from transformers import AutoTokenizer
>>> import transformers
>>> import torch

>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipeline = transformers.pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     torch_dtype=torch.bfloat16,
...     trust_remote_code=True,
...     device_map="auto",
... )
```

<Tip>

Note that:
* Falcon models were trained using the `bfloat16` datatype, so we recommend you use the same. This requires a recent 
version of CUDA and works best on modern cards. 
* You need to allow remote code execution. This is because the Falcon models use a new architecture that is not part of transformers yet - instead, the code necessary is provided by the model authors in the repo.
</Tip>

Now that we have the model loaded via the pipeline, let's explore how you can use prompts to solve NLP tasks.

#### Text classification

One of the most common forms of text classification is sentiment analysis, which assigns a label like positive, negative, 
or neutral to a sequence of text. Let's write a prompt that instructs the model to classify a given text (a movie review):

```python
>>> prompt = """Classify the text into neutral, negative or positive. 
    Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
    Sentiment:
    """

>>> sequences = pipeline(
...     prompt,
...     max_length=20,
...     do_sample=True,
...     top_k=10,
...     num_return_sequences=1,
...     eos_token_id=tokenizer.eos_token_id,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")

Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

<Tip>

You may notice that in addition to the prompt, we pass a number of text generation parameters. `max_length` controls the 
length of the generated output. To learn about the rest, please refer to the [Text generation strategies](../generation_strategies) guide.
</Tip>

#### Named Entity Recognition

Named Entity Recognition (NER) is a task of finding named entities in a piece of text, such as a person, location, or organization.

```python
>>> prompt = """Return a list of named entities in the text.
    Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
    Named entities:
    """

>>> sequences = pipeline(
...     prompt,
...     max_length=80,
...     do_sample=True,
...     top_k=10,
...     num_return_sequences=1,
...     eos_token_id=tokenizer.eos_token_id,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")

Result: Return a list of named entities in the text.
Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
Named entities:
- The Golden State Warriors
- San Francisco
- NBA
```

As you can see, the model correctly identified two named entities, however, it has also added a new entity that is not in the text. 
Further in the guide we'll talk about techniques you can use to improve your prompts.

#### Translation

```python
>>> prompt = """Translate the English text to Italian.
    Text: Sometimes, I've believed as many as six impossible things before breakfast.
    Translation:
    """

>>> sequences = pipeline(
...     prompt,
...     max_length=100,
...     do_sample=True,
...     top_k=10,
...     num_return_sequences=1,
...     eos_token_id=tokenizer.eos_token_id,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")

Result: Translate the English text to Italian.
Text: Sometimes, I've believed as many as six impossible things before breakfast.
Translation:
A volte, ho creduto a sei impossibili cose prima di colazione.
```


#### Text summarization

```python
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
    Write a summary of the above text.
    Summary:
    """

>>> sequences = pipeline(
...     prompt,
...     max_length=300,
...     do_sample=True,
...     top_k=10,
...     num_return_sequences=1,
...     eos_token_id=tokenizer.eos_token_id,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
Write a summary of the above text.
Summary:
Permaculture is an ecosystem design mimicking natural, diverse functionalities and resilience, aiming to meet basic needs and prepare for climate change in communities.
```

(???) #### Information extraction

#### Question answering

(???) #### Conversation

(???) #### Code generation

(???) #### Reasoning




### Prompting language

Mention English language vs non-english

(check whether the model is suitable for the alternative language.)
(performance may be similar but the same prompt in another language may result in more tokens, which can affect latency.)
(More tokens processed translate to increased computation, which, in simplified terms, results in longer latency.)
Anecdotal: procedural prompts work better in ENglish (possibly due to LLMs trained on code where English is the most common language)


## Best practices of LLM prompting

Start with a simple prompt and iterate from there. 
Instructions are commonly placed at the beginning of the prompt. Use clear separator to indicate where the instructions are. 
The more descriptive and detailed the prompt is, the better the results. 
Dos are better than don’ts (tell the model what to do, don’t tell what NOT to do. It can do that anyways)

## Prompt formatting and structure

(add a note on system prompts and chat templates)

## Advanced prompting techniques

### Few-shot prompting

Limitations of few-shot prompting - doesn’t work well on complex reasoning tasks
Few-shot learning significantly uses up the token budget of your prompts, which can be expensive - especially if it doesn't let you go down a level or two in intelligence and per-token cost.
Additionally, models can learn things you didn't teach. For example, providing two positive examples and one negative example can teach a model that the third message is always wrong - which actually happened to us a few times!


### Chain-of-thought


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
