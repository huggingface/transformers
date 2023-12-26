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

# Transformers Agents

<Tip warning={true}>

Transformers Agents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

Transformers version v4.29.0, building on the concept of *tools* and *agents*. You can play with in
[this colab](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj).

In short, it provides a natural language API on top of transformers: we define a set of curated tools and design an 
agent to interpret natural language and to use these tools. It is extensible by design; we curated some relevant tools, 
but we'll show you how the system can be extended easily to use any tool developed by the community.

Let's start with a few examples of what can be achieved with this new API. It is particularly powerful when it comes 
to multimodal tasks, so let's take it for a spin to generate images and read text out loud.

```py
agent.run("Caption the following image", image=image)
```

| **Input**                                                                                                                   | **Output**                        |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |

---

```py
agent.run("Read the following text out loud", text=text)
```
| **Input**                                                                                                               | **Output**                                   |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio>

---

```py
agent.run(
    "In the following `document`, where will the TRRF Scientific Advisory Council Meeting take place?",
    document=document,
)
```
| **Input**                                                                                                                   | **Output**     |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | ballroom foyer |

## Quickstart

Before being able to use `agent.run`, you will need to instantiate an agent, which is a large language model (LLM). 
We provide support for openAI models as well as opensource alternatives from BigCode and OpenAssistant. The openAI
models perform better (but require you to have an openAI API key, so cannot be used for free); Hugging Face is
providing free access to endpoints for BigCode and OpenAssistant models.

To start with, please install the `agents` extras in order to install all default dependencies.
```bash
pip install transformers[agents]
```

To use openAI models, you instantiate an [`OpenAiAgent`] after installing the `openai` dependency:

```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

To use BigCode or OpenAssistant, start by logging in to have access to the Inference API:

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

Then, instantiate the agent

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

This is using the inference API that Hugging Face provides for free at the moment. If you have your own inference
endpoint for this model (or another one) you can replace the URL above with your URL endpoint.

<Tip>

StarCoder and OpenAssistant are free to use and perform admirably well on simple tasks. However, the checkpoints
don't hold up when handling more complex prompts. If you're facing such an issue, we recommend trying out the OpenAI
model which, while sadly not open-source, performs better at this given time.

</Tip>

You're now good to go! Let's dive into the two APIs that you now have at your disposal.

### Single execution (run)

The single execution method is when using the [`~Agent.run`] method of the agent:

```py
agent.run("Draw me a picture of rivers and lakes.")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>

It automatically selects the tool (or tools) appropriate for the task you want to perform and runs them appropriately. It
can perform one or several tasks in the same instruction (though the more complex your instruction, the more likely
the agent is to fail).

```py
agent.run("Draw me a picture of the sea then transform the picture to add an island")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>


Every [`~Agent.run`] operation is independent, so you can run it several times in a row with different tasks.

Note that your `agent` is just a large-language model, so small variations in your prompt might yield completely
different results. It's important to explain as clearly as possible the task you want to perform. We go more in-depth
on how to write good prompts [here](custom_tools#writing-good-user-inputs).

If you'd like to keep a state across executions or to pass non-text objects to the agent, you can do so by specifying
variables that you would like the agent to use. For example, you could generate the first image of rivers and lakes, 
and ask the model to update that picture to add an island by doing the following:

```python
picture = agent.run("Generate a picture of rivers and lakes.")
updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
```

<Tip>

This can be helpful when the model is unable to understand your request and mixes tools. An example would be:

```py
agent.run("Draw me the picture of a capybara swimming in the sea")
```

Here, the model could interpret in two ways:
- Have the `text-to-image` generate a capybara swimming in the sea
- Or, have the `text-to-image` generate capybara, then use the `image-transformation` tool to have it swim in the sea

In case you would like to force the first scenario, you could do so by passing it the prompt as an argument:

```py
agent.run("Draw me a picture of the `prompt`", prompt="a capybara swimming in the sea")
```

</Tip>


### Chat-based execution (chat)

The agent also has a chat-based approach, using the [`~Agent.chat`] method:

```py
agent.chat("Generate a picture of rivers and lakes")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("Transform the picture so that there is a rock in there")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

This is an interesting approach when you want to keep the state across instructions. It's better for experimentation, 
but will tend to be much better at single instructions rather than complex instructions (which the [`~Agent.run`]
method is better at handling).

This method can also take arguments if you would like to pass non-text types or specific prompts.

### ⚠️ Remote execution

For demonstration purposes and so that it could be used with all setups, we had created remote executors for several 
of the default tools the agent has access for the release. These are created using 
[inference endpoints](https://huggingface.co/inference-endpoints).

We have turned these off for now, but in order to see how to set up remote executors tools yourself,
we recommend reading the [custom tool guide](./custom_tools).

### What's happening here? What are tools, and what are agents?

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png">

#### Agents

The "agent" here is a large language model, and we're prompting it so that it has access to a specific set of tools.

LLMs are pretty good at generating small samples of code, so this API takes advantage of that by prompting the 
LLM gives a small sample of code performing a task with a set of tools. This prompt is then completed by the 
task you give your agent and the description of the tools you give it. This way it gets access to the doc of the 
tools you are using, especially their expected inputs and outputs, and can generate the relevant code.

#### Tools

Tools are very simple: they're a single function, with a name, and a description. We then use these tools' descriptions 
to prompt the agent. Through the prompt, we show the agent how it would leverage tools to perform what was 
requested in the query.

This is using brand-new tools and not pipelines, because the agent writes better code with very atomic tools. 
Pipelines are more refactored and often combine several tasks in one. Tools are meant to be focused on
one very simple task only.

#### Code-execution?!

This code is then executed with our small Python interpreter on the set of inputs passed along with your tools. 
We hear you screaming "Arbitrary code execution!" in the back, but let us explain why that is not the case.

The only functions that can be called are the tools you provided and the print function, so you're already 
limited in what can be executed. You should be safe if it's limited to Hugging Face tools. 

Then, we don't allow any attribute lookup or imports (which shouldn't be needed anyway for passing along 
inputs/outputs to a small set of functions) so all the most obvious attacks (and you'd need to prompt the LLM 
to output them anyway) shouldn't be an issue. If you want to be on the super safe side, you can execute the 
run() method with the additional argument return_code=True, in which case the agent will just return the code 
to execute and you can decide whether to do it or not.

The execution will stop at any line trying to perform an illegal operation or if there is a regular Python error 
with the code generated by the agent.

### A curated set of tools

We identify a set of tools that can empower such agents. Here is an updated list of the tools we have integrated 
in `transformers`:

- **Document question answering**: given a document (such as a PDF) in image format, answer a question on this document ([Donut](./model_doc/donut))
- **Text question answering**: given a long text and a question, answer the question in the text ([Flan-T5](./model_doc/flan-t5))
- **Unconditional image captioning**: Caption the image! ([BLIP](./model_doc/blip))
- **Image question answering**: given an image, answer a question on this image ([VILT](./model_doc/vilt))
- **Image segmentation**: given an image and a prompt, output the segmentation mask of that prompt ([CLIPSeg](./model_doc/clipseg))
- **Speech to text**: given an audio recording of a person talking, transcribe the speech into text ([Whisper](./model_doc/whisper))
- **Text to speech**: convert text to speech ([SpeechT5](./model_doc/speecht5))
- **Zero-shot text classification**: given a text and a list of labels, identify to which label the text corresponds the most ([BART](./model_doc/bart))
- **Text summarization**: summarize a long text in one or a few sentences ([BART](./model_doc/bart))
- **Translation**: translate the text into a given language ([NLLB](./model_doc/nllb))

These tools have an integration in transformers, and can be used manually as well, for example:

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### Custom tools

While we identify a curated set of tools, we strongly believe that the main value provided by this implementation is 
the ability to quickly create and share custom tools.

By pushing the code of a tool to a Hugging Face Space or a model repository, you're then able to leverage the tool 
directly with the agent. We've added a few 
**transformers-agnostic** tools to the [`huggingface-tools` organization](https://huggingface.co/huggingface-tools):

- **Text downloader**: to download a text from a web URL
- **Text to image**: generate an image according to a prompt, leveraging stable diffusion
- **Image transformation**: modify an image given an initial image and a prompt, leveraging instruct pix2pix stable diffusion
- **Text to video**: generate a small video according to a prompt, leveraging damo-vilab

The text-to-image tool we have been using since the beginning is a remote tool that lives in 
[*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)! We will
continue releasing such tools on this and other organizations, to further supercharge this implementation.

The agents have by default access to tools that reside on [`huggingface-tools`](https://huggingface.co/huggingface-tools).
We explain how to you can write and share your tools as well as leverage any custom tool that resides on the Hub in [following guide](custom_tools).

### Code generation

So far we have shown how to use the agents to perform actions for you. However, the agent is only generating code
that we then execute using a very restricted Python interpreter. In case you would like to use the code generated in 
a different setting, the agent can be prompted to return the code, along with tool definition and accurate imports.

For example, the following instruction
```python
agent.run("Draw me a picture of rivers and lakes", return_code=True)
```

returns the following code

```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

that you can then modify and execute yourself.
