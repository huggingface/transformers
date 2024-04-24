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
# Agents and tools

Large Language Models (LLMs) trained to perform [causal language modeling](./tasks/language_modeling.) can tackle a wide range of tasks, but they often struggle with basic tasks like logic, calculation, and search. When prompted in domains in which they do not perform well, they often fail to generate the answer we expect them to.

One approach to overcome this weakness is to create an *agent*, a program powered by an LLM. The agent is empowered by *tools* to help the agent perform an action.

```python
agent.run("Caption the following image", image=image)
```

```markdown
| **Input**                                                                                                                   | **Output**                        |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |
```

---

```python
agent.run("Read the following text out loud", text=text)
```

```markdown
| **Input**                                                                                                               | **Output**                                   |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio>
```

In this tutorial, you'll learn what agents and tools are, how to build them with Transformers, and how to use them.


## Agents

### What is an agent?

An agent is a system that uses an LLM as its engine, and it has access to functions called *tools*.

These *tools* are functions for performing a task, and they contain all necessary description for the agent to properly use them.

The agent can be programmed to:
- devise a series of actions/tools and run them all at once like the `CodeAgent` for example
- plan and execute actions/tools one by one and wait for the outcome of each action before launching the next one like the `ReactJSONAgent` for example

### Types of agents

#### Code agent

This agent has a planning step, then generates python code to execute all its actions at once. It natively handles different input and output types for its tools, thus it is the recommended choice for multimodal tasks.

#### React agents

This is the go-to agent to solve reasoning tasks, since the ReAct framework ([Yao et al., 2022](https://huggingface.co/papers/2210.03629)) makes it really efficient to think on the basis of its previous observations.

We implement two versions of ReactJSONAgent: 
- [`~ReactJSONAgent`] generates tool calls as a JSON in its output.
- [`~ReactCodeAgent`] is a new type of ReactJSONAgent that generates its tool calls as blobs of code, which works really well for LLMs that have strong coding performance.

> [!TIP]
> Read [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) blog post to learn more the ReAct agent.

![Framework of a React Agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

For example, here is how a ReAct agent would work its way through the following question.

```
Task: How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?

Thought: To begin, I need to find the number of layers in the encoder of BERT base and the architecture proposed in Attention is All You Need. I will start by searching for the number of layers in BERT base.
Action:
{
    "action": "search",
    "action_input": "number of layers in BERT base encoder"
}
Observation: 12 layers

Thought: I need to search for the number of layers in the encoder of the architecture proposed in Attention is All You Need.
Action:
{
    "action": "search",
    "action_input": "number of layers in the encoder of the architecture proposed in Attention is All You Need"
}
Observation: Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers.

Thought: Now that I have the number of layers for both BERT base and the architecture proposed in Attention is All You Need, I can calculate the difference.
Action:
{
    "action": "calculator",
    "action_input": "12 - 6"
}
Observation: 6

Thought: Now that I have the calculated difference, I need to provide the final answer.
Action:
{
    "action": "final_answer",
    "action_input": "BERT base encoder has 6 more layers than the encoder from the architecture proposed in Attention is All You Need"
}
Observation: BERT base encoder has 6 more layers than the encoder from the architecture proposed in Attention is All You Need
```

### How can I build an agent?

To initialize an agent, you need these arguments:

- an LLM to power your agent - the agent is not exactly the LLM, it’s more like the agent is a program that uses an LLM as its engine.
- a system prompt: what the LLM engine will be prompted with to generate its output
- a toolbox from which the agent pick tools to execute
- a parser to extract from the LLM output which tools are to call and with which arguments

Upon initialization of the agent system, the tool attributes are used to generate a tool description, then baked into the agent’s `system_prompt` to let it know which tools it can use and why.

To start with, please install the `agents` extras in order to install all default dependencies.

```bash
pip install transformers[agents]
```

Build your LLM engine by defining a `llm_engine` method which accepts a list of [messages](./chat_templating.) and returns text. This callable also needs to accept a `stop` argument that indicates when to stop generating.

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

def llm_engine(messages, stop=["Task"]) -> str:
    response = client.chat_completion(messages, stop_sequences=stop, return_full_text=False, max_new_tokens=1000)
    answer = response.choices[0].message.content
    for stop_seq in stop:
        if answer[-len(stop_seq) :] == stop_seq:
            answer = answer[: -len(stop_seq)]
    return answer
```

You could use any `llm_engine` method as long as:
1. it follows the [messages format](./chat_templating.) for its input (`List[Dict[str, str]]`) and returns a `str`
2. it stops generating outputs *before* the sequences passed in the argument `stop`

You also need a `tools` argument which accepts a list of `Tools`. You can provide an empty list for `tools`, but use the default toolbox with the optional argument `add_base_tools=True`.

Now you can create an agent, like `CodeAgent`, and run it. For convenience, we also provide the `HfEngine` class that uses `huggingface_hub.InferenceClient` under the hood.

```python
from transformers import CodeAgent, HfEngine

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")

agent = CodeAgent(llm_engine=llm_engine, tools=[], add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes")
```

You can even leave argument `llm_engine` undefined, and an `HfEngine` will be loaded by default.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes")
```

The prompt and output parser were automatically defined, but you can easily inspect them by calling the `system_prompt_template` on your agent.

![Untitled](Doc%20agents%20ac753b9a3b934eaba22f659ba994c4bd/Untitled%201.png)

```python
agent.system_prompt_template
```

It's important to explain as clearly as possible the task you want to perform.
Every [`~Agent.run`] operation is independent, and since an agent is powered by an LLM, minor variations in your prompt might yield completely different results.
You can also run an agent consecutively for different tasks.


#### Code execution

A Python interpreter executes the code on a set of inputs passed along with your tools.
This should be safe because the only functions that can be called are the tools you provided (especially if it's only tools by Hugging Face) and the print function, so you're already limited in what can be executed.

The Python interpreter also doesn't allow any attribute lookup or imports (which shouldn't be needed for passing inputs/outputs to a small set of functions) so all the most obvious attacks shouldn't be an issue.

The execution will stop at any code trying to perform an illegal operation or if there is a regular Python error with the code generated by the agent.

### The system prompt

An agent, or rather the LLM that drives the agent, generates an output based on the system prompt. The system prompt can be customized and tailored to the intended task. For example, check out the system prompt of the ReAct agent.

```text
Solve the following task as best you can. You have access to the following tools:

<<tool_descriptions>>

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The value in the "action" field should belong to this list: <<tool_names>>.

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is an example of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task you are given.

And you should ALWAYS answer in the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

ALWAYS provide a 'Thought:' and an 'Action:' part. You MUST provide at least the 'Action:' part to move forward.
To provide the final answer to the task, use an action blob with "action": 'final_answer' tool.It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
{
  "action": 'final_answer',
  "action_input": "insert your final answer here"
}

Now begin!
```

The system prompt includes:
- An *introduction* that explains how the agent should behave and what tools are.
- A description of all the tools that is defined by a `<<tool_descriptions>>` token that is dynamically replaced at runtime with the tools defined/chosen by the user.
    - The tool description comes from the tool attributes, `name`, `description`, `inputs` and `output_type`,  and a simple `jinja2` template that you can refine.
- The expected output format.
- An explanation of the task to perform.

You could improve the system prompt, for example, by adding an explanation of the output format.

For maximum flexibility, you can overwrite the whole system prompt template by passing your custom prompt as an argument to the `system_prompt` parameter.

```python
from transformers import ReactJSONAgent
from transformers.agents import CalculatorTool

agent = ReactJSONAgent(tools = [CalculatorTool()], system_prompt="{your_custom_prompt}")
```

> [!WARNING]
> Please make sure to define the `<<all_tools>>` string somewhere in the `template` so the agent is aware 
of the available tools.

## Tools

A tool is an atomic function to be used by an agent.

You can for instance check the [~CalculatorTool]: it has a name, a description, input descriptions, an output type, and a `__call__` method to perform the action.

When the agent is initialized, the tool attributes are used to generate a tool description which is baked into the agent's system prompt. This lets the agent know which tools it can use and why.

### Default toolbox

Transformers comes with a default toolbox for empowering agents, that you can add to your agent upon initialization with argument `add_default_toolbox = True`:

- **Document question answering**: given a document (such as a PDF) in image format, answer a question on this document ([Donut](./model_doc/donut))
- **Image captioning**: Caption the image! ([BLIP](./model_doc/blip))
- **Image question answering**: given an image, answer a question on this image ([VILT](./model_doc/vilt))
- **Image segmentation**: given an image and a prompt, output the segmentation mask of that prompt ([CLIPSeg](./model_doc/clipseg))
- **Speech to text**: given an audio recording of a person talking, transcribe the speech into text ([Whisper](./model_doc/whisper))
- **Text to speech**: convert text to speech ([SpeechT5](./model_doc/speecht5))

You can manually use a tool by calling the [`load_tool`] function and a task to perform.


```python
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### Create a new tool

You can create your own tool for use cases not covered by the default tools from Hugging Face.
For example, let's create a tool that returns the most downloaded model for a given task from the Hub.

You'll start with the code below.

```python
from huggingface_hub import list_models

task = "text-classification"

model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(model.id)
```

This code can be converted into a class that inherits from the [`Tool`] superclass.


The custom tool needs:
- An attribute `name`, which corresponds to the name of the tool itself. The name usually describes what the tool does. Since the code returns the model with the most downloads for a task, let's name is `model_download_counter`.
- An attribute `description` is used to populate the agent's system prompt.
- An `inputs` attribute, which is a dictionary with keys `"type"` and `"description"`. It contains information that helps the Python interpreter make educated choices about the input.
- An `output_type` attribute, which specifies the output type.
- A `__call__` method which contains the inference code to be executed.


```python
from transformers import Tool
from huggingface_hub import list_models

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = (
        "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. "
        "It returns the name of the checkpoint."
    )

    inputs = {
        "task": {
            "type": str,
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = str

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

Now that the custom `HfModelDownloadsTool` class is ready, you can save it to a file named model_downloads.py and import it for use.


```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

You can also share your custom tool to the Hub by calling [`~Tool.push_to_hub`] on the tool. Make sure you've created a repository for it on the Hub and are using a token with read access.

```python
tool.push_to_hub("{your_username}/hf-model-downloads")
```

Load the tool with the [`~Tool.load_tool`] function and pass it to the `tools` parameter in your agent.

```python
from transformers import load_tool, CodeAgent

model_download_tool = load_tool("m-ric/hf-model-downloads")
agent = CodeAgent(tools=[model_download_tool], llm_engine=llm_engine)
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

You get the following:
```text
==Executing the code below:==
task = 'text-to-video'
model_name = model_download_counter(task=task)
print(f"The most downloaded model in the '{task}' task is '{model_name}'.")
```

And the output:
`"The most downloaded model in the 'text-to-video' task is 'ByteDance/AnimateDiff-Lightning'."`


### Manage agent toolbox

If you have already initialized an agent, it is inconvenient to reinitialize it from scratch with a tool you want to use. With Transformers, you can manage an agent's toolbox by adding or replacing a tool.

Let's add the `model_download_tool` to an existing agent initialized with only the default toolbox.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
agent.toolbox.add_tool(model_download_tool)
```
Now we can leverage both the new tool and the previous text-to-speech tool:

```python
agent.run(
    "Can you read out loud the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```


| **Audio**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |


> [!WARNING]
> Beware when adding tools to an agent that already works well because it can bias selection towards your tool or select another tool other than the one already defined.


Use the `agent.toolbox.update_tool()` method to replace an existing tool in the agent's toolbox.
This is useful if your new tool is a one-to-one replacement of the existing tool because the agent already knows how to perform that specific task.
Just make sure the new tool follows the same API as the replaced tool or adapt the system prompt template to ensure all examples using the replaced tool are updated.

```py
agent.toolbox.update_tool()
```

### Use gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) is a powerful library that allows using Hugging
Face Spaces as tools. It supports many existing Spaces as well as custom Spaces.

Transformers supports `gradio_tools` with the [`Tool.from_gradio`] method. For example, let's use the [`StableDiffusionPromptGeneratorTool`](https://github.com/freddyaboulton/gradio-tools/blob/main/gradio_tools/tools/prompt_generator.py) from `gradio-tools` toolkit for improving prompts to generate better images.

Import and instantiate the tool:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool

gradio_tool = StableDiffusionPromptGeneratorTool()
```

Pass the tool to the `Tool.from_gradio` method:

```python
from transformers import Tool

tool = Tool.from_gradio(gradio_tool)
```

Now you can use it just like any other tool. For example, let's improve the prompt  `a rabbit wearing a space suit`.

```python
from transformers import CodeAgent

agent = CodeAgent(llm_engine, tools=[tool], add_base_tools=True)

agent.run(
    "Improve this prompt: 'A rabbit wearing a space suit', then generate an image of it.",
)
```

The model adequately leverages the tool:
```text
==Explanation from the agent==
I will use the following  tools: `StableDiffusionPromptGenerator` to improve the prompt, then `image_generator` to generate an image according to the improved prompt.

==Code generated by the agent==
improved_prompt = StableDiffusionPromptGenerator('A rabbit wearing a space suit')
print(f"The improved prompt is {improved_prompt}.")
image = image_generator(improved_prompt)
```

Before finally generating the image:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png">


> [!WARNING]
> gradio-tools require *textual* inputs and outputs even when working with different modalities like image and audio objects. Image and audio inputs and outputs are currently incompatible.

### Use LangChain tools

To import a tool from LangChain, use the `from_langchain()` method.

```python
# Load langchain tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Initialize transformers tool from langchain tool
from transformers import Tool

tool = Tool.from_langchain(langchain_tool)
```