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
# Agents

Large Language Models (LLMs) trained to perform [causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) can tackle a wide range of tasks, but they often struggle with basic tasks like logic, calculation, and search. The worst scenario is when they perform poorly in a domain, such as math, yet still attempt to handle all the calculations themselves.

One approach to overcome this weakness is to embed the LLM into a system where it has the ability to call tools: such a system is called an LLM Agent.

The augmentation with tools gives the LLM superpowers. See for youselves:

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

Here you will learn:

- what an agent is,
- how to build one using `transformers` ,
- and how to use it.

## Quickstart

### What is an agent?

The definition of LLM Agents is quite broad: all systems that use LLMs as their engine, and have the possibility to perform actions on their environment based on observations.

When trying to accomplish a task, an agent can be either programmed to:

- devise a series of actions and run them all at once, like our `CodeAgent`
- or plan and execute them one by one to wait for the outcome of the each action before launching the next one, thus following a Reflexion ⇒ Action ⇒ Perception cycle. Our `ReactAgent` implements this latter framework.

![Framework of a React Agent](Doc%20agents%20ac753b9a3b934eaba22f659ba994c4bd/Untitled.png)

Framework of a React Agent

### What is a tool?

A tool is an atomic function to be used by an agent.

You can for instance check the [~CalculatorTool], which we use

It has a name, a description, input and output descriptions, and a __call__ method that will perform the action.

Upon initialization of the agent system, the tool attributes are used to generate a tool description, then baked into the agent’s `system_prompt`  to let it know which tools it can use and why.

---

Here is an example adapted from [our blog post](https://huggingface.co/blog/open-source-llms-as-agents) showing how powerful the ReAct agent framework can be to solve problems: Mixtral-8x7B answers the question:

> “**Which city has a larger population, Guiyang or Tacheng?”**
> 

```python
***Thought:** To answer this question, I need to find the current populations of both Guiyang and Tacheng. I will use the search tool to find this information.
**Action:**
{
  "action": "search",
  "action_input": "current population of Guiyang"
}
**Observation:** 4.697 million (2016)*

***Thought:*** *I have obtained the population of Guiyang, which is 4.697 million (as of 2016). Now, I will use the search tool again to find the population of Tacheng.
**Action:**
{
  "action": "search",
  "action_input": "current population of Tacheng"
}
**Observation:**
Population: Xinjiang: Tacheng data was reported at 1,110.000 Person th in 2021. This records a decrease from the previous number of 1,138.638 Person th for 2020.
Population: Xinjiang: Tacheng data is updated yearly, averaging 1,023.000 Person th from Dec 2005 to 2021, with 17 observations.

**Thought:**
I have obtained the population of Tacheng, which is approximately 1.11 million (as of 2021). Comparing the two populations, Guiyang has a larger population than Tacheng.
I now know the final answer.
**Action:
{
	"action": "final_answer",
	"action_input": "**Guiyang has a larger population, which is approximately 4.697 million (as of 2016), compared to Tacheng's population of approximately 1.11 million (as of 2021)."
}
**Observation:**
Guiyang has a larger population, which is approximately 4.697 million (as of 2016), compared to Tacheng's population of approximately 1.11 million (as of 2021).*
```

### How can I build an agent?

To initialize an agent, you need these arguments:

- an LLM to power your agent - the agent is not exactly the LLM, it’s more like the agent is a program that uses an LLM as its engine.
- a system prompt: what the LLM engine will be prompted with to generate its output
- a toolbox from which the agent pick tools to execute
- a parser to extract from the LLM output which tools are to call and with which arguments

To start with, please install the `agents` extras in order to install all default dependencies.

```bash
pip install transformers[agents]
```

To build your LLM engine, you have to define a `llm_callable` method, that will be given text and return text. This callable needs to accept a `stop` argument defining stop sequences indicating when to stop generating its output. For instance as follows:

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

def llm_callable(query: str, stop=["Task"]) -> str:
    response = client.text_generation(query, stop_sequences=stop, return_full_text=False, max_new_tokens=1000)
    for stop_seq in stop:
        if response[-len(stop_seq) :] == stop_seq:
            response = response[: -len(stop_seq)]
    return response
```

You could use any other `llm_callable` method, as long as:

- it takes a `str` as input and returns a `str`
- it will stop generating output before the sequences passed in argument `stop`

Then you can define your agent and run it.

```python
from transformers import CodeAgent

agent = CodeAgent(llm_callable=llm_callable)

agent.run("Draw me a picture of rivers and lakes")
```

The system prompt, toolbox and output parser were automatically defined here, but you can easily inspect them.

![Untitled](Doc%20agents%20ac753b9a3b934eaba22f659ba994c4bd/Untitled%201.png)

```python
agent.system_prompt
```

[show prompt]

Note that your agent is powered by a LLM, so small variations in your prompt might yield completely different results. It's important to explain as clearly as possible the task you want to perform. We go more in-depth on how to write good prompts [here](custom_tools#writing-good-user-inputs).

Every [`~Agent.run`] operation is independent, so you can run it several times in a row with different tasks.

Here is an example of how powerful the agent setup can be:

[(show agent run to solve a GAIA task)]

## Implementation of agents

### ReactAgent

This is the go-to agent to solve reasoning tasks, since the ReAct framework makes it really efficient to think on the basis of its previous observations.

### CodeAgent

This class is the one to use when trying to execute multimodal tasks, since it natively handles different input and output types for its tools. It has a planning step, then generates python code to execute all actions at once.

### #### Code execution?!

This code is then executed with our small Python interpreter on the set of inputs passed along with your tools. We hear you screaming "Arbitrary code execution!" in the back, but let us explain why that is not the case.

The only functions that can be called are the tools you provided and the print function, so you're already limited in what can be executed. You should be safe if it's limited to Hugging Face tools. 

Then, we don't allow any attribute lookup or imports (which shouldn't be needed anyway for passing along  inputs/outputs to a small set of functions) so all the most obvious attacks (and you'd need to prompt the LLM  to output them anyway) shouldn't be an issue.

The execution will stop at any line trying to perform an illegal operation or if there is a regular Python error with the code generated by the agent.

### Default toolbox

We curated a set of tools that can empower agents. Here is an updated list of the tools we have integrated in `transformers`:

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

```python
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

# Getting the best out of your agents

To be performant, your agents should be tailored to the task you intend to give them.

The things that you can easily customize in a `transformers` agent are:

- the system prompt
- the toolbox

We will now see how to optimize the usage of both of these!

### Customizing the prompt

As we’ve seen above, the LLM generates its output based on a prompt. Let’s take a look at our system prompt for the React agent:

[insert prompt]

```python
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

The prompt has these parts:

- Introduction: how the agent should behave, explanation of the concept of tools.
- Description of all the tools. This is defined by a `<<tool_descriptions>>` token that is dynamically replaced at runtime with the tools defined/chosen by the user.
    - The description is built based on the tool attributes `name`, `description`, `inputs` and `output_type`  with a simple `jinja2` template, that you can refine.
- Clarifications on the output format
- Introduction of the task at hand.

This could be improved: for instance by adding explanations of the output format.

For maximum flexibility, you can overwrite the whole prompt template as explained above by passing your custom prompt as an argument:

```python
agent = ReactAgent(llm_callable, system_prompt=your_custom_prompt)
```

```markdown
<Tip warning={true}>

Please make sure to have the `<<all_tools>>` string defined somewhere in the `template` so that the agent can be aware 
of the tools, it has available to it.

</Tip>
```

### Using custom tools

In this section, we'll be leveraging two existing custom tools that are specific to image generation:

- We replace [huggingface-tools/image-transformation](https://huggingface.co/spaces/huggingface-tools/image-transformation),
  with [diffusers/controlnet-canny-tool](https://huggingface.co/spaces/diffusers/controlnet-canny-tool) to allow for more image modifications.
- We add a new tool for image upscaling to the default toolbox: 
  [diffusers/latent-upscaler-tool](https://huggingface.co/spaces/diffusers/latent-upscaler-tool) replace the existing image-transformation tool.

We'll start by loading the custom tools with the convenient `load_tool` function:

```python
from transformers import load_tool

controlnet_transformer = load_tool("diffusers/controlnet-canny-tool")
upscaler = load_tool("diffusers/latent-upscaler-tool")
```

Upon adding custom tools to an agent, the tools' complete description is automatically
included in the agents' prompts. Thus, it is imperative that custom tools have
a well-written description and name in order for the agent to understand how to use them.
Let's take a look at the description and name of `controlnet_transformer`:

```python
print(f"Description: '{controlnet_transformer.description}'")
print(f"Name: '{controlnet_transformer.name}'")
```

gives 

```python
Description: 'This is a tool that transforms an image with ControlNet according to a prompt. 
It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the modified image.'
Name: 'image_transformer'
```

The name and description are accurate and fit the style of the [curated set of tools](./transformers_agents#a-curated-set-of-tools).

Next, let's instantiate an agent with `controlnet_transformer` and `upscaler`:

```python
tools = [controlnet_transformer, upscaler]
agent = CodeAgent(llm_callable, tools=tools)
```

This command should give you the following info:

```python
image_transformer has been replaced by <transformers_modules.diffusers.controlnet-canny-tool.bd76182c7777eba9612fc03c0
8718a60c0aa6312.image_transformation.ControlNetTransformationTool object at 0x7f1d3bfa3a00> as provided in `additional_tools`
```

The set of curated tools already has an `image_transformer` tool which is hereby replaced with our custom tool.

<Tip>

Overwriting existing tools can be beneficial if we want to use a custom tool exactly for the same task as an existing tool 
because the agent is well-versed in using the specific task. Beware that the custom tool should follow the exact same API 
as the overwritten tool in this case, or you should adapt the prompt template to make sure all examples using that
tool are updated.

</Tip>

The upscaler tool was given the name `image_upscaler` which is not yet present in the default toolbox and is therefore simply added to the list of tools.
You can always have a look at the toolbox that is currently available to the agent via the `agent.toolbox` attribute:

```python
print("\n".join([f"- {a}" for a in agent.toolbox.keys()]))
```

```python
- document_qa
- image_captioner
- image_qa
- image_segmenter
- transcriber
- summarizer
- text_classifier
- text_qa
- text_reader
- translator
- image_transformer
- text_downloader
- image_generator
- video_generator
- image_upscaler
```

Note how `image_upscaler` is now part of the agents' toolbox.

Let's now try out the new tools! We will re-use the image we generated in [Transformers Agents Quickstart](./transformers_agents#single-execution-run).

from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png"
)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

Let's transform the image into a beautiful winter landscape:

```py
image = agent.run("Transform the image: 'A frozen lake and snowy forest'", image=image)
```

```text
==Explanation from the agent==
I will use the following tool: `image_transformer` to transform the image.

==Code generated by the agent==
image = image_transformer(image, prompt="A frozen lake and snowy forest")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter.png" width=200> 

The new image processing tool is based on ControlNet which can make very strong modifications to the image.
By default the image processing tool returns an image of size 512x512 pixels. Let's see if we can upscale it.

```py
image = agent.run("Upscale the image", image)
```

```text
==Explanation from the agent==
I will use the following tool: `image_upscaler` to upscale the image.

==Code generated by the agent==
upscaled_image = image_upscaler(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter_upscale.png" width=400> 

The agent automatically mapped our prompt "Upscale the image" to the just added upscaler tool purely based on the description and name of the upscaler tool 
and was able to correctly run it.

Next, let's have a look at how you can create a new custom tool.

### Adding new tools

In this section, we show how to create a new tool that can be added to the agent.

#### Creating a new tool

We'll first start by creating a tool. We'll add the not-so-useful yet fun task of fetching the model on the Hugging Face
Hub with the most downloads for a given task.

We can do that with the following code:

```python
from huggingface_hub import list_models

task = "text-classification"

model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(model.id)
```

For the task `text-classification`, this returns `'facebook/bart-large-mnli'`, for `translation` it returns `'google-t5/t5-base`.

How do we convert this to a tool that the agent can leverage? All tools depend on the superclass `Tool` that holds the
main attributes necessary. We'll create a class that inherits from it:

```python
from transformers import Tool

class HFModelDownloadsTool(Tool):
    pass
```

This class has a few needs:
- An attribute `name`, which corresponds to the name of the tool itself. To be in tune with other tools which have a
  performative name, we'll name it `model_download_counter`.
- An attribute `description`, which will be used to populate the prompt of the agent.
- `inputs` and `outputs` attributes. Defining this will help the python interpreter make educated choices about types,
  and will allow for a gradio-demo to be spawned when we push our tool to the Hub. They're both a list of expected
  values, which can be `text`, `image`, or `audio`.
- A `__call__` method which contains the inference code. This is the code we've played with above!

Here's what our class looks like now:

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
        "name": {
            "type": str,
            "description": "the name of the category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = str

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

We now have our tool handy. Save it in a file and import it from your main script. Let's name this file
`model_downloads.py`, so the resulting import code looks like this:

```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

In order to let others benefit from it and for simpler initialization, we recommend pushing it to the Hub under your 
namespace. To do so, just call `push_to_hub` on the `tool` variable:

```python
tool.push_to_hub("hf-model-downloads")
```

You now have your code on the Hub! Let's take a look at the final step, which is to have the agent use it.

#### Having the agent use the tool

We now have our tool that lives on the Hub which can be instantiated as such (change the user name for your tool):

```python
from transformers import load_tool

tool = load_tool("lysandre/hf-model-downloads")
```

In order to use it in the agent, simply pass it to the agent initialization method:

```python
from transformers import CodeAgent

agent = CodeAgent(llm_callable, tools=[tool])

agent.run(
    "Can you read out loud the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```
which outputs the following:
```text
==Code generated by the agent==
model = model_download_counter(task="text-to-video")
print(f"The model with the most downloads is {model}.")
audio_model = text_reader(model)

==Result==
The model with the most downloads is damo-vilab/text-to-video-ms-1.7b.
```

and generates the following audio.

| **Audio**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |

<Tip>

Depending on the LLM, some are quite brittle and require very exact prompts in order to work well. Having a well-defined
name and description of the tool is paramount to having it be leveraged by the agent.

</Tip>

### Replacing existing tools

Replacing existing tools can be done simply by assigning a new item to the agent's toolbox. Here's how one would do so:

```python
from transformers import CodeAgent, load_tool

agent = CodeAgent(llm_callable)
agent.toolbox["image-transformation"] = load_tool("diffusers/controlnet-canny-tool")
```

<Tip>

Beware when replacing tools with others! This will also adjust the agent's prompt. This can be good if you have a better
prompt suited for the task, but it can also result in your tool being selected way more than others or for other
tools to be selected instead of the one you have defined.

</Tip>

### Leveraging gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) is a powerful library that allows using Hugging
Face Spaces as tools. It supports many existing Spaces as well as custom Spaces to be designed with it.

We offer support for `gradio_tools` by using the `Tool.from_gradio` method. For example, we want to take
advantage of the `StableDiffusionPromptGeneratorTool` tool offered in the `gradio-tools` toolkit so as to
improve our prompts and generate better images.

We first import the tool from `gradio_tools` and instantiate it:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool

gradio_tool = StableDiffusionPromptGeneratorTool()
```

We pass that instance to the `Tool.from_gradio` method:

```python
from transformers import Tool

tool = Tool.from_gradio(gradio_tool)
```

Now we can manage it exactly as we would a usual custom tool. We leverage it to improve our prompt
` a rabbit wearing a space suit`:

```python
from transformers import CodeAgent

agent = CodeAgent(llm_callable, additional_tools=[tool])

agent.run("Generate an image of the `prompt` after improving it.", prompt="A rabbit wearing a space suit")
```

The model adequately leverages the tool:
```text
==Explanation from the agent==
I will use the following  tools: `StableDiffusionPromptGenerator` to improve the prompt, then `image_generator` to generate an image according to the improved prompt.

==Code generated by the agent==
improved_prompt = StableDiffusionPromptGenerator(prompt)
print(f"The improved prompt is {improved_prompt}.")
image = image_generator(improved_prompt)
```

Before finally generating the image:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png">

<Tip warning={true}>

gradio-tools requires *textual* inputs and outputs, even when working with different modalities. This implementation
works with image and audio objects. The two are currently incompatible, but will rapidly become compatible as we
work to improve the support.

</Tip>

### Importing tools from LangChain

You can also use method `from_langchain`to quickly initialize a tool from a LangChain tool.
This goes as follows: 

```python
# Load langchain tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Initialize transformers tool from langchain tool
from transformers.tools.base import Tool

tool = Tool.from_langchain(langchain_tool)
```