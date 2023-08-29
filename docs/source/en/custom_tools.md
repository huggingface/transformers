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

# Custom Tools and Prompts

<Tip>

If you are not aware of what tools and agents are in the context of transformers, we recommend you read the
[Transformers Agents](transformers_agents) page first.

</Tip>

<Tip warning={true}>

Transformers Agent is an experimental API that is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

Creating and using custom tools and prompts is paramount to empowering the agent and having it perform new tasks.
In this guide we'll take a look at:

- How to customize the prompt
- How to use custom tools
- How to create custom tools

## Customizing the prompt

As explained in [Transformers Agents](transformers_agents) agents can run in [`~Agent.run`] and [`~Agent.chat`] mode.
Both the `run` and `chat` modes underlie the same logic. The language model powering the agent is conditioned on a long 
prompt and completes the prompt by generating the next tokens until the stop token is reached.
The only difference between the two modes is that during the `chat` mode the prompt is extended with 
previous user inputs and model generations. This allows the agent to have access to past interactions,
seemingly giving the agent some kind of memory.

### Structure of the prompt

Let's take a closer look at how the prompt is structured to understand how it can be best customized.
The prompt is structured broadly into four parts.

- 1. Introduction: how the agent should behave, explanation of the concept of tools.
- 2. Description of all the tools. This is defined by a `<<all_tools>>` token that is dynamically replaced at runtime with the tools defined/chosen by the user.
- 3. A set of examples of tasks and their solution
- 4. Current example, and request for solution.

To better understand each part, let's look at a shortened version of how the `run` prompt can look like:

````text
I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
[...]
You can print intermediate results if it makes sense to do so.

Tools:
- document_qa: This is a tool that answers a question about a document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
- image_captioner: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to the caption and returns a text that contains the description in English.
[...]

Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.

Answer:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```

Task: "Identify the oldest person in the `document` and create an image showcasing the result as a banner."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

[...]

Task: "Draw me a picture of rivers and lakes"

I will use the following
````

The introduction (the text before *"Tools:"*) explains precisely how the model shall behave and what it should do.
This part most likely does not need to be customized as the agent shall always behave the same way.

The second part (the bullet points below *"Tools"*) is dynamically added upon calling `run` or `chat`. There are 
exactly as many bullet points as there are tools in `agent.toolbox` and each bullet point consists of the name 
and description of the tool:

```text
- <tool.name>: <tool.description>
```

Let's verify this quickly by loading the document_qa tool and printing out the name and description.

```py
from transformers import load_tool

document_qa = load_tool("document-question-answering")
print(f"- {document_qa.name}: {document_qa.description}")
```

which gives:
```text
- document_qa: This is a tool that answers a question about a document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
```

We can see that the tool name is short and precise. The description includes two parts, the first explaining 
what the tool does and the second states what input arguments and return values are expected.

A good tool name and tool description are very important for the agent to correctly use it. Note that the only
information the agent has about the tool is its name and description, so one should make sure that both 
are precisely written and match the style of the existing tools in the toolbox. In particular make sure the description
mentions all the arguments expected by name in code-style, along with the expected type and a description of what they
are.

<Tip>

Check the naming and description of the curated Transformers tools to better understand what name and 
description a tool is expected to have. You can see all tools with the [`Agent.toolbox`] property.

</Tip>

The third part includes a set of curated examples that show the agent exactly what code it should produce
for what kind of user request. The large language models empowering the agent are extremely good at 
recognizing patterns in a prompt and repeating the pattern with new data. Therefore, it is very important
that the examples are written in a way that maximizes the likelihood of the agent to generating correct,
executable code in practice. 

Let's have a look at one example:

````text
Task: "Identify the oldest person in the `document` and create an image showcasing the result as a banner."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

````

The pattern the model is prompted to repeat has three parts: The task statement, the agent's explanation of 
what it intends to do, and finally the generated code. Every example that is part of the prompt has this exact 
pattern, thus making sure that the agent will reproduce exactly the same pattern when generating new tokens.

The prompt examples are curated by the Transformers team and rigorously evaluated on a set of 
[problem statements](https://github.com/huggingface/transformers/blob/main/src/transformers/tools/evaluate_agent.py)
to ensure that the agent's prompt is as good as possible to solve real use cases of the agent.

The final part of the prompt corresponds to:
```text
Task: "Draw me a picture of rivers and lakes"

I will use the following
```

is a final and unfinished example that the agent is tasked to complete. The unfinished example
is dynamically created based on the actual user input. For the above example, the user ran:

```py
agent.run("Draw me a picture of rivers and lakes")
```

The user input - *a.k.a* the task: *"Draw me a picture of rivers and lakes"* is cast into the 
prompt template: "Task: <task> \n\n I will use the following". This sentence makes up the final lines of the 
prompt the agent is conditioned on, therefore strongly influencing the agent to finish the example 
exactly in the same way it was previously done in the examples.

Without going into too much detail, the chat template has the same prompt structure with the 
examples having a slightly different style, *e.g.*:

````text
[...]

=====

Human: Answer the question in the variable `question` about the image stored in the variable `image`.

Assistant: I will use the tool `image_qa` to answer the question on the input image.

```py
answer = image_qa(text=question, image=image)
print(f"The answer is {answer}")
```

Human: I tried this code, it worked but didn't give me a good result. The question is in French

Assistant: In this case, the question needs to be translated first. I will use the tool `translator` to do this.

```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(text=translated_question, image=image)
print(f"The answer is {answer}")
```

=====

[...]
````

Contrary, to the examples of the `run` prompt, each `chat` prompt example has one or more exchanges between the 
*Human* and the *Assistant*. Every exchange is structured similarly to the example of the `run` prompt. 
The user's input is appended to behind *Human:* and the agent is prompted to first generate what needs to be done 
before generating code. An exchange can be based on previous exchanges, therefore allowing the user to refer
to past exchanges as is done *e.g.* above by the user's input of "I tried **this** code" refers to the 
previously generated code of the agent.

Upon running `.chat`, the user's input or *task* is cast into an unfinished example of the form:
```text
Human: <user-input>\n\nAssistant:
```
which the agent completes. Contrary to the `run` command, the `chat` command then appends the completed example
to the prompt, thus giving the agent more context for the next `chat` turn.

Great now that we know how the prompt is structured, let's see how we can customize it!

### Writing good user inputs

While large language models are getting better and better at understanding users' intentions, it helps 
enormously to be as precise as possible to help the agent pick the correct task. What does it mean to be 
as precise as possible?

The agent sees a list of tool names and their description in its prompt. The more tools are added the 
more difficult it becomes for the agent to choose the correct tool and it's even more difficult to choose
the correct sequences of tools to run. Let's look at a common failure case, here we will only return 
the code to analyze it.

```py
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

agent.run("Show me a tree", return_code=True)
```

gives:

```text
==Explanation from the agent==
I will use the following tool: `image_segmenter` to create a segmentation mask for the image.


==Code generated by the agent==
mask = image_segmenter(image, prompt="tree")
```

which is probably not what we wanted. Instead, it is more likely that we want an image of a tree to be generated.
To steer the agent more towards using a specific tool it can therefore be very helpful to use important keywords that 
are present in the tool's name and description. Let's have a look.
```py
agent.toolbox["image_generator"].description
```

```text
'This is a tool that creates an image according to a prompt, which is a text description. It takes an input named `prompt` which contains the image description and outputs an image.
```

The name and description make use of the keywords "image", "prompt", "create" and "generate". Using these words will most likely work better here. Let's refine our prompt a bit.

```py
agent.run("Create an image of a tree", return_code=True)
```

gives:
```text
==Explanation from the agent==
I will use the following tool `image_generator` to generate an image of a tree.


==Code generated by the agent==
image = image_generator(prompt="tree")
```

Much better! That looks more like what we want. In short, when you notice that the agent struggles to 
correctly map your task to the correct tools, try looking up the most pertinent keywords of the tool's name
and description and try refining your task request with it.

### Customizing the tool descriptions

As we've seen before the agent has access to each of the tools' names and descriptions. The base tools 
should have very precise names and descriptions, however, you might find that it could help to change the 
the description or name of a tool for your specific use case. This might become especially important 
when you've added multiple tools that are very similar or if you want to use your agent only for a certain 
domain, *e.g.* image generation and transformations.

A common problem is that the agent confuses image generation with image transformation/modification when 
used a lot for image generation tasks, *e.g.*
```py
agent.run("Make an image of a house and a car", return_code=True)
```
returns
```text
==Explanation from the agent== 
I will use the following tools `image_generator` to generate an image of a house and `image_transformer` to transform the image of a car into the image of a house.

==Code generated by the agent==
house_image = image_generator(prompt="A house")
car_image = image_generator(prompt="A car")
house_car_image = image_transformer(image=car_image, prompt="A house")
```

which is probably not exactly what we want here. It seems like the agent has a difficult time 
to understand the difference between `image_generator` and `image_transformer` and often uses the two together.

We can help the agent here by changing the tool name and description of `image_transformer`. Let's instead call it `modifier`
to disassociate it a bit from "image" and "prompt":
```py
agent.toolbox["modifier"] = agent.toolbox.pop("image_transformer")
agent.toolbox["modifier"].description = agent.toolbox["modifier"].description.replace(
    "transforms an image according to a prompt", "modifies an image"
)
```

Now "modify" is a strong cue to use the new image processor which should help with the above prompt. Let's run it again.

```py
agent.run("Make an image of a house and a car", return_code=True)
```

Now we're getting:
```text
==Explanation from the agent==
I will use the following tools: `image_generator` to generate an image of a house, then `image_generator` to generate an image of a car.


==Code generated by the agent==
house_image = image_generator(prompt="A house")
car_image = image_generator(prompt="A car")
```

which is definitely closer to what we had in mind! However, we want to have both the house and car in the same image. Steering the task more toward single image generation should help:

```py
agent.run("Create image: 'A house and car'", return_code=True)
```

```text
==Explanation from the agent==
I will use the following tool: `image_generator` to generate an image.


==Code generated by the agent==
image = image_generator(prompt="A house and car")
```

<Tip warning={true}>

Agents are still brittle for many use cases, especially when it comes to 
slightly more complex use cases like generating an image of multiple objects.
Both the agent itself and the underlying prompt will be further improved in the coming 
months making sure that agents become more robust to a variety of user inputs.

</Tip>

### Customizing the whole prompt

To give the user maximum flexibility, the whole prompt template as explained in [above](#structure-of-the-prompt)
can be overwritten by the user. In this case make sure that your custom prompt includes an introduction section, 
a tool section, an example section, and an unfinished example section. If you want to overwrite the `run` prompt template, 
you can do as follows:

```py
template = """ [...] """

agent = HfAgent(your_endpoint, run_prompt_template=template)
```

<Tip warning={true}>

Please make sure to have the `<<all_tools>>` string and the `<<prompt>>` defined somewhere in the `template` so that the agent can be aware 
of the tools, it has available to it as well as correctly insert the user's prompt.

</Tip>

Similarly, one can overwrite the `chat` prompt template. Note that the `chat` mode always uses the following format for the exchanges:
```text
Human: <<task>>

Assistant:
```

Therefore it is important that the examples of the custom `chat` prompt template also make use of this format.
You can overwrite the `chat` template at instantiation as follows.

```
template = """ [...] """

agent = HfAgent(url_endpoint=your_endpoint, chat_prompt_template=template)
```

<Tip warning={true}>

Please make sure to have the `<<all_tools>>` string defined somewhere in the `template` so that the agent can be aware 
of the tools, it has available to it.

</Tip>

In both cases, you can pass a repo ID instead of the prompt template if you would like to use a template hosted by someone in the community. The default prompts live in [this repo](https://huggingface.co/datasets/huggingface-tools/default-prompts) as an example.

To upload your custom prompt on a repo on the Hub and share it with the community just make sure:
- to use a dataset repository
- to put the prompt template for the `run` command in a file named `run_prompt_template.txt`
- to put the prompt template for the `chat` command in a file named `chat_prompt_template.txt`

## Using custom tools

In this section, we'll be leveraging two existing custom tools that are specific to image generation:

- We replace [huggingface-tools/image-transformation](https://huggingface.co/spaces/huggingface-tools/image-transformation),
  with [diffusers/controlnet-canny-tool](https://huggingface.co/spaces/diffusers/controlnet-canny-tool) 
  to allow for more image modifications.
- We add a new tool for image upscaling to the default toolbox: 
  [diffusers/latent-upscaler-tool](https://huggingface.co/spaces/diffusers/latent-upscaler-tool) replace the existing image-transformation tool.

We'll start by loading the custom tools with the convenient [`load_tool`] function:

```py
from transformers import load_tool

controlnet_transformer = load_tool("diffusers/controlnet-canny-tool")
upscaler = load_tool("diffusers/latent-upscaler-tool")
```

Upon adding custom tools to an agent, the tools' descriptions and names are automatically
included in the agents' prompts. Thus, it is imperative that custom tools have
a well-written description and name in order for the agent to understand how to use them.
Let's take a look at the description and name of `controlnet_transformer`:

```py
print(f"Description: '{controlnet_transformer.description}'")
print(f"Name: '{controlnet_transformer.name}'")
```

gives 
```text
Description: 'This is a tool that transforms an image with ControlNet according to a prompt. 
It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the modified image.'
Name: 'image_transformer'
```

The name and description are accurate and fit the style of the [curated set of tools](./transformers_agents#a-curated-set-of-tools).
Next, let's instantiate an agent with `controlnet_transformer` and `upscaler`:

```py
tools = [controlnet_transformer, upscaler]
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=tools)
```

This command should give you the following info:

```text
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

```py
print("\n".join([f"- {a}" for a in agent.toolbox.keys()]))
```

```text
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

```py
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png"
)
```

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

For the task `text-classification`, this returns `'facebook/bart-large-mnli'`, for `translation` it returns `'t5-base`.

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
        "It takes the name of the category (such as text-classification, depth-estimation, etc), and "
        "returns the name of the checkpoint."
    )

    inputs = ["text"]
    outputs = ["text"]

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

In order to use it in the agent, simply pass it in the `additional_tools` parameter of the agent initialization method:

```python
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

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
from transformers import HfAgent, load_tool

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.toolbox["image-transformation"] = load_tool("diffusers/controlnet-canny-tool")
```

<Tip>

Beware when replacing tools with others! This will also adjust the agent's prompt. This can be good if you have a better
prompt suited for the task, but it can also result in your tool being selected way more than others or for other
tools to be selected instead of the one you have defined.

</Tip>

## Leveraging gradio-tools

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
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

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

## Future compatibility with Langchain

We love Langchain and think it has a very compelling suite of tools. In order to handle these tools,
Langchain requires *textual* inputs and outputs, even when working with different modalities.
This is often the serialized version (i.e., saved to disk) of the objects.

This difference means that multi-modality isn't handled between transformers-agents and langchain.
We aim for this limitation to be resolved in future versions, and welcome any help from avid langchain
users to help us achieve this compatibility.

We would love to have better support. If you would like to help, please 
[open an issue](https://github.com/huggingface/transformers/issues/new) and share what you have in mind.
