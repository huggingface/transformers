# Do anything with Transformers

Transformers support all modalities and has many models performing many different types of tasks. But it can get confusing to mix and match them to solve the problem at hand, which is why we have developed a new API of **tools** and **agents**. Given a prompt in natural language and a set of tools, an agent will determine the right code to run with the tools and chain them properly to give you the result you expected.

Let's start with examples!

## Examples

First we need an agent, which is a fancy word to design a LLM tasked with writing the code you will need. We support the traditional openai LLMs but you should really try the opensource alternatives developed by the community which:
- clearly state the data they have been trained on
- you can run on your own cloud or hardware
- have built-in versioning 

<!--TODO for the release we should have a publicly available agent and if token is none, we grab the HF token-->

```py
from transformers.tools import EndpointAgent

agent = EndpointAgent(
    url_endpoint=your_endpoint,
    token=your_hf_token,
)

# from transformers.tools import OpenAiAgent

# agent = OpenAiAgent(api_key=your_openai_api_key)
```

### Task 1: Classifying text in (almost) any language

Now to execute a given task, we need to pick a set of tools in `transformers` and send them to our agent. Let's say you want to classify a text in a non-English language, and you have trouble finding a model trained in that language. You can pick a translation tool and a standard text classification tool:

```py
from transformers.tools import TextClassificationTool, TranslationTool

tools = [TextClassificationTool(), TranslationTool(src_lang="fra_Latn", tgt_lang="eng_Latn")]
```

then you just run this by your agent:

```py
agent.run(
    "Determine if the following `text` (in French) is positive or negative.",
    tools=tools,
    text="J'aime beaucoup Hugging Face!"
)
```

Note that you can send any additional inputs in a variable that you named in your prompt (between backticks because it helps the LLM). For text inputs, you can just put them in the prompt:

```py
agent.run(
    """Determine if the following text: "J'aime beaucoup Hugging Face!" (in French) is positive or negative.""",
    tools=tools,
)
```

In both cases, you should see the agent generate code using your set of tools that is then executed to provide you the answer you were looking for. Neat!

If you don't have the hardware to run the models translating and classifying the text, you can use the inference API by selecting a remote tool:


```py
from transformers.tools import RemoteTextClassificationTool, TranslationTool

tools = [RemoteTextClassificationTool(), TranslationTool(src_lang="fra_Latn", tgt_lang="eng_Latn")]

agent.run(
    "Determine if the following `text` (in French) is positive or negative.",
    tools=tools,
    text="J'aime beaucoup Hugging Face!"
)
```

This was still all text-based. Let's now get to something more exciting, combining vision and speech

## Example 2:

Let's say we want to hear out loud what is in a given image. There are models that do image-captioning in Transformers, and other models that generate speech from text, but how to combine them? Quite easily:

<!--TODO add the audio reader tool once it exists-->

```py
import requests
from PIL import Image
from transformers.tools import ImageCaptioningTool, TextToSpeechTool

tools = [ImageCaptioningTool(), TextToSpeechTool()]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

speech = agent.run(
    "Tell me out loud what the `image` contains.",
    tools=tools,
    image=image
)
```

Note that here you have to pass your input as a separate variable since you can't really embed your image in the text.

In all those examples, we have been using the default checkpoint for a given tool, but you can specify the one you want! For instance, the image-captioning tool uses BLIP by default, but let's upgrade to BLIP-2

<!--TODO Once it works, use the inference API for BLIP-2 here as it's heavy-->

```py
tools = [ImageCaptioningTool("Salesforce/blip2-opt-2.7b"), TextToSpeechTool()]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

speech = agent.run(
    "Tell me out loud what the `image` contains.",
    tools=tools,
    image=image
)
```

Add more examples?

## How does it work ?

LLMs are pretty good at generating small samples of code, so this API takes advantage of that by prompting the LLM to give a small sample of code performing a task with a set of tools. This prompt is then completed by the task you give your agent and the description of the tools you give it. This way it gets access to the doc of the tools you are using, especially their expected inputs and outputs and can generate the relevant code.

This is using brand-new tools and not pipelines, because the agent writes better code with very atomic tools. Pipelines are more refactored and often combine several tasks in one. Tools are really meant to be focused one very simple task only.

This code is then executed with our small Python interpreter on the set of inputs passed along with your tools. I hear you screaming "Arbitrary code execution!" in the back, but calm down a minute and let me explain.

The only functions that can be called are the tools you provided and the print function, so you're already limited in what can be executed. You should be safe if it's limited to Hugging Face tools. Then we don't allow any attribute lookup or imports (which shouldn't be needed anyway for passing along inputs/outputs to a small set of functions) so all the most obvious attacks (and you'd need to prompt the LLM to output them anyway) shouldn't be an issue. If you want to be on the super safe side, you can execute the `run()` method with the additional argument `return_code=True`, in which case the agent will just return the code to execute and you can decide whether to do it or not.

Note that LLMs are still not *that* good at producing the small amount of code to chain the tools, so we added some logic to fix typos during the evaluation: there are often misnamed variable names or dictionary keys.

The execution will stop at any line trying to perform an illegal operation or if there is a regular Python error with the code generated by the agent. 

## Future developments

We hope you're as excited by this new API as we are. Here are a few things we are thinking of adding next if we see the community is interested:
- Make the agent pick the tools itself in a first step.
- Make the run command more chat-based, so you can copy-paste any error message you see in a next step to have the LLM fix its code, or ask for some improvements.
- Add support for more type of agents

