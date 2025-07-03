<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chat basics

Chat models are conversational models you can send and receive messages from. There are many chat models available to choose from, but in general, larger models tend to be better though that's not always the case. The model size is often included in the name, like "8B" or "70B", and it describes the number of parameters. Mixture-of-expert (MoE) models have names like "8x7B" or "141B-A35B" which means it's a 56B and 141B parameter model. You can try quantizing larger models to reduce memory requirements, otherwise you'll need ~2 bytes of memory per parameter.

Check model leaderboards like [OpenLLM](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) and [LMSys Chatbot Arena](https://chat.lmsys.org/?leaderboard) to further help you identify the best chat models for your use case. Models that are specialized in certain domains (medical, legal text, non-English languages, etc.) may sometimes outperform larger general purpose models.

> [!TIP]
> Chat with a number of open-source models for free on [HuggingChat](https://hf.co/chat/)!

This guide shows you how to quickly start chatting with Transformers from the command line, how build and format a conversation, and how to chat using the [`TextGenerationPipeline`].

## transformers CLI


### Interactive chat session

After you've [installed Transformers](./installation.md), chat with a model directly from the command line as shown below. It launches an interactive session with a model, with a few base commands listed at the start of the session.

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers-chat-cli.png"/>
</div>

You can launch the CLI with arbitrary `generate` flags, with the format `arg_1=value_1 arg_2=value_2 ...`

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct do_sample=False max_new_tokens=10
```

For a full list of options, run the command below.

```bash
transformers chat -h
```

The chat is implemented on top of the [AutoClass](./model_doc/auto), using tooling from [text generation](./llm_tutorial) and [chat](./chat_templating).


### Serving a model and using MCP tools

> [!WARNING]
> This section is experimental and subject to changes in future versions

Powering the `chat` interface, we have a server that takes user messages and returns completions. The server has a chat completion API compatible with the OpenAI SDK, so you can also quickly experiment with `transformers` models on existing aplications. To launch a server separately, use the `transformers serve` CLI:

```bash
transformers serve Menlo/Jan-nano
```

Under the hood, the `chat` CLI launches and uses `transformers serve`. This server is also an MCP client, which can receive information available MCP servers (i.e. tools), massage their information into the model prompt, and prepare calls to these tools when the model commands to do so. Naturally, this requires a model that is trained to use tools.

At the moment, MCP tool usage in `transformers` has the following constraints:
- `chat` can't handle tools, but the [`tiny-agents`](https://huggingface.co/blog/python-tiny-agents) CLI can;
- Only the `qwen` family of models is supported.

The first step to use MCP tools is to let the model know which tools are available. As an example, let's consider a `tiny-agents` configuration file with a reference to an [image generation MCP server](https://evalstate-flux1-schnell.hf.space/).

> [!TIP]
> Many Hugging Face Spaces can be used as MCP servers. You can find all compatible Spaces [here](https://huggingface.co/spaces?filter=mcp-server).

```json
{
    "model": "http://localhost:8000",
    "provider": "local",
    "servers": [
        {
            "type": "sse",
            "config": {
                "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
            }
        }
    ]
}
```

You can then launch your `tiny-agents` chat interface with the following command.

```bash
tiny-agents run path/to/your/config.json
```

If you have a server (from `transformers serve`) running in the background, you're ready to use MCP tools from a local model! For instance, here's the example of a chat session:

```bash
Agent loaded with 1 tools:
 • flux1_schnell_infer
»  Generate an image of a cat on the moon
<Tool req_0_tool_call>flux1_schnell_infer {"prompt": "a cat on the moon", "seed": 42, "randomize_seed": true, "width": 1024, "height": 1024, "num_inference_steps": 4}

Tool req_0_tool_call
[Binary Content: Image image/webp, 57732 bytes]
The task is complete and the content accessible to the User
Image URL: https://evalstate-flux1-schnell.hf.space/gradio_api/file=/tmp/gradio/3dbddc0e53b5a865ed56a4e3dbdd30f3f61cf3b8aabf1b456f43e5241bd968b8/image.webp
380576952

I have generated an image of a cat on the moon using the Flux 1 Schnell Image Generator. The image is 1024x1024 pixels and was created with 4 inference steps. Let me know if you would like to make any changes or need further assistance!
```


## TextGenerationPipeline

[`TextGenerationPipeline`] is a high-level text generation class with a "chat mode". Chat mode is enabled when a conversational model is detected and the chat prompt is [properly formatted](./llm_tutorial#wrong-prompt-format).

To start, build a chat history with the following two roles.

- `system` describes how the model should behave and respond when you're chatting with it. This role isn't supported by all chat models.
- `user` is where you enter your first message to the model.

```py
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
```

Create the [`TextGenerationPipeline`] and pass `chat` to it. For large models, setting [device_map="auto"](./models#big-model-inference) helps load the model quicker and automatically places it on the fastest device available. Changing the data type to [torch.bfloat16](./models#model-data-type) also helps save memory.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

```txt
(sigh) Oh boy, you're asking me for advice? You're gonna need a map, pal! Alright,
alright, I'll give you the lowdown. But don't say I didn't warn you, I'm a robot, not a tour guide!

So, you wanna know what's fun to do in the Big Apple? Well, let me tell you, there's a million
things to do, but I'll give you the highlights. First off, you gotta see the sights: the Statue of
Liberty, Central Park, Times Square... you know, the usual tourist traps. But if you're lookin' for
something a little more... unusual, I'd recommend checkin' out the Museum of Modern Art. It's got
some wild stuff, like that Warhol guy's soup cans and all that jazz.

And if you're feelin' adventurous, take a walk across the Brooklyn Bridge. Just watch out for
those pesky pigeons, they're like little feathered thieves! (laughs) Get it? Thieves? Ah, never mind.

Now, if you're lookin' for some serious fun, hit up the comedy clubs in Greenwich Village. You might
even catch a glimpse of some up-and-coming comedians... or a bunch of wannabes tryin' to make it big. (winks)

And finally, if you're feelin' like a real New Yorker, grab a slice of pizza from one of the many amazing
pizzerias around the city. Just don't try to order a "robot-sized" slice, trust me, it won't end well. (laughs)

So, there you have it, pal! That's my expert advice on what to do in New York. Now, if you'll
excuse me, I've got some oil changes to attend to. (winks)
```

Use the `append` method on `chat` to respond to the models message.

```py
chat = response[0]["generated_text"]
chat.append(
    {"role": "user", "content": "Wait, what's so wild about soup cans?"}
)
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

```txt
(laughs) Oh, you're killin' me, pal! You don't get it, do you? Warhol's soup cans are like, art, man!
It's like, he took something totally mundane, like a can of soup, and turned it into a masterpiece. It's
like, "Hey, look at me, I'm a can of soup, but I'm also a work of art!"
(sarcastically) Oh, yeah, real original, Andy.

But, you know, back in the '60s, it was like, a big deal. People were all about challenging the
status quo, and Warhol was like, the king of that. He took the ordinary and made it extraordinary.
And, let me tell you, it was like, a real game-changer. I mean, who would've thought that a can of soup could be art? (laughs)

But, hey, you're not alone, pal. I mean, I'm a robot, and even I don't get it. (winks)
But, hey, that's what makes art, art, right? (laughs)
```

## Performance

Transformers load models in full precision by default, and for a 8B model, this requires ~32GB of memory! Reduce memory usage by loading a model in half-precision or bfloat16 (only uses ~2 bytes per parameter). You can even quantize the model to a lower precision like 8-bit or 4-bit with [bitsandbytes](https://hf.co/docs/bitsandbytes/index).

> [!TIP]
> Refer to the [Quantization](./quantization/overview) docs for more information about the different quantization backends available.

Create a [`BitsAndBytesConfig`] with your desired quantization settings and pass it to the pipelines `model_kwargs` parameter. The example below quantizes a model to 8-bits.

```py
from transformers import pipeline, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", model_kwargs={"quantization_config": quantization_config})
```

In general, larger models are slower in addition to requiring more memory because text generation is bottlenecked by **memory bandwidth** instead of compute power. Each active parameter must be read from memory for every generated token. For a 16GB model, 16GB must be read from memory for every generated token.

The number of generated tokens/sec is proportional to the total memory bandwidth of the system divided by the model size. Depending on your hardware, total memory bandwidth can vary. Refer to the table below for approximate generation speeds for different hardware types.

| Hardware | Memory bandwidth |
|---|---|
| consumer CPU | 20-100GB/sec |
| specialized CPU (Intel Xeon, AMD Threadripper/Epyc, Apple silicon) | 200-900GB/sec |
| data center GPU (NVIDIA A100/H100) | 2-3TB/sec |

The easiest solution for improving generation speed is to either quantize a model or use hardware with higher memory bandwidth.

You can also try techniques like [speculative decoding](./generation_strategies#speculative-decoding), where a smaller model generates candidate tokens that are verified by the larger model. If the candidate tokens are correct, the larger model can generate more than one token per `forward` pass. This significantly alleviates the bandwidth bottleneck and improves generation speed.

> [!TIP]
> Parameters may not be active for every generated token in MoE models such as [Mixtral](./model_doc/mixtral), [Qwen2MoE](./model_doc/qwen2_moe.md), and [DBRX](./model_doc/dbrx). As a result, MoE models generally have much lower memory bandwidth requirements and can be faster than a regular LLM of the same size. However, techniques like speculative decoding are ineffective with MoE models because parameters become activated with each new speculated token.
