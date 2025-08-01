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

## chat CLI

After you've [installed Transformers](./installation), chat with a model directly from the command line as shown below. It launches an interactive session with a model, with a few base commands listed at the start of the session.

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

The chat is implemented on top of the [AutoClass](./model_doc/auto), using tooling from [text generation](./llm_tutorial) and [chat](./chat_templating). It uses the `transformers serve` CLI under the hood ([docs](./serving.md#serve-cli)).


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

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
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
> Parameters may not be active for every generated token in MoE models such as [Mixtral](./model_doc/mixtral), [Qwen2MoE](./model_doc/qwen2_moe), and [DBRX](./model_doc/dbrx). As a result, MoE models generally have much lower memory bandwidth requirements and can be faster than a regular LLM of the same size. However, techniques like speculative decoding are ineffective with MoE models because parameters become activated with each new speculated token.
