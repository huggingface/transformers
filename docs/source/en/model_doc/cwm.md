<-- Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->

# Code World Model (CWM)

## Overview

The Code World Model (CWM) model was proposed in [CWM: An Open-Weights LLM for Research on Code
Generation with World Models](https://ai.facebook.com/research/publications/cwm) by Meta FAIR CodeGen Team.
CWM is an LLM for code generation and reasoning about code that has, in particular, been trained
to better represent and reason about how code and commands affect the state of a program or system.
Specifically, we mid-trained CWM on a large number of observation-action trajectories from Python
execution traces and agentic interactions in containerized environments. We post-trained with
extensive multi-task RL in verifiable coding, math, and multi-turn software engineering environments.

The abstract from the paper is the following:

> *We release Code World Model (CWM), a 32-billion-parameter open-weights LLM, to advance research
on code generation with world models. To improve code understanding beyond what can be learned
from training on static code alone, we mid-train CWM on a large amount of observation-action
trajectories from Python interpreter and agentic Docker environments, and perform extensive multi-
task reasoning RL in verifiable coding, math, and multi-turn software engineering environments. With
CWM, we provide a strong testbed for researchers to explore the opportunities world modeling affords
for improving code generation with reasoning and planning in computational environments. We
present first steps of how world models can benefit agentic coding, enable step-by-step simulation of
Python code execution, and show early results of how reasoning can benefit from the latter. CWM is
a dense, decoder-only LLM trained with a context size of up to 131 k tokens. Independent of its world
modeling capabilities, CWM offers strong performance on general coding and math tasks: it reaches
pass@1 scores of 65.8 % on SWE-bench Verified (with test-time scaling), 68.9 % on LiveCodeBench,
96.6 % on Math-500, and 76.0 % on AIME 2024. To support further research on code world modeling,
we release model checkpoints after mid-training, SFT, and RL.*

Tips:

CWM requires a dedicated system prompt to function optimally during inference. Without proper prompt
configuration, CWM's output quality may be significantly degraded. The following serves as the default
system prompt for reasoning tasks. For agentic workflows, append the relevant tool specifications
after this base prompt. Checkout the original code repository for more details.

```text
You are a helpful AI assistant. You always reason before responding, using the following format:

<think>
your internal reasoning
</think>
your external response
```

This model was contributed by [Meta FAIR CodeGen Team](https://huggingface.co/facebook).
The original code can be found [here](https://github.com/facebookresearch/cwm).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'facebook/cwm'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

system_prompt = """
You are a helpful AI assistant. You always reason before responding, using the following format:

<think>
your internal reasoning
</think>
your external response
""".strip()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Write a haiku about recursion in programming."}
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
    preserve_previous_think=True, # Switches between keeping thinking blocks from previous messages or not. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
print(tokenizer.decode(output_ids))
```

<details>
<summary>Produces the following output:</summary>

    Okay, I need to write a haiku about recursion in programming. Let's start by recalling what recursion is. Recursion is a programming technique where a function calls itself in its own definition, often used to solve problems that can be broken down into smaller, similar sub-problems.

    A haiku is a three-line poem with a syllable count of 5-7-5. So, I need to capture the essence of recursion in a concise and poetic way. Let's think about the key elements of recursion: the function calling itself, the base case that stops the recursion, and the idea of breaking down problems into smaller parts.

    Maybe I can use imagery related to mirrors or reflections, since recursion often involves a function reflecting or repeating itself. Alternatively, I could use natural imagery like a spiral or a tree, which suggests self-similar patterns.

    Let me try to come up with some ideas. For example, "Function calls itself" could be a starting point. But how to make that into a haiku? Maybe something like:

    Function calls itself
    Base case stops the endless loop
    Spiral of code

    Hmm, that's a start. But maybe I can make it more poetic. Let's think about the idea of a function calling itself as a kind of infinite loop, but with a base case that stops it. That's like a spiral that unwinds at the center.

    Another angle: recursion is often used for problems that have a recursive structure, like tree traversals or factorial calculations. Maybe I can use a natural image to represent that.

    How about:

    Branches split again
    Leaves unfold in endless dance
    Roots hold the base case

    That's a bit more poetic. The branches splitting represent the recursive calls, the leaves unfolding could represent the processing of each sub-problem, and the roots holding the base case are what stops the recursion.

    Alternatively, maybe something simpler:

    Mirror within mirror
    Reflections infinite deep
    Base case breaks the glass

    This uses the mirror metaphor to represent the function calling itself, with the base case being the point where the recursion stops, breaking the infinite reflection.

    I think I like the mirror idea better. Let me refine that:

    Mirror within mirror
    Infinite reflections dance
    Base case shatters glass

    Yes, that seems to capture the idea of recursion well. The mirror represents the function calling itself, the infinite reflections are the recursive calls, and the base case shattering the glass is what stops the recursion.

    Let me check the syllable count:

    Mirror within mirror (5 syllables: Mir-ror-with-in-mir-ror)
    Infinite reflections dance (7 syllables: In-fi-nite-ref-lec-tions-dance)
    Base case shatters glass (5 syllables: Base-case-shat-ters-glass)

    Yes, that fits the 5-7-5 structure. I think that's a good haiku about recursion.
    </think>
    Mirror within mirror
    Infinite reflections dance
    Base case shatters glass<|eot_id|>

</details>

## CwmConfig

[[autodoc]] CwmConfig

## CwmPreTrainedModel

[[autodoc]] CwmPreTrainedModel
    - forward

## CwmModel

[[autodoc]] CwmModel
    - forward

## CwmForCausalLM

[[autodoc]] CwmForCausalLM
