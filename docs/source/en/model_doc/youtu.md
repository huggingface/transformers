<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-31 and added to Hugging Face Transformers on 2026-01-26.*

# Youtu-LLM

## Overview

The Youtu-LLM model was proposed in [Youtu-LLM Technical Report](https://huggingface.co/papers/2512.24618) by Tencent Youtu Team.

The abstract from the paper is the following:
We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities.

### Usage tips

The model uses Multi-head Latent Attention (MLA) architectures for efficient inference. The model can be used for various language tasks after being pre-trained on approximate 11 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages. The following example demonstrates how to load the model, enable Reasoning Mode, and use the re module to parse the "Thought Process" and the "Final Answer" from the output.

```python
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configure Model
model_id = "tencent/Youtu-LLM-2B"

# 2. Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

# 3. Construct Dialogue Input
prompt = "Hello"
messages = [{"role": "user", "content": prompt}]

# Use apply_chat_template to construct input; set enable_thinking=True to activate Reasoning Mode
input_ids = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt",
    enable_thinking=True
).to(model.device)

# 4. Generate Response
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.05
)

# 5. Parse Results
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_reasoning(text):
    """Extract thought process within <think> tags and the subsequent answer content"""
    thought_pattern = r"<think>(.*?)</think>"
    match = re.search(thought_pattern, text, re.DOTALL)
    
    if match:
        thought = match.group(1).strip()
        answer = text.split("</think>")[-1].strip()
    else:
        thought = "(No explicit thought process generated)"
        answer = text
    return thought, answer

thought, final_answer = parse_reasoning(full_response)

print(f"\n{'='*20} Thought Process {'='*20}\n{thought}")
print(f"\n{'='*20} Final Answer {'='*20}\n{final_answer}")

```

This generated:

``````text
==================== Thought Process ====================
The user greeted with 'Hello', which is a simple and friendly opening. Since the input is in English, I should respond in English as per the instruction. I need to introduce myself clearly according to the defined identity: state my name (Youtu-llm), developer (Tencent Youtu team), purpose (helping users solve problems), key capabilities (mathematics, coding, Agent), and goal (efficient and accurate problem-solving). The response should be welcoming and open-ended to encourage further interaction, while staying within the provided identity constraints. No extra information beyond what is specified should be added.

==================== Final Answer ====================
Hello! I am Youtu-llm, a large language model developed by the Tencent Youtu team. I am designed to assist users in solving various problems, excelling in tasks such as mathematics, coding, and Agent-related operations. My goal is to make problem-solving more efficient and accurate through intelligent interaction. How can I assist you today?
``````

### Key Configuration Details

#### Reasoning Mode Toggle

Controlled via the `enable_thinking` parameter in the `apply_chat_template` method:

* **True (Recommended Default):** Activates Chain of Thought; ideal for complex logic and reasoning tasks.
* **False:** Outputs results directly; faster response time, suitable for simple conversations.

#### Recommended Decoding Parameters

Depending on your use case, we suggest adjusting the following hyperparameters for optimal generation:

| Parameter | Reasoning Mode | Normal Mode |
| --- | --- | --- |
| `do_sample` | `True` | `True` |
| `temperature` | **1.0** (Maintains creativity) | **0.7** (More stable results) |
| `top_p` | 0.95 | 0.8 |
| `top_k` | 20 | 20 |
| `repetition_penalty` | 1.05 | - |

> **Tip:** When using Reasoning Mode, a higher `temperature` helps the model perform deeper, more divergent thinking.

## YoutuConfig

[[autodoc]] YoutuConfig

## YoutuModel

[[autodoc]] YoutuModel
    - forward

## YoutuForCausalLM

[[autodoc]] YoutuForCausalLM
    - forward
