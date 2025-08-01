<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSeek-V3

## Overview

The DeepSeek-V3 model was proposed in [DeepSeek-V3 Technical Report](https://huggingface.co/papers/2412.19437) by DeepSeek-AI Team.

The abstract from the paper is the following:
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.

## Limitations and call for contribution!

We are super happy to make this code community-powered, and would love to see how you can best optimize the following: 

- current implementation uses the "naive" attention compution (so not really MLA)
- current implementation loops through the experts. This should be replaced. Pointers to use `get_packed_weights` from `integrations/tensor_parallel`. 
- current implementation uses the eleuther formula for ROPE, using the original one would be more efficient! (should still follow our API)
- static cache is not supported (this should be just a generation config issue / config shape issues)

### Usage tips
The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

You can run the model in `FP8` automatically, using 2 nodes of 8 H100 should be more than enough! 

```python
# `run_deepseek_v1.py`
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

tokenizer = AutoTokenizer.from_pretrained("deepseek-r1")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


model = AutoModelForCausalLM.from_pretrained("deepseek-r1", device_map="auto", dtype=torch.bfloat16)
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs))
print(time.time()-start)
```
This generated: 

``````
<｜Assistant｜><think>
Okay, the user wants to demonstrate how chat templating works. Let me break down what that means. Chat templating is about structuring the conversation data, especially for models that need specific input formats. Maybe they're referring to something like how messages are formatted with roles (user, assistant, system) in APIs like OpenAI.

First, I should explain what chat templating is. It's the process of formatting conversation data into a structured format that the model can understand. This usually includes roles and content. For example, user messages, assistant responses, and system messages each have their own role tags.

They might want an example. Let me think of a simple conversation. The user says "Hello, how are you?" and the assistant responds "I'm doing great. How can I help you today?" Then the user follows up with wanting to show off chat templating. So the example should include the history and the new message.

In some frameworks, like Hugging Face's Transformers, chat templates are applied using Jinja2 templates. The template might look something like combining system messages, then looping through user and assistant messages with appropriate tags. For instance, using {% for message in messages %} and assigning roles like <|user|>, <|assistant|>, etc.

I should structure the example with the messages array, showing each role and content. Then apply a hypothetical template to convert that into a formatted string the model uses. Also, mention that different models have different templating requirements, like using special tokens or varying role labels.

Wait, the user mentioned "chat templating" in the context of showing off. Maybe they want a practical example they can present. So providing a code snippet or a structured data example would be helpful. Let me outline a typical messages array and then the templated output.

Also, it's important to note that proper templating ensures the model knows the conversation flow, which is crucial for generating coherent responses. Maybe include a note about why it's important, like maintaining context and role-specific processing.

Let me check if there are any common mistakes or things to avoid. For example, not closing tags properly, or mismatching roles. But maybe that's too detailed unless the user asks. Focus on the positive example first.

Putting it all together, the response should have an example messages array, the applied template, and the final formatted string. Maybe use angle brackets or special tokens as placeholders. Also, mention that this helps in training or fine-tuning models with structured data.

I think that's a solid approach. Let me structure it step by step to make it clear.
</think>

Chat templating is a way to structure conversation data (e.g., user/assistant interactions) into a format that language models understand. This is especially important for models trained to handle multi-turn dialogues, where the input must explicitly separate roles (user, assistant, system, etc.) and messages. Let’s break this down with an example!

---

### **Step 1: Raw Conversation History**
Suppose we have this conversation:
- **User**: "Hello, how are you?"
- **Assistant**: "I'm doing great. How can I help you today?"
- **User**: "I'd like to show off how chat templating works!"

---

### **Step 2: Structured Messages**
In frameworks like Hugging Face Transformers or OpenAI, conversations are often formatted as a list of dictionaries with `role` and `content`:
```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

---

### **Step 3: Apply a Chat Template**
A **chat template** converts this structured data into a single string formatted for the model. For example, using a Jinja-style template (common in Hugging Face):

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        <|user|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'assistant' %}
        <|assistant|>{{ message['content'] }}<|end|>
    {% endif %}
{% endfor %}
<|assistant|>
```

---

### **Step 4: Final Templated Output**
Applying the template to our `messages` list would produce:
```text
<|user|>Hello, how are you?<|end|>
<|assistant|>I'm doing great. How can I help you today?<|end|>
<|user|>I'd like to show off how chat templating works!<|end|>
<|assistant|>
```

This tells the model:  
1. The conversation history (user/assistant turns).  
2. The model’s turn to generate a response (`<|assistant|>` at the end).  

---

### **Key Notes**:
- **Role Separation**: Tags like `<|user|>` and `<|assistant|>` help the model distinguish speakers.
- **Special Tokens**: Models often use unique tokens (e.g., `<|end|>`) to mark message boundaries.
- **Flexibility**: Templates vary by model (e.g., OpenAI uses `{"role": "user", "content": "..."}` instead of tags).

---

### **Why This Matters**:
- **Consistency**: Ensures the model understands dialogue structure.
- **Context Preservation**: Maintains the flow of multi-turn conversations.
- **Alignment**: Matches the format the model was trained on for better performance.

Want to dive deeper or see a specific framework’s implementation (e.g., OpenAI, Llama, Mistral)? Let me know! 😊<｜end▁of▁sentence｜>
``````

Use the following to run it
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0|1 --rdzv-id an_id --rdzv-backend c10d --rdzv-endpoint master_addr:master_port run_deepseek_r1.py
```

If you have: 
```bash
[rank0]: ncclInternalError: Internal check failed.
[rank0]: Last error:
[rank0]: Bootstrap : no socket interface found
```
error, it means NCCL was probably not loaded. 


## DeepseekV3Config

[[autodoc]] DeepseekV3Config

## DeepseekV3Model

[[autodoc]] DeepseekV3Model
    - forward

## DeepseekV3ForCausalLM

[[autodoc]] DeepseekV3ForCausalLM
    - forward
