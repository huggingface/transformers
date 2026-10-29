<!--Copyright 2026 Reflection AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
        <img alt="Expert parallelism" src="https://img.shields.io/badge/Expert%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Beam

## Overview

[Beam](https://reflection.ai/blog/introducing-beam) is Reflection's first open-weight model, a sparse mixture-of-experts (MoE) language model with 501B total parameters and 23B active parameters per token, built for coding, reasoning, and agentic workloads. It was pretrained on 23.8T tokens and post-trained with large-scale asynchronous reinforcement learning. Beam is text-only, and it reasons before every answer. A reasoning effort setting in the chat template controls how much.

Beam has 52 decoder layers and combines the following features:

- **Hybrid sliding and full attention.** Every sixth layer and the last layer use full attention, and the others attend to the most recent `sliding_window` (512) tokens. `layer_types` sets the pattern.
- **RoPE on sliding layers, NoPE on full layers.** `rope_parameters` is keyed by layer type, and `None` marks a layer type without positional encoding. The full-attention layers scale queries instead (scalable softmax): the query at 1-based position `n` is multiplied by `(1 + n / 2048) ** 0.1`, which keeps attention from flattening as the context grows.
- **Gated attention.** Grouped-query attention with 48 query heads and 8 key/value heads, and RMSNorm on each query and key head. A sigmoid gate computed from the attention input, with a learned bias per head, scales the attention output elementwise before the output projection.
- **Sparse MoE.** The first 3 layers use a dense SwiGLU MLP, and the remaining 49 layers route each token to 8 of 256 experts, with no shared expert (`mlp_layer_types` sets the pattern). The router scores experts with a sigmoid, balances load with a learned per-expert bias instead of an auxiliary loss, and restricts each token to the best `topk_group` of `n_group` expert groups. The routing weights are normalized, and the MoE output is scaled by `routed_scaling_factor`.
- **Sandwich norms with depth-scaled residuals.** Each attention and MLP block is wrapped in an RMSNorm before and after. Residual additions are rescaled so that, after `k` contributions (counting the embeddings as the first), the residual stream is their sum divided by `sqrt(k)`.
- **Final logit softcapping.** Logits are capped with `30 * tanh(logits / 30)` (`final_logit_softcapping`).

The implementation computes RMSNorm scaling, the SwiGLU activation, RoPE, query scaling, routing, and softcapping in FP32 to match the numerics Beam was trained and served with. The router weights and the expert bias are kept in FP32 at load time, and the returned `logits` are always FP32.

You can find all the original Beam checkpoints under the [Reflection](https://huggingface.co/reflection) organization.

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).

## Usage examples

Beam's chat template formats conversations in Beam's Harmony format, and its [response template](../chat_response_parsing) parses the generated text back into a message with `thinking`, `content`, and `tool_calls` fields. The examples below generate a reply with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

The BF16 checkpoint is about 1 TB, so the examples shard it across 8 devices, placing whole routed experts on each device with [expert parallelism](../expert_parallelism). Launch them with `torchrun --nproc-per-node 8 <script.py>`.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import DistributedConfig, pipeline

pipe = pipeline(
    task="text-generation",
    model="reflection/Beam-1.0-501B-A23B",
    model_kwargs={"distributed_config": DistributedConfig(tp_size=8, ep_size=8)},
)

messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
output = pipe(messages, max_new_tokens=2048)
reply = output[0]["generated_text"][-1]
print(reply["thinking"])
print(reply["content"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

model_id = "reflection/Beam-1.0-501B-A23B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, distributed_config=DistributedConfig(tp_size=8, ep_size=8))

messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, reasoning_effort="low"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048)
generated = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :])
message = tokenizer.parse_response(generated, prefix=inputs["input_ids"][0])
print(message["thinking"])
print(message["content"])
```

</hfoption>
</hfoptions>

Prefer this over `device_map="auto"`, which places whole layers on each device, runs them one device at a time, and can run out of memory while loading.

Generation samples with Beam's recommended `temperature=0.7` and `top_p=0.99`, which the checkpoint's [`GenerationConfig`] sets as defaults. Pass `do_sample=False` to decode greedily.

### Reasoning effort

Pass `reasoning_effort` to [`~PreTrainedTokenizerBase.apply_chat_template`] to control how much Beam reasons. It accepts an integer budget or one of the labels below. Beam was not trained with efforts below 8, so smaller values aren't respected.

| Label | `minimal` | `low` | `medium` | `high` | `xhigh` | `max` |
|---|---|---|---|---|---|---|
| Effort | 8 | 32 | 64 | 128 | 256 | 512 |

When `reasoning_effort` is unset, the template uses `max`. If `max_new_tokens` runs out during reasoning, the parsed message has `thinking` but no `content`.

### Multi-turn conversations

Append the parsed message to the history as is. Beam's output degrades without its earlier reasoning, which the chat template reads from `thinking` (`reasoning_content` and `reasoning` are also accepted). By default, an assistant turn without reasoning renders with an empty reasoning block. Pass `allow_missing_thinking=False` to raise an error instead when `add_generation_prompt=True`, which is recommended when serving Beam.

```python
messages.append(message)
messages.append({"role": "user", "content": "And why are sunsets red?"})
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
```

### Tool calling

Pass `tools` to both [`~PreTrainedTokenizerBase.apply_chat_template`] and [`~PreTrainedTokenizerBase.parse_response`]. With `tools`, parsed arguments are cast to the types in each tool's schema; without it, every argument is a string.

```python
def get_weather(city: str):
    """
    Get the current weather in a city.

    Args:
        city: The city name.
    """
    return "Sunny, 22 C"


messages = [{"role": "user", "content": "What's the weather like in Paris right now?"}]
inputs = tokenizer.apply_chat_template(
    messages, tools=[get_weather], add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
generated = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :])
message = tokenizer.parse_response(generated, prefix=inputs["input_ids"][0], tools=[get_weather])
print(message["tool_calls"])
# [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}}]

messages.append(message)
for call in message["tool_calls"]:
    result = get_weather(**call["function"]["arguments"])
    messages.append({"role": "tool", "name": call["function"]["name"], "content": result})
```

Then generate again from `messages` to get the final answer. Parsed tool calls have no `id`, and the format matches results to calls by position, so return tool results in the order of the calls. Generation stops after a final answer (`<|return|>`) or a tool call (`<|call|>`); `generation_config.json` lists both.

For streaming, [`~PreTrainedTokenizerBase.get_response_parser`] returns a parser that you `feed()` text chunks to as they are generated. See [Response parsing](../chat_response_parsing) for more details.

## Notes

- **Template arguments.** Besides `reasoning_effort`, the chat template accepts `model_identity` (replaces the first line of the system message), `builtin_tools`, and `allow_missing_thinking`. `builtin_tools` should only name `search` and `browse` tools, and only with the definitions Beam was trained with. Other tools only need `tools`. `system` and `developer` messages are only rendered at the start of the conversation.
- **Training.** The chat template marks the assistant output for `apply_chat_template(..., return_dict=True, return_assistant_tokens_mask=True)`, so TRL's `SFTConfig(assistant_only_loss=True)` trains on assistant tokens only. The mask covers each assistant turn's reasoning, answer, and tool calls.
- **No auxiliary loss.** The router balances expert load through its bias, so `BeamForCausalLM` computes no load-balancing loss. `output_router_logits=True` still returns the router logits.

## BeamConfig

[[autodoc]] BeamConfig

## BeamModel

[[autodoc]] BeamModel
    - forward

## BeamForCausalLM

[[autodoc]] BeamForCausalLM
    - forward
