<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-02 and added to Hugging Face Transformers on 2022-03-23 and contributed by [edbeeching](https://huggingface.co/edbeeching).*

# Decision Transformer

[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://huggingface.co/papers/2106.01345) reframes reinforcement learning as a conditional sequence modeling problem, using a causally masked Transformer architecture instead of traditional value functions or policy gradients. It generates actions by autoregressively conditioning on past states, actions, and a desired return, allowing the model to produce future actions that achieve specified rewards. This approach leverages advances from language modeling, such as GPT and BERT, for scalability and simplicity. Despite its straightforward design, Decision Transformer matches or surpasses state-of-the-art model-free offline RL performance on benchmarks like Atari, OpenAI Gym, and Key-to-Door tasks.

<hfoptions id="usage">
<hfoption id="DecisionTransformerModel">

```py
import torch
from transformers import DecisionTransformerModel

model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium", dtype="auto")
model.eval()

env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

state = env.reset()
states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

with torch.no_grad():
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=target_return,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
```

</hfoption>
</hfoptions>

## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig

## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model
    - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel
    - forward

