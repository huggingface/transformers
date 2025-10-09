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
*This model was released on 2021-06-03 and added to Hugging Face Transformers on 2023-06-20 and contributed by [CarlCochet](https://huggingface.co/CarlCochet).*

# Trajectory Transformer

[Trajectory Transformer](https://huggingface.co/papers/2106.02039) explores reinforcement learning (RL) as a sequence modeling problem, utilizing a Transformer architecture to model distributions over trajectories and beam search for planning. This approach simplifies design decisions and is effective across various RL tasks, including long-horizon dynamics prediction, imitation learning, goal-conditioned RL, and offline RL. Combining this method with existing model-free algorithms results in a state-of-the-art planner for sparse-reward, long-horizon tasks.

<hfoptions id="usage">
<hfoption id="TrajectoryTransformerModel">

```py
import torch
from transformers import TrajectoryTransformerModel

model = TrajectoryTransformerModel.from_pretrained("CarlCochet/trajectory-transformer-halfcheetah-medium-v2", dtype="auto")
model.eval()

observations_dim, action_dim, batch_size = 17, 6, 256
seq_length = observations_dim + action_dim + 1

trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(
    device
)
targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

outputs = model(
    trajectories,
    targets=targets,
    use_cache=True,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)
```

</hfoption>
</hfoptions>

## TrajectoryTransformerConfig

[[autodoc]] TrajectoryTransformerConfig

## TrajectoryTransformerModel

[[autodoc]] TrajectoryTransformerModel
    - forward

