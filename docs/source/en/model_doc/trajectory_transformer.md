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

# Trajectory Transformer

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The Trajectory Transformer model was proposed in [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)  by Michael Janner, Qiyang Li, Sergey Levine.

The abstract from the paper is the following:

*Reinforcement learning (RL) is typically concerned with estimating stationary policies or single-step models,
leveraging the Markov property to factorize problems in time. However, we can also view RL as a generic sequence
modeling problem, with the goal being to produce a sequence of actions that leads to a sequence of high rewards.
Viewed in this way, it is tempting to consider whether high-capacity sequence prediction models that work well
in other domains, such as natural-language processing, can also provide effective solutions to the RL problem.
To this end, we explore how RL can be tackled with the tools of sequence modeling, using a Transformer architecture
to model distributions over trajectories and repurposing beam search as a planning algorithm. Framing RL as sequence
modeling problem simplifies a range of design decisions, allowing us to dispense with many of the components common
in offline RL algorithms. We demonstrate the flexibility of this approach across long-horizon dynamics prediction,
imitation learning, goal-conditioned RL, and offline RL. Further, we show that this approach can be combined with
existing model-free algorithms to yield a state-of-the-art planner in sparse-reward, long-horizon tasks.*

This model was contributed by [CarlCochet](https://huggingface.co/CarlCochet). The original code can be found [here](https://github.com/jannerm/trajectory-transformer).

## Usage tips

This Transformer is used for deep reinforcement learning. To use it, you need to create sequences from
actions, states and rewards from all previous timesteps. This model will treat all these elements together
as one big sequence (a trajectory).

## TrajectoryTransformerConfig

[[autodoc]] TrajectoryTransformerConfig

## TrajectoryTransformerModel

[[autodoc]] TrajectoryTransformerModel
    - forward
