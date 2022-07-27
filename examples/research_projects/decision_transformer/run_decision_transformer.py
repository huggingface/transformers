import numpy as np
import torch

import gym
from mujoco_py import GlfwContext
from transformers import DecisionTransformerModel


GlfwContext(offscreen=True)  # Create a window to init GLFW.


def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # we don't care about the past rewards in this model

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    if model.config.max_length is not None:
        states = states[:, -model.config.max_length :]
        actions = actions[:, -model.config.max_length :]
        returns_to_go = returns_to_go[:, -model.config.max_length :]
        timesteps = timesteps[:, -model.config.max_length :]

        # pad all tokens to sequence length
        attention_mask = torch.cat(
            [torch.zeros(model.config.max_length - states.shape[1]), torch.ones(states.shape[1])]
        )
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat(
            [
                torch.zeros(
                    (states.shape[0], model.config.max_length - states.shape[1], model.config.state_dim),
                    device=states.device,
                ),
                states,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
            [
                torch.zeros(
                    (actions.shape[0], model.config.max_length - actions.shape[1], model.config.act_dim),
                    device=actions.device,
                ),
                actions,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [
                torch.zeros(
                    (returns_to_go.shape[0], model.config.max_length - returns_to_go.shape[1], 1),
                    device=returns_to_go.device,
                ),
                returns_to_go,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.zeros(
                    (timesteps.shape[0], model.config.max_length - timesteps.shape[1]), device=timesteps.device
                ),
                timesteps,
            ],
            dim=1,
        ).to(dtype=torch.long)
    else:
        attention_mask = None

    _, action_preds, _ = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


# build the environment

env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_ep_len = 1000
device = "cuda"
scale = 1000.0  # normalization for rewards/returns
TARGET_RETURN = 3600 / scale  # evaluation conditioning targets, 3600 is reasonable from the paper LINK
state_mean = np.array(
    [
        1.311279,
        -0.08469521,
        -0.5382719,
        -0.07201576,
        0.04932366,
        2.1066856,
        -0.15017354,
        0.00878345,
        -0.2848186,
        -0.18540096,
        -0.28461286,
    ]
)
state_std = np.array(
    [
        0.17790751,
        0.05444621,
        0.21297139,
        0.14530419,
        0.6124444,
        0.85174465,
        1.4515252,
        0.6751696,
        1.536239,
        1.6160746,
        5.6072536,
    ]
)
state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)

# Create the decision transformer model
model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
model = model.to(device)
model.eval()

for ep in range(10):
    episode_return, episode_length = 0, 0
    state = env.reset()
    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    for t in range(max_ep_len):
        env.render()
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = get_action(
            model,
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break
