import random
import torch
import numpy as np
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from datasets import load_dataset


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


if __name__ == "__main__":

    config = DecisionTransformerConfig()
    config.act_dim = 6
    config.state_dim = 17

    model = DecisionTransformerModel(config)

    dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")

    print(model.config)
    exit()

    sample = dataset["train"][0]

    num_trajectories = len(dataset["train"])
    K = 20
    trajectories = dataset["train"]
    device = "cpu"

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(np.arange(num_trajectories), size=batch_size, replace=True)

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            for k in traj.keys():
                traj[k] = np.array(traj[k])

            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, config.state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, config.act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + max_len).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= config.max_ep_len] = config.max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj["rewards"][si:], gamma=1.0)[: s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:

                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, config.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - 0.0) / 1.0
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, config.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / 100
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(4)

    result = model(
        states,
        actions,
        rewards,
        rtg[:, :-1],
        timesteps,
        attention_mask=attention_mask,
    )

    print(result)
