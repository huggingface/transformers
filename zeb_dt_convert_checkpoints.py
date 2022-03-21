import torch
import gym
from transformers import DecisionTransformerModel, DecisionTransformerConfig

# original_checkpoint = torch.load("dt_hopper_old.pth")
# new_checkpoint = torch.load("dt_hopper.pth")

# print(original_checkpoint.keys())
# print("#" * 80)
# print(new_checkpoint.keys())

# updated_checkpoint = {}

# for k in original_checkpoint.keys():
#     sub = k.replace("transformer.", "encoder.")
#     updated_checkpoint[sub] = original_checkpoint[k]
#     if sub in new_checkpoint.keys():
#         print(
#             original_checkpoint[k].shape == new_checkpoint[sub].shape,
#             sub,
#             original_checkpoint[k].shape,
#             new_checkpoint[sub].shape,
#         )
#     else:
#         print("not avail", k, sub)

# print("#" * 80)
# for k in new_checkpoint.keys():
#     sub = k.replace("encoder.", "transformer.")

#     if sub in original_checkpoint.keys():
#         print(
#             new_checkpoint[k].shape == original_checkpoint[sub].shape,
#             sub,
#             new_checkpoint[k].shape,
#             original_checkpoint[sub].shape,
#         )
#     else:
#         print("not avail", k, sub)
# updated_checkpoint["encoder.wpe.weight"] = new_checkpoint["encoder.wpe.weight"]
# torch.save(updated_checkpoint, "dt_hopper_copied.pth")

env_names = ["halfcheetah", "hopper", "walker2d"]
env_ids = ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3"]
difficulties = ["expert", "medium", "medium-replay"]


def update_state_dict(original_state_dict, new_state_dict):
    updated_state_dict = {}
    for k in original_state_dict.keys():
        sub = k.replace("transformer.", "encoder.")
        updated_state_dict[sub] = original_state_dict[k]

    updated_state_dict["encoder.wpe.weight"] = new_state_dict["encoder.wpe.weight"]

    return updated_state_dict


for env_name, env_id in zip(env_names, env_ids):

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    config = DecisionTransformerConfig(
        state_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=1000,
        max_length=20,
    )
    model = DecisionTransformerModel(config)
    new_state_dict = model.state_dict()

    for difficulty in difficulties:
        input_file_name = f"checkpoints/{env_name}/{difficulty}/9.pth"
        output_file_name = f"checkpoints/{env_name}_{difficulty}.pth"

        original_state_dict = torch.load(input_file_name)
        updated_state_dict = update_state_dict(original_state_dict, new_state_dict)

        model2 = DecisionTransformerModel(config)
        model2.load_state_dict(updated_state_dict)
        model2.push_to_hub(f"decision-transformer-gym-{env_name}-{difficulty}")

        torch.save(updated_state_dict, output_file_name)
