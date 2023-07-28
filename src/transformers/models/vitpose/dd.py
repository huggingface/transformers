import torch
import timm
from huggingface_hub import hf_hub_download

def modify_variable_names(checkpoint_path, new_variable_names):
    # Step 1: Load the .pth file
    checkpoint_path = hf_hub_download(repo_id="shauray/VitPose", filename="vitpose-b.pth", repo_type="model")
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model = timm.create_model(checkpoint_path, preetrained=True)
    print(model)

    # Step 2: Iterate and modify variable names
    for old_name, new_name in new_variable_names.items():
        if old_name in checkpoint:
            checkpoint[new_name] = checkpoint.pop(old_name)
            print(checkpoint[new_name])

    # Step 3: Save the modified dictionary to a new .pth file
    new_checkpoint_path = "modified_checkpoint.pth"
    torch.save(checkpoint, new_checkpoint_path)

# Example usage:
# Suppose you want to rename 'model.weight' to 'model.param' and 'model.bias' to 'model.bias_param'.
# You can provide the mapping as a dictionary as shown below:
new_variable_names = {'model.weight': 'model.param', 'model.bias': 'model.bias_param'}

# Modify the variable names in the .pth file

modify_variable_names("vitpose_small.pth", new_variable_names)
