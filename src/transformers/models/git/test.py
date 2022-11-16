from transformers import GITConfig, GITForCausalLM, GITVisionConfig


# Initializing a GIT microsoft/git-base style configuration
configuration = GITConfig(vision_config=GITVisionConfig())

# Initializing a model (with random weights) from the microsoft/git-base style configuration
model = GITForCausalLM(configuration)

for name, param in model.named_parameters():
    print(name, param.shape)
