from transformers import GITConfig, GITVisionConfig, GITModel

# Initializing a GIT microsoft/git-base style configuration
configuration = GITConfig(vision_config=GITVisionConfig())

# Initializing a model (with random weights) from the microsoft/git-base style configuration
model = GITModel(configuration)