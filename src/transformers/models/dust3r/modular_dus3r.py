# TODO:
"""
Use modular on ViT or other transformers vision models you think can fit Dust3R
We should replicate the following: https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal/blob/main/dust3r/dust3r.py
We are using multiple "blocks" such as Dust3rEncoder, Dust3rDecoder etc.., I would suggest using inheritence, and inheriting 
from say Vit: something like this:
say I want to replciate the Dust3rEncoder, i would do something like this:
class Dust3rEncoder(ViTEncoder):
    # my custom implementation of dust3r

class Dust3RPreTrainedModel(VitPretrainedModel):
    pass
class Dust3RModel(Dust3RPreTrainedModel):
    def __init__(....):
        self.encoder = Dust3rEncoder(...)
        self.decoder = Dust3rDecoder(...)
        self.head = Dust3rHead(...)
    
    def forward():
        # add forward logic similar to https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal/blob/main/dust3r/dust3r.py#L50

# test your created model first
# random weights is OK for now , let's first make sure a first pass works
"""
