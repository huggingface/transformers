import torch.nn as nn

class Siglip2TextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Placeholder logic
        self.dummy = nn.Identity()

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_attentions=None, output_hidden_states=None):
        return self.dummy(input_ids)
