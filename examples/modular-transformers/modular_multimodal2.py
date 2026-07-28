"""
Here, because clip is not consistent with the use of the "Text" and "Vision" prefixes, we cannot simply use
```
class Multimodal2VisionModel(CLIPVisionModel):
    pass
```
with the hope that all dependencies will be renamed as `Multimodal2VisionClass`. For this reason, if we want consistency and
use the "Vision" part everywhere, we need to overwrite the intermediate classes and add the prefix everytime.
This adds noise to the modular, but is unfortunately unavoidable.
"""

from torch import nn

from transformers.models.clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPPreTrainedModel,
    CLIPVisionModel,
)


class Multimodal2VisionAttention(CLIPAttention):
    pass


class Multimodal2VisionMLP(CLIPMLP):
    pass


class Multimodal2VisionEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__()
        self.mlp = Multimodal2VisionMLP(config)
        self.self_attn = Multimodal2VisionAttention(config)


class Multimodal2VisionEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Multimodal2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class Multimodal2VisionPreTrainedModel(CLIPPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": Multimodal2VisionEncoderLayer,
        "attentions": Multimodal2VisionAttention,
    }

    def _init_weights(self, module):
        if isinstance(module, Multimodal2VisionMLP):
            pass


# `CLIPVisionModel` inherits from `CLIPPreTrainedModel`. We need to add the 2nd base here to add the `Vision` part
class Multimodal2VisionModel(CLIPVisionModel, Multimodal2VisionPreTrainedModel):
    _no_split_modules = ["Multimodal2VisionEncoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Multimodal2VisionEncoder(config)
