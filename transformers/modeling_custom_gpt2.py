from pathlib import Path
from lm.inference import ModelWrapper
from lm.model import OutputGetters

from transformers.configuration_gpt2 import GPT2Config

class CustomGPT2:
    config = GPT2Config()

    def __init__(self, model: ModelWrapper):
        self.model = model

    @classmethod
    def from_pretrained(cls, path: Path):
        model = ModelWrapper.load_encoder(f'{path}/model.pt', True, False, 'cpu', output_getter=OutputGetters.raw)
        return cls(model)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, input_ids, **kwargs):
        output = self.model(input_ids)['logits']
        return output,
