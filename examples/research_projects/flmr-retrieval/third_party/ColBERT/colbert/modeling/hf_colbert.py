import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from colbert.utils.utils import torch_load_dnn


class HF_ColBERT(BertPreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.dim = colbert_config.dim
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        # if colbert_config.relu:
        #     self.score_scaler = nn.Linear(1, 1)

        self.init_weights()

        # if colbert_config.relu:
        #     self.score_scaler.weight.data.fill_(1.0)
        #     self.score_scaler.bias.data.fill_(-8.0)

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        if name_or_path.endswith('.dnn'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

            obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
            obj.base = base

            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config, ignore_mismatched_sizes=True)
        obj.base = name_or_path

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if name_or_path.endswith('.dnn'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base

            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj

"""
TODO: It's easy to write a class generator that takes "name_or_path" and loads AutoConfig to check the Architecture's
      name, finds that name's *PreTrainedModel and *Model in dir(transformers), and then basically repeats the above.

      It's easy for the BaseColBERT class to instantiate things from there.
"""

