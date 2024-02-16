import os
import torch

from colbert.utils.utils import torch_load_dnn

from transformers import AutoTokenizer
from colbert.modeling.hf_colbert import HF_ColBERT
from colbert.infra.config import ColBERTConfig


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name, colbert_config=None):
        super().__init__()

        self.name = name
        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name), colbert_config)
        self.model = HF_ColBERT.from_pretrained(name, colbert_config=self.colbert_config)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.bert
    
    @property
    def linear(self):
        return self.model.linear
    
    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)
        if getattr(self, "vision_projection", None) is not None:
            # save using pytorch save
            torch.save(self.vision_projection.state_dict(), os.path.join(path, "vision_projection.pt"))
            print(f"Saved vision projection to {os.path.join(path, 'vision_projection.pt')}")
        if getattr(self, "doc_vision_projection", None) is not None:
            # save using pytorch save
            # print("saving doc vision projection", self.doc_vision_projection.state_dict())
            torch.save(self.doc_vision_projection.state_dict(), os.path.join(path, "doc_vision_projection.pt"))
            print(f"Saved doc vision projection to {os.path.join(path, 'doc_vision_projection.pt')}")
        
        self.colbert_config.save_for_checkpoint(path)


if __name__ == '__main__':
    import random
    import numpy as np

    from colbert.infra.run import Run
    from colbert.infra.config import RunConfig

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    with Run().context(RunConfig(gpus=2)):
        m = BaseColBERT('bert-base-uncased', colbert_config=ColBERTConfig(Run().config, doc_maxlen=300, similarity='l2'))
        m.colbert_config.help()
        print(m.linear.weight)
        m.save('/future/u/okhattab/tmp/2021/08/model.deleteme2/')

    m2 = BaseColBERT('/future/u/okhattab/tmp/2021/08/model.deleteme2/')
    m2.colbert_config.help()
    print(m2.linear.weight)

    exit()

    m = BaseColBERT('/future/u/okhattab/tmp/2021/08/model.deleteme/')
    print('BaseColBERT', m.linear.weight)
    print('BaseColBERT', m.colbert_config)

    exit()

    # m = HF_ColBERT.from_pretrained('nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large')
    m = HF_ColBERT.from_pretrained('/future/u/okhattab/tmp/2021/08/model.deleteme/')
    print('HF_ColBERT', m.linear.weight)

    m.save_pretrained('/future/u/okhattab/tmp/2021/08/model.deleteme/')

    # old = OldColBERT.from_pretrained('bert-base-uncased')
    # print(old.bert.encoder.layer[10].attention.self.value.weight)

    # random.seed(12345)
    # np.random.seed(12345)
    # torch.manual_seed(12345)

    dnn = torch_load_dnn(
        "/future/u/okhattab/root/TACL21/experiments/Feb26.NQ/train.py/ColBERT.C3/checkpoints/colbert-60000.dnn")
    # base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

    # new = BaseColBERT.from_pretrained('bert-base-uncased', state_dict=dnn['model_state_dict'])

    # print(new.bert.encoder.layer[10].attention.self.value.weight)

    print(dnn['model_state_dict']['linear.weight'])
    # print(dnn['model_state_dict']['bert.encoder.layer.10.attention.self.value.weight'])

    # # base_model_prefix
