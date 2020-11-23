import torch
from .modeling_electra import ElectraModel
from .configuration_electra import ElectraConfig

if __name__ == "__main__":
    conf = ElectraConfig.from_json_file("/home/abhishek/huggingface/models/convbert/config.json")
    ei = ElectraModel(conf)
    x = torch.randint(0, 100, (4, 64))
    print(ei(input_ids=x))