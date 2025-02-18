from transformers.models.llama import LlamaModel, LlamaConfig
import torch

def run_llama():
    #model initialize
    llamaconfig = LlamaConfig(
        vocab_size= 32000, hidden_size=4096//2,
        intermediate_size=11008//2,
        num_hidden_layers=32//2,
        num_attention_heads=32//2,
        max_prosition_embeddings=2048//2
    )
    
    llamamodel = LlamaModel(config=llamaconfig)
    #input parameter
    input_ids = torch.randint(
        low=0, high=llamaconfig.vocab_size, size=(4,30)
    )
    #run model
    res = llamamodel(input_ids)# llamamodel.forward(input_ids)
    print(res)

if __name__ =='__main__':
    run_llama()