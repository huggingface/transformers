from transformers import H3Config, H3ForCausalLM 

model = H3ForCausalLM(H3Config())

for name, param in model.named_parameters():
    print(name, param.shape)