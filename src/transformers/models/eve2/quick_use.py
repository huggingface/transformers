from transformers.models.eve2.modeling_eve2 import Eve2ForCausalLM


model_path = "/home/nas/buffer/steven/EVE/EVEv2"
model = Eve2ForCausalLM.from_pretrained(model_path)
model.to("cuda")

print(model)
