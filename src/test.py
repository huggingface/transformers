from transformers.models.llama import LlavaLlamaForCausalLM

pipe = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/LLaVA-Lightning-MPT-7B-preview",load_in_4_bits=True)
pipe = pipe.to("cuda")

print(pipe)
