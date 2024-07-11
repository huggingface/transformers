from transformers import GLMForCausalLM, GLMConfig, GLMModel, GLMTokenizer

model = GLMModel(GLMConfig())
tokenizer = GLMTokenizer.from_pretrained("THUDM/glm-4-9b-chat")
print(model)
breakpoint()
