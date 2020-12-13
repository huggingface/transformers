from transformers import file_utils
models = file_utils.get_cached_models()
total_size = 0
for model in models:
    total_size+= model[2]
    print(model)
print(total_size)