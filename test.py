from src.transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, VptqConfig



import onnxruntime as ort

# model_path = "../anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_SELU/model.onnx"

# providers = [
#     (
#         "CUDAExecutionProvider",
#         {
#             "device_id": 0,
#             "arena_extend_strategy": "kNextPowerOfTwo",
#             "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
#             "cudnn_conv_algo_search": "EXHAUSTIVE",
#             "do_copy_in_default_stream": True,
#         },
#     ),
#     "CPUExecutionProvider",
# ]

# session = ort.InferenceSession(model_path, providers=providers)

model = AutoModelForCausalLM.from_pretrained("../Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft", device_map="cuda")
input_text = "Hello my name is"
gen_config = GenerationConfig(max_new_tokens=32)
gen_config.do_sample = False
tokenizer = AutoTokenizer.from_pretrained("../Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft")
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
out = model.generate(**input_ids, max_new_tokens=32, do_sample=False)
out = tokenizer.decode(out[0], skip_special_tokens=True)
print(out)


quantization_config = VptqConfig()
_ = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", quantization_config=quantization_config)
model = AutoModelForCausalLM.from_pretrained("BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf")

print(model)