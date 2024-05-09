from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig, GenerationConfig

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

generation_config = GenerationConfig.from_pretrained("/home/raushan/gen_config.json")

# first generate text with watermark and without
watermark_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
model.generation_config.pad_token_id = 50256

out = model.generate(**inputs, do_sample=False, max_length=20)
out_watermarked = model.generate(**inputs, watermarking_config={"bias": 2.5}, do_sample=False, max_length=20)

# now we can instantiate the detector and check the generated text
detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermark_config)
detection_out_watermarked = detector(out_watermarked, return_dict=True)
detection_out = detector(out, return_dict=True)
print(detection_out_watermarked.z_score)
print(detection_out.z_score)
