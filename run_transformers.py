from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-350m")

texts= [
	"Hello my name is Thomas and I like", 
	"Hello my name is",
	"He is working on"]
tokenized_texts = tok.batch_encode_plus(texts, padding=True, return_tensors="pt", return_attention_mask=True)
print(model(**tokenized_texts).logits)
print("Tokenized text", tokenized_texts)
print(tok.batch_decode(model.generate(**tokenized_texts, max_length=50, do_sample=False, use_cache=False)))
print(tok.batch_decode(model.generate(**tokenized_texts, max_length=50, do_sample=False, use_cache=True)))
