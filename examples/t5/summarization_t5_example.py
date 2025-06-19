# examples/t5/summarization_t5_example.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Sample input text
text = """The Hugging Face Transformers library provides thousands of pretrained models to perform tasks on texts 
such as classification, information extraction, question answering, summarization, translation, and text generation."""

# Add prefix required by T5
input_text = "summarize: " + text
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(input_ids, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original:\n", text)
print("\nSummary:\n", summary)
