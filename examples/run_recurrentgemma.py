"""Temporary example to test whether model generation works correctly."""

from transformers import AutoTokenizer
from transformers import RecurrentGemmaForCausalLM
from transformers.models import recurrentgemma

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = RecurrentGemmaForCausalLM(
    config=recurrentgemma.RecurrentGemmaConfig(
        # ....
    )
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
