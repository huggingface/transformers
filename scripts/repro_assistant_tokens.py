from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    assistant = AutoModelForCausalLM.from_pretrained("gpt2")

    inputs = tok("hello", return_tensors="pt")
    _ = model.generate(**inputs, assistant_model=assistant, num_assistant_tokens=5)

    print("assistant num_assistant_tokens (actual):",
          assistant.generation_config.num_assistant_tokens)

if __name__ == "__main__":
    main()
