"""
This is a script to run 6eoo inference using the transfomers lib. 

You can use to vibe check model outputs. 

I see outputs like:
- "The image depicts a vibrant street scene at the entrance to a Chinatown, likely in an urban area..."

(which is correct given: https://www.ilankelman.org/stopsigns/australia.jpg)

I've tested this script with both:
- The raw tif exported checkpoint `gsutil -m cp -r gs://cohere-command/experimental_models/c3-sweep-6eoog65n-e0ry-fp16/tif_export`
- The checkpoint I uploaded to a private HF repo `julianmack/command-vision-test01`

Change the model_id below to switch between them.
"""
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 6eoo inference script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="CohereLabs/c4ai-command-a-03-2025",
        help="Path to the model or Hugging Face model ID",
    )
    args = parser.parse_args()
    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
    )

    inputs = inputs.to(model.device)

    print("running generation...")
    gen_tokens = model.generate(
        inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.3,
        # use_cache=False,
    )

    print(
        tokenizer.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    )
