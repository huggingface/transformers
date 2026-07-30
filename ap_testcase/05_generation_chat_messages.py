"""Validate generation from chat messages.

Covers text-only chat, explicit multimodal inputs, automatic loading from content blocks, thinking
activation, and sampling parameters.
"""

import os
import tempfile

import numpy as np
import torch
from _common import bootstrap, finish, run_case, setup_failure
from PIL import Image


def setup():
    """SETUP

    Load the processor and conditional-generation model.
    """
    transformers, checkpoint = bootstrap(("Apertus1p5ForConditionalGeneration", "AutoProcessor"))
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    print("SETUP: loading model (bf16, CPU) ...")
    model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    return processor, model


def generate(processor, model, inputs, max_new_tokens=16):
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def test_image():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))


def test_audio():
    return (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)


def case_1_text_chat(processor, model):
    """CASE 1: TEXT CHAT

    Generate from a text-only chat turn.
    """
    messages = [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    completion = generate(processor, model, inputs, max_new_tokens=8)
    assert "Bern" in completion, f"expected 'Bern', got {completion!r}"
    return f"completion {completion!r}"


def case_2_explicit_media(processor, model):
    """CASE 2: EXPLICIT MEDIA

    Generate from explicitly supplied image and audio.
    """
    messages = [{"role": "user", "content": "<|image|> What do you see, and what do you hear? <|audio|>"}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[test_image()], audio=[test_audio()], return_tensors="pt")
    completion = generate(processor, model, inputs, max_new_tokens=24)
    assert completion.strip(), "generation returned an empty completion"
    return f"completion {completion!r}"


def case_3_automatic_media(processor, model):
    """CASE 3: AUTOMATIC MEDIA

    Load local media from chat content blocks.
    """
    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "test_image.png")
        test_image().save(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        assert "pixel_values" in inputs, "the chat template did not load the image"
        completion = generate(processor, model, inputs, max_new_tokens=24)
    assert completion.strip(), "generation returned an empty completion"
    return f"completion {completion!r}"


def case_4_thinking_toggle(processor, model):
    """CASE 4: THINKING TOGGLE

    Activate deliberation through the chat template.
    """
    messages = [{"role": "user", "content": "Is 17 a prime number? Answer yes or no."}]
    assert "Deliberation: disabled" in processor.apply_chat_template(messages, add_generation_prompt=True), (
        "thinking must stay disabled by default"
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = processor.tokenizer.decode(inputs["input_ids"][0])
    assert "Deliberation: enabled" in prompt, "enable_thinking=True did not switch the developer block"

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    new_ids = output[0, inputs["input_ids"].shape[1] :]
    assert new_ids.numel(), "generation produced no new tokens"
    inner_prefix_id = processor.tokenizer.convert_tokens_to_ids("<|inner_prefix|>")
    assert inner_prefix_id in new_ids.tolist(), "the model did not open a deliberation span"
    # decode with the special tokens kept: they delimit the thinking span
    raw = processor.tokenizer.decode(new_ids)
    return f"deliberation opened; continuation starts {raw[:60]!r}"


def case_5_sampling(processor, model):
    """CASE 5: SAMPLING PARAMETERS

    Reproduce seeded sampling with thinking enabled.
    """
    messages = [{"role": "user", "content": "Name one Swiss lake."}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    sampling = {"do_sample": True, "temperature": 0.8, "top_p": 0.9, "top_k": 50, "max_new_tokens": 24}
    generations = []
    for _ in range(2):
        torch.manual_seed(0)
        with torch.no_grad():
            generations.append(model.generate(**inputs, **sampling))
    assert torch.equal(generations[0], generations[1]), "the same seed must reproduce the sampled continuation"
    raw = processor.tokenizer.decode(generations[0][0, inputs["input_ids"].shape[1] :])
    return f"temperature 0.8, top_p 0.9, top_k 50; continuation starts {raw[:60]!r}"


def main():
    try:
        processor, model = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_text_chat, processor, model),
            run_case(case_2_explicit_media, processor, model),
            run_case(case_3_automatic_media, processor, model),
            run_case(case_4_thinking_toggle, processor, model),
            run_case(case_5_sampling, processor, model),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
