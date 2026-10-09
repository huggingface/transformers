"""Validate generation from raw text.

Covers text continuation, image and audio placeholders, and batched versus per-sample parity.
"""

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


def decode_new(processor, output, inputs):
    return [
        processor.tokenizer.decode(sequence[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for sequence in output
    ]


def test_image():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))


def test_audio():
    return (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)


def case_1_text_continuation(processor, model):
    """CASE 1: TEXT CONTINUATION

    Continue a raw text prompt.
    """
    inputs = processor(text="The capital of France is", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    completion = decode_new(processor, output, inputs)[0]
    assert "Paris" in completion, f"expected 'Paris', got {completion!r}"
    return f"completion {completion!r}"


def case_2_image_prompt(processor, model):
    """CASE 2: IMAGE PROMPT

    Generate from raw text with an image placeholder.
    """
    inputs = processor(text="<|image|> The picture shows", images=[test_image()], return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    assert output.shape[1] > inputs["input_ids"].shape[1], "generation produced no new tokens"
    completion = decode_new(processor, output, inputs)[0]
    return f"completion {completion!r}"


def case_3_audio_prompt(processor, model):
    """CASE 3: AUDIO PROMPT

    Generate from raw text with an audio placeholder.
    """
    inputs = processor(text="<|audio|> The recording contains", audio=[test_audio()], return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    assert output.shape[1] > inputs["input_ids"].shape[1], "generation produced no new tokens"
    completion = decode_new(processor, output, inputs)[0]
    return f"completion {completion!r}"


def case_4_batch_parity(processor, model):
    """CASE 4: BATCH PARITY

    Match greedy and beam generations with uneven image/audio groups to per-sample runs.
    """
    image = test_image().resize((64, 64))
    audio = test_audio()[:2400]
    texts = [
        "The capital of France is",
        "<|image|><|image|><|audio|><|audio|> Describe these inputs.",
        "<|image|> The image shows",
        "<|audio|> The audio contains",
    ]
    images = [[], [image, test_image().resize((64, 80))], [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)], []]
    audios = [[], [audio, audio[:1800]], [], [audio]]
    processor_kwargs = {"images_kwargs": {"do_resize": False}, "return_tensors": "pt"}
    batched = processor(
        text=texts,
        images=images,
        audio=audios,
        padding=True,
        **processor_kwargs,
    )
    for num_beams in (1, 2):
        options = {"max_new_tokens": 2, "do_sample": False, "num_beams": num_beams}
        with torch.no_grad():
            output = model.generate(**batched, **options)
        batched_tokens = output[:, batched["input_ids"].shape[1] :].tolist()

        single_tokens = []
        for text, sample_images, sample_audio in zip(texts, images, audios):
            inputs = processor(text=text, images=sample_images or None, audio=sample_audio or None, **processor_kwargs)
            with torch.no_grad():
                output = model.generate(**inputs, **options)
            single_tokens.append(output[0, inputs["input_ids"].shape[1] :].tolist())
        assert batched_tokens == single_tokens, (
            f"num_beams={num_beams}: batched {batched_tokens} != per-sample {single_tokens}"
        )
    return "greedy and beam-2 match for text-only, image-only, audio-only and multiple-media prompts"


def main():
    try:
        processor, model = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_text_continuation, processor, model),
            run_case(case_2_image_prompt, processor, model),
            run_case(case_3_audio_prompt, processor, model),
            run_case(case_4_batch_parity, processor, model),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
