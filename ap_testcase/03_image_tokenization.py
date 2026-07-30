"""Validate processor-driven and manual image tokenization.

Checks vocabulary offsets, grid geometry, and code agreement across the two preprocessing paths.
"""

import numpy as np
import torch
from _common import bootstrap, finish, run_case, setup_failure
from PIL import Image


def setup():
    """SETUP

    Load the processor, model, and synthetic image.
    """
    transformers, checkpoint = bootstrap(("Apertus1p5ForConditionalGeneration", "AutoProcessor"))
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    print("SETUP: loading model (bf16, CPU) ...")
    model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8))
    return processor, model, image


def processor_tokens(processor, model, image):
    inputs = processor(text="<|image|>", images=[image], return_tensors="pt")
    with torch.no_grad():
        vocab_ids = model.model.get_image_tokens(inputs["pixel_values"], inputs["image_sizes"])
    return inputs, vocab_ids


def case_1_processor_path(processor, model, image):
    """CASE 1: PROCESSOR PATH

    Convert processor output into image tokens.
    """
    inputs, vocab_ids = processor_tokens(processor, model, image)
    config = model.config
    placeholders = processor.tokenizer.decode(inputs["input_ids"][0]).count("<|image|>")
    assert vocab_ids.numel() == placeholders, "expected one code per placeholder"
    assert int(vocab_ids.min()) >= config.image_token_offset, "image token below the configured offset"
    assert int(vocab_ids.max()) < config.image_token_offset + config.vision_tokenizer_config.codebook_size, (
        "image token exceeds the configured codebook"
    )
    first = int(vocab_ids[0])
    expected = f"<|visual token {first - config.image_token_offset}|>"
    assert processor.tokenizer.convert_ids_to_tokens(first) == expected, "incorrect vocabulary token mapping"
    return f"{vocab_ids.numel()} codes; first token {expected}"


def case_2_manual_path(processor, model, image):
    """CASE 2: MANUAL PATH

    Compare manual preprocessing with processor tokens.
    """
    inputs, vocab_ids = processor_tokens(processor, model, image)
    target_h, target_w = (int(side) for side in inputs["image_sizes"][0])
    resized = image.convert("RGB").resize((target_w, target_h), Image.BICUBIC)
    pixels = torch.tensor(np.asarray(resized) / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)[None]

    with torch.no_grad():
        code_grid = model.model.vision_tokenizer.encode(pixels)[0]

    expected_shape = (target_h // 16, target_w // 16)
    assert code_grid.shape == expected_shape, f"expected grid {expected_shape}, got {tuple(code_grid.shape)}"
    manual_vocab_ids = code_grid.flatten() + model.config.image_token_offset
    agreement = (manual_vocab_ids == vocab_ids).float().mean().item()
    assert agreement > 0.9, "the PIL and torchvision resize kernels changed too many codes"
    return f"grid {tuple(code_grid.shape)}; code agreement {agreement:.1%}"


def main():
    try:
        processor, model, image = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_processor_path, processor, model, image),
            run_case(case_2_manual_path, processor, model, image),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
