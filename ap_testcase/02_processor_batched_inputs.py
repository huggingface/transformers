"""Validate batched processor inputs.

Covers nested and flat media ownership, empty media collections, and strict placeholder-count errors.
"""

import numpy as np
from _common import bootstrap, finish, run_case, setup_failure


def setup():
    """SETUP

    Load the processor without model weights.
    """
    transformers, checkpoint = bootstrap(("AutoProcessor",), processor_only=True)
    return transformers.AutoProcessor.from_pretrained(checkpoint)


rng = np.random.default_rng(0)


def image(height, width):
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def clip(seconds):
    return rng.standard_normal(int(24000 * seconds), dtype=np.float32)


def texts():
    return [
        "Transcribe this recording: <|audio|>",
        "<|image|> Describe the image.",
        "Compare <|image|> and <|image|> while listening to <|audio|> and <|audio|>.",
    ]


def case_1_nested_media(processor):
    """CASE 1: NESTED MEDIA

    Process explicit per-sample media lists.
    """
    images = [[], [image(256, 256)], [image(256, 256), image(320, 256)]]
    audio = [[clip(1.0)], [], [clip(0.5), clip(2.0)]]
    out = processor(text=texts(), images=images, audio=audio, padding=True, return_tensors="pt")

    assert out["pixel_values"].shape[0] == 3, "three images total, flattened over the batch"
    assert out["input_features"].shape[0] == 3, "three clips total, flattened over the batch"
    assert out["input_ids"].shape[0] == 3 and out["attention_mask"].shape[0] == 3, "expected three samples"
    sample0 = processor.tokenizer.decode(out["input_ids"][0], skip_special_tokens=False)
    assert "<|img_start|>" not in sample0 and "<|audio_start|>" in sample0, "sample 0 has audio but no image"
    counts = [processor.tokenizer.decode(ids).count("<|audio|>") for ids in out["input_ids"]]
    assert counts == [40, 0, 20 + 80], f"per-sample audio placeholders follow clip lengths, got {counts}"
    return f"3 images and 3 clips; audio placeholders {counts}"


def case_2_flat_media(processor):
    """CASE 2: FLAT MEDIA

    Consume flat media lists by placeholder order. Should be equal to nested explicit lists.
    """
    nested = processor(
        text=texts(),
        images=[[], [image(256, 256)], [image(256, 256), image(320, 256)]],
        audio=[[clip(1.0)], [], [clip(0.5), clip(2.0)]],
        padding=True,
        return_tensors="pt",
    )
    flat = processor(
        text=texts(),
        images=[image(256, 256), image(256, 256), image(320, 256)],
        audio=[clip(1.0), clip(0.5), clip(2.0)],
        padding=True,
        return_tensors="pt",
    )
    assert flat["image_sizes"].tolist() == nested["image_sizes"].tolist(), (
        "flat and nested forms must assign the same images"
    )
    return f"image sizes {flat['image_sizes'].tolist()}"


def case_3_empty_media(processor):
    """CASE 3: EMPTY MEDIA

    Treat empty media collections as text-only.
    """
    out = processor(text=["plain text", "more plain text"], images=[[], []], audio=[], padding=True)
    assert "pixel_values" not in out and "input_features" not in out, "empty media produced media tensors"
    return "no media tensors returned"


def case_4_count_validation(processor):
    """CASE 4: COUNT VALIDATION

    Number of placeholder should match the given media items for each sample.
    """
    invalid_inputs = [
        ({"text": "<|image|>", "images": [image(256, 256), image(256, 256)]}, "more images than placeholders"),
        ({"text": "<|image|><|image|>", "images": [image(256, 256)]}, "more placeholders than images"),
        ({"text": "no placeholder at all", "images": [image(256, 256)]}, "an image without any placeholder"),
        ({"text": "look at <|image|>"}, "an image placeholder without any media"),
        ({"text": ["a", "<|audio|>b"], "audio": [[clip(1.0)], []]}, "nested audio on the wrong sample"),
    ]
    for kwargs, label in invalid_inputs:
        try:
            processor(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for: {label}")
    return f"rejected {len(invalid_inputs)} mismatches"


def main():
    try:
        processor = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_nested_media, processor),
            run_case(case_2_flat_media, processor),
            run_case(case_3_empty_media, processor),
            run_case(case_4_count_validation, processor),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
