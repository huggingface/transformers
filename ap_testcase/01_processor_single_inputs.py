"""Validate single-input processor behavior.

Covers text, image, audio, combined media, and remote URLs, checking placeholder expansion and output tensors.
"""

import numpy as np
from _common import CaseSkipped, bootstrap, finish, run_case, setup_failure


def setup():
    """SETUP

    Load the processor and create synthetic media.
    """
    transformers, checkpoint = bootstrap(("AutoProcessor",), processor_only=True)
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    dummy_image = np.random.default_rng(0).integers(
        0, 255, (300, 200, 3), dtype=np.uint8
    )  # resized to a 16x16 patch grid
    dummy_audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)  # 1 s
    return processor, dummy_image, dummy_audio


def case_1_text(processor):
    """CASE 1: TEXT

    Process a text-only prompt.
    """
    out = processor(text="What is the capital of Switzerland?", return_tensors="pt")
    assert set(out.keys()) == {"input_ids", "attention_mask"}, f"unexpected output keys: {set(out.keys())}"
    return f"input_ids {tuple(out['input_ids'].shape)}"


def case_2_image(processor, image):
    """CASE 2: IMAGE

    Process text with one image.
    """
    out = processor(text="<|image|> Describe this image.", images=[image], return_tensors="pt")
    decoded = processor.tokenizer.decode(out["input_ids"][0])
    height, width = (int(side) for side in out["image_sizes"][0])
    grid_h, grid_w = height // 16, width // 16
    assert decoded.count("<|image|>") == grid_h * grid_w, "one placeholder per 16x16 patch of the resized image"
    assert decoded.count("<|img_end_of_row|>") == grid_h - 1, "rows joined by exactly H-1 separators"
    assert f"<|img_start|>{grid_h}*{grid_w}<|img_token_start|>" in decoded, "height-first size header"
    assert out["pixel_values"].shape == (1, 3, height, width), (
        f"pixel_values has shape {tuple(out['pixel_values'].shape)}"
    )
    return f"300x200 -> {height}x{width}; {grid_h}x{grid_w} grid"


def case_3_audio(processor, audio):
    """CASE 3: AUDIO

    Process text with one audio clip.
    """
    out = processor(text="Transcribe: <|audio|>", audio=[audio], return_tensors="pt")
    decoded = processor.tokenizer.decode(out["input_ids"][0])
    assert decoded.count("<|audio|>") == 40, "1 s of 24 kHz audio -> ceil(24000/600) = 40 codes"
    assert "<|audio_start|>" in decoded and "<|audio_end|>" in decoded, "missing audio boundary tokens"
    assert out["input_features"].shape == (1, 1, 24000), (
        f"input_features has shape {tuple(out['input_features'].shape)}"
    )
    assert int(out["feature_attention_mask"].sum()) == 24000, "feature attention mask does not cover the clip"
    return f"40 placeholders; input_features {tuple(out['input_features'].shape)}"


def case_4_image_and_audio(processor, image, audio):
    """CASE 4: IMAGE AND AUDIO

    Process text with both media types.
    """
    out = processor(
        text="<|image|> What do you see, and what is said here: <|audio|>",
        images=[image],
        audio=[audio],
        return_tensors="pt",
    )
    assert {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_sizes",
        "input_features",
        "feature_attention_mask",
    } <= set(out.keys()), f"missing multimodal output keys; got {set(out.keys())}"
    height, width = (int(side) for side in out["image_sizes"][0])
    image_placeholders = (height // 16) * (width // 16)
    return f"sequence length {out['input_ids'].shape[-1]}; {image_placeholders} image + 40 audio placeholders"


def case_5_media_urls(processor):
    """CASE 5: MEDIA URLS

    Fetch and process remote media.
    """
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    audio_url = "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
    try:
        out = processor(text="<|image|> and <|audio|>", images=[image_url], audio=[audio_url], return_tensors="pt")
    except (ImportError, OSError) as error:  # missing librosa, or a network/fetch failure
        raise CaseSkipped(f"needs network and librosa ({type(error).__name__}: {error})") from error

    samples = int(out["feature_attention_mask"].sum())
    placeholders = processor.tokenizer.decode(out["input_ids"][0]).count("<|audio|>")
    assert placeholders == -(-samples // 600), "fetched audio is resampled to 24 kHz before counting"
    return f"image {tuple(out['pixel_values'].shape)}; {samples} audio samples -> {placeholders} placeholders"


def main():
    try:
        processor, image, audio = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_text, processor),
            run_case(case_2_image, processor, image),
            run_case(case_3_audio, processor, audio),
            run_case(case_4_image_and_audio, processor, image, audio),
            run_case(case_5_media_urls, processor),
        ]

    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
