"""Validate processor-driven and manual audio tokenization.

Checks token counts, exact manual parity, and the effect of processor peak normalization.
"""

import numpy as np
import torch
from _common import bootstrap, finish, run_case, setup_failure


def setup():
    """SETUP

    Load the processor, model, and synthetic audio.
    """
    transformers, checkpoint = bootstrap(("Apertus1p5ForConditionalGeneration", "AutoProcessor"))
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    print("SETUP: loading model (bf16, CPU) ...")
    model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    seconds = 1.5
    time = np.arange(int(24000 * seconds)) / 24000.0
    waveform = (0.8 * np.sin(2 * np.pi * 440.0 * time) * np.exp(-time)).astype(np.float32)
    return processor, model, waveform


def processor_tokens(processor, model, waveform):
    inputs = processor(text="<|audio|>", audio=[waveform], return_tensors="pt")
    with torch.no_grad():
        vocab_ids = model.model.get_audio_tokens(inputs["input_features"], inputs["feature_attention_mask"])
    return inputs, vocab_ids


def normalized_audio_codes(model, waveform):
    peak = max(float(np.abs(waveform).max()), 1e-10)
    normalized = waveform * (10 ** (-3.0 / 20.0) / peak)
    audio = torch.tensor(normalized, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        return model.model.audio_tokenizer.encode(audio).audio_codes.flatten()


def case_1_processor_path(processor, model, waveform):
    """CASE 1: PROCESSOR PATH

    Convert processor output into audio tokens.
    """
    inputs, vocab_ids = processor_tokens(processor, model, waveform)
    placeholders = processor.tokenizer.decode(inputs["input_ids"][0]).count("<|audio|>")
    expected_codes = -(-len(waveform) // 600)
    assert vocab_ids.numel() == placeholders == expected_codes, (
        f"expected {expected_codes} codes and placeholders, got {vocab_ids.numel()} and {placeholders}"
    )
    first = int(vocab_ids[0])
    expected = f"<|audio token {first - model.config.audio_token_offset}|>"
    assert processor.tokenizer.convert_ids_to_tokens(first) == expected, "incorrect vocabulary token mapping"
    return f"{vocab_ids.numel()} codes; first token {expected}"


def case_2_manual_path(processor, model, waveform):
    """CASE 2: MANUAL PATH

    Match normalized manual codes exactly.
    """
    _, vocab_ids = processor_tokens(processor, model, waveform)
    code_ids = normalized_audio_codes(model, waveform)
    manual_vocab_ids = code_ids + model.config.audio_token_offset
    assert torch.equal(manual_vocab_ids, vocab_ids), "manual and processor paths produced different codes"
    return f"{code_ids.numel()} codes; bit-identical"


def case_3_normalization(processor, model, waveform):
    """CASE 3: NORMALIZATION

    Confirm peak normalization changes the codes.
    """
    _, vocab_ids = processor_tokens(processor, model, waveform)
    with torch.no_grad():
        raw_ids = (
            model.model.audio_tokenizer.encode(
                torch.tensor(waveform, dtype=torch.float32)[None, None, :]
            ).audio_codes.flatten()
            + model.config.audio_token_offset
        )
    agreement = (raw_ids == vocab_ids).float().mean().item()
    assert agreement < 1.0, "unnormalized audio unexpectedly produced identical codes"
    return f"unnormalized agreement {agreement:.1%}"


def main():
    try:
        processor, model, waveform = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_processor_path, processor, model, waveform),
            run_case(case_2_manual_path, processor, model, waveform),
            run_case(case_3_normalization, processor, model, waveform),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
