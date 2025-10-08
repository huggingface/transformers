from transformers import MarianTokenizer, MarianMTModel
import sys
import torch
import numpy as np
import onnxruntime as ort

# standalone script to test the ONNX export of the Marian model
_required = ["httpx", "huggingface_hub",
             "sentencepiece", "onnx", "onnxruntime", "torch"]
_missing = []
for _pkg in _required:
    try:
        __import__(_pkg)
    except Exception:
        _missing.append(_pkg)
if _missing:
    raise SystemExit(
        "Missing runtime dependencies: " + ", ".join(_missing) +
        "\nInstall with: pip install -U " + " ".join(_missing)
    )

# Optional: enforce minimal hub version used by current transformers mainline
try:
    import huggingface_hub as _hub
    try:
        from packaging import version as _v
        if _v.parse(_hub.__version__) < _v.parse("1.0.0rc4"):
            print(
                f"[WARN] huggingface_hub>={"1.0.0rc4"} recommended, found {_hub.__version__}.\n"
                "If you encounter import errors, upgrade with: pip install -U \"huggingface_hub>=1.0.0rc4,<2.0.0\"",
                file=sys.stderr,
            )
    except Exception:
        # If packaging is unavailable, skip strict version check.
        pass
except Exception:
    pass

try:
    # Import wrapper from installed package if available
    from transformers.models.marian.modeling_marian import MarianDecoderOnnxWrapper
except Exception:
    MarianDecoderOnnxWrapper = None


# Config

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
INPUT_SENTENCE = (
    "Using handheld GPS devices and programs like Google Earth, "
    "members of the Trio Tribe, who live in the rainforests of southern Suriname, "
    "map out their ancestral lands to help strengthen their territorial claims."
)
MAX_LENGTH = 20  # max length for generation comparison


# Load tokenizer & model

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()


# PyTorch generation

inputs = tokenizer(INPUT_SENTENCE, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=64)
with torch.no_grad():
    hf_output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_LENGTH,
        num_beams=1,
        do_sample=False
    )

print("PyTorch output IDs:", hf_output_ids[0].tolist())
print("PyTorch decoded:", tokenizer.decode(
    hf_output_ids[0], skip_special_tokens=True))


# Compute encoder outputs

with torch.no_grad():
    encoder_outputs = model.model.encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
    )
encoder_hidden_states = encoder_outputs.last_hidden_state


# Wrap decoder for ONNX export


# Use built-in ONNX-friendly wrapper when available; otherwise fallback to local definition
if MarianDecoderOnnxWrapper is None:
    class MarianDecoderOnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.model.decoder
            self.lm_head = model.lm_head
            self.final_logits_bias = model.final_logits_bias

        def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = self.lm_head(
                outputs.last_hidden_state) + self.final_logits_bias
            return logits

decoder_wrapper = MarianDecoderOnnxWrapper(model)


# Export to ONNX

dummy_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
dummy_enc_mask = inputs["attention_mask"]
torch.onnx.export(
    decoder_wrapper,
    args=(dummy_input_ids, encoder_hidden_states, dummy_enc_mask),
    f="decoder_with_lm_marian.onnx",
    input_names=["input_ids", "encoder_hidden_states",
                 "encoder_attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "encoder_hidden_states": {0: "batch", 1: "src_seq"},
        "encoder_attention_mask": {0: "batch", 1: "src_seq"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=17,
    do_constant_folding=False,
)
print("ONNX export done!")


# Step-by-step ONNX decoding

session = ort.InferenceSession("decoder_with_lm_marian.onnx")

decoder_input_ids = np.array(
    [[model.config.decoder_start_token_id]], dtype=np.int64)
# Include the start token to mirror HF generate token IDs for Marian
generated_ids = [model.config.decoder_start_token_id]

for step in range(MAX_LENGTH-1):
    ort_inputs = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy(),
        "encoder_attention_mask": inputs["attention_mask"].detach().cpu().numpy(),
    }
    logits = session.run(["logits"], ort_inputs)[0]
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_ids.append(next_token)
    decoder_input_ids = np.concatenate(
        [decoder_input_ids, [[next_token]]], axis=1)
    if next_token == tokenizer.eos_token_id:
        break

print("ONNX decoded IDs:", generated_ids)
print("ONNX decoded text:", tokenizer.decode(
    generated_ids, skip_special_tokens=True))


# Compare PyTorch vs ONNX

expected_ids = hf_output_ids[0][: len(generated_ids)].tolist()
match = expected_ids == generated_ids
print(f"Token-by-token match: {match}")
if not match:
    for i, (exp, act) in enumerate(zip(expected_ids, generated_ids)):
        if exp != act:
            print(
                f"Pos {i}: PyTorch={exp}, ONNX={act} -> {tokenizer.decode([act])}")
assert match, "ONNX decoding does not match PyTorch greedy output"
