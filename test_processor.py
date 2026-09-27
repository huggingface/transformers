# import sys; sys.path.insert(0, 'src')
# import safetensors.torch

# print("=== Original checkpoint keys (first 30) ===")
# weights = safetensors.torch.load_file(r'C:\Users\Vandit\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-Tokenizer-12Hz\snapshots\7dd38ad4e9bad454aae9cd937d0cd577604fe229\model.safetensors')
# print(f"Total original keys: {len(weights)}")
# with open(r'C:\Users\Vandit\Desktop\coding\transformers\transformers\og_keys.txt', 'w') as f:
#     for k in weights.keys():
#         f.write(f"{k}\n")

# from transformers import AutoConfig
# from transformers.models.qwen3_tts_tokenizer_multi_codebook.modeling_qwen3_tts_tokenizer_multi_codebook import Qwen3TTSTokenizerMultiCodebookModel
# c = AutoConfig.for_model('qwen3_tts_tokenizer_multi_codebook')
# m = Qwen3TTSTokenizerMultiCodebookModel(c)
# sd = m.state_dict()
# print(f"Total HF keys: {len(sd)}")
# with open(r'C:\Users\Vandit\Desktop\coding\transformers\transformers\hf_keys.txt', 'w') as f:
#     for k in sd.keys():
#         f.write(f"{k}\n")

# print("Keys written to og_keys.txt and hf_keys.txt")

# og_keys = set(weights.keys())
# hf_keys = set(sd.keys())

# print("\n=== In OG but not in HF (need remapping) ===")
# for k in sorted(og_keys - hf_keys):
#     print(f"  OG: {k}")

# print("\n=== In HF but not in OG (extra in HF) ===")
# for k in sorted(hf_keys - og_keys):
#     print(f"  HF: {k}")


# utils/generate_qwen3_tts_tokenizer_mc_fixture.py
import json
import torch
import numpy as np
from pathlib import Path
from transformers import Qwen3TTSTokenizerMultiCodebookModel

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # or local path
OUT = Path("tests/fixtures/qwen3_tts_tokenizer_multi_codebook")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

model = Qwen3TTSTokenizerMultiCodebookModel.from_pretrained(CHECKPOINT, dtype=torch.bfloat16)
model.eval()

# 1 second of audio at 24kHz
input_values = torch.randn(1, 24000, dtype=torch.bfloat16)

with torch.no_grad():
    encoded = model.encode(input_values)

result = {
    "input_values": input_values.squeeze(0).float().tolist(),
    "audio_codes": encoded.audio_codes.cpu().tolist(),
}

with open(OUT / "expected_results.json", "w") as f:
    json.dump(result, f)

print("Saved codec fixture.")
