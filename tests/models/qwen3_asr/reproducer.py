# 1) Install deps:
#    1.1) git clone https://huggingface.co/spaces/Qwen/Qwen3-ASR
#    1.2) cd qwen3-asr
#    1.3) pip install -r requirements.txt
# 2) Put this file in tests/models/qwen3_asr
# 3) Run: python tests/models/qwen3_asr/reproducer.py
#
# This script generates two fixtures:
#   - fixtures/qwen3_asr/expected_results_single.json
#   - fixtures/qwen3_asr/expected_results_batched.json

import json
from pathlib import Path

import torch

# append path for import: /root/transformers/qwen3-asr
import sys
sys.path.append("qwen3-asr")
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor  

def _pad_batch(seqs, pad_id: int):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]

@torch.inference_mode()
def _generate_single(processor, model, sound_path: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a helpful ASR assistant."},
                {
                    "type": "audio",
                    "path": sound_path,
                },
            ],
        }
    ]
    batch = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)
    seq = model.generate(**batch, max_new_tokens=64, do_sample=False).sequences
    inp_len = batch["input_ids"].shape[1]
    gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq        
    text = processor.batch_decode(seq, skip_special_tokens=True)
    return text, gen_ids[0].tolist()

if __name__ == "__main__":
    # Output paths
    ROOT = Path(__file__).parent.parent.parent
    FIXT_DIR = ROOT / "fixtures" / "qwen3_asr"
    FIXT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_SINGLE = FIXT_DIR / "expected_results_single.json"
    RESULTS_BATCHED = FIXT_DIR / "expected_results_batched.json"

    # Load model
    MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
    processor = Qwen3ASRProcessor.from_pretrained(MODEL_ID)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map=None, dtype=torch.bfloat16
    ).eval()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    
    # Single
    single_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
    single_text, single_ids = _generate_single(processor, model, single_url)
    single_payload = {
        "transcriptions": [single_text],
        "token_ids": _pad_batch([single_ids], pad_id),
    }
    with open(RESULTS_SINGLE, "w", encoding="utf-8") as f:
        json.dump(single_payload, f, ensure_ascii=False)
    print(f"Wrote {RESULTS_SINGLE}")

    # Batch
    urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    ]

    batched_texts, batched_ids, batched_input_ids = [], [], []
    for url in urls:
        text, ids = _generate_single(processor, model, url)
        batched_texts.append(text)
        batched_ids.append(ids)

    batched_payload = {
        "transcriptions": batched_texts,
        "token_ids": _pad_batch(batched_ids, pad_id),
    }
    with open(RESULTS_BATCHED, "w", encoding="utf-8") as f:
        json.dump(batched_payload, f, ensure_ascii=False)
    print(f"Wrote {RESULTS_BATCHED}")