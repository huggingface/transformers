# 1) Install NeMo, run: pip install nemo_toolkit[asr]
# 2) Put file in tests/models/parakeet
# 3) Run: python tests/models/parakeet/reproducer_batched.py

from nemo.collections.asr.models import ASRModel
import torch
import os
import json
from datasets import load_dataset
import soundfile as sf
from pathlib import Path

TMP_DIR = "./tmp"
NUM_SAMPLES = 1 
PAD_TOKEN_ID = 1024

# relative to tests/models/parakeet
RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_single.json"
results = {}

os.makedirs(TMP_DIR, exist_ok=True)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.sort("id")[:NUM_SAMPLES]["audio"]

for i, sample in enumerate(dataset):
    audio = sample["array"]
    sampling_rate = sample["sampling_rate"]
    audio_path = os.path.join(TMP_DIR, f"{i}.wav")
    sf.write(audio_path, audio, 16000)

samples = [os.path.join(TMP_DIR, f"{i}.wav") for i in range(NUM_SAMPLES)]
samples.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

model = ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b", map_location=torch.device('cuda:0'))
model.use_pytorch_sdpa = True
out = model.transcribe(
    audio=samples,
    batch_size=len(samples),
    return_hypotheses=False,
    num_workers=0,
    channel_selector=None,
    augmentor=None,
    verbose=False,
    timestamps=None,
    override_config=None
)

for el in out:
    print("text:", el.text)
    print("score:", el.score)
# write to dict
results["transcriptions"] = [el.text for el in out]
results["scores"] = [float(el.score) for el in out]

# let's pad every sequence with padding token to have a batched tensor as in Transformers
sequences = [el.y_sequence for el in out]
max_length = max(len(seq) for seq in sequences)
padded_sequences = []

for seq in sequences:
    padding_length = max_length - len(seq)
    if padding_length > 0:
        padded_seq = torch.cat([seq, torch.full((padding_length,), PAD_TOKEN_ID, dtype=seq.dtype)])
    else:
        padded_seq = seq
    padded_sequences.append(padded_seq)

batched_tensor = torch.stack(padded_sequences)
print(batched_tensor)
results["token_ids"] = batched_tensor.tolist() 

# save json
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)