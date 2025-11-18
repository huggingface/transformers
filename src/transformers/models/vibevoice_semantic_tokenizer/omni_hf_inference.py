"""
Setup
```
conda create -n omni python=3.10 -y
conda activate omni
pip install omnilingual-asr
conda install -c conda-forge libsndfile==1.0.31
```
"""


import torch
from huggingface_hub import hf_hub_download
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, Wav2Vec2AsrConfig


"""
Original: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md#31-parallel-generation-with-ctc-models
"""


pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M")


"""
Hugging Face Hub support (not working())
"""

# --- 1️⃣ Download model weights from Hugging Face Hub ---
repo_id = "bezzam/omniASR-CTC-300M"
model_checkpoint = "omniASR-CTC-300M.pt"
tokenizer_path = "omniASR_tokenizer.model"  # TODO use

# Downloads the file (and caches it automatically)
checkpoint_path = hf_hub_download(repo_id=repo_id, filename=model_checkpoint)

# --- 2️⃣ Load the checkpoint ---
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# --- 3️⃣ Inspect keys (optional) ---
# Usually you'll see something like 'state_dict', 'cfg', 'optimizer'
print(checkpoint.keys())

# --- 4️⃣ Initialize model config ---
config = Wav2Vec2AsrConfig()  # you can adjust based on checkpoint['cfg'] if available
model = Wav2Vec2AsrModel(config)

# --- 5️⃣ Load weights ---
# Use checkpoint['state_dict'] if available, else the checkpoint itself
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)

# load into pipline
pipeline = ASRInferencePipeline(
    model=model, 
    tokenizer=tokenizer,
)

