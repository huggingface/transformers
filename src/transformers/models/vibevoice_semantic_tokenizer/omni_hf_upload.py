"""
Example output: https://huggingface.co/bezzam/omniASR-CTC-300M/tree/main
"""

from huggingface_hub import HfApi, create_repo
import requests
import os

hf_account = "bezzam"

# Specify the model weights URL: https://github.com/facebookresearch/omnilingual-asr?tab=readme-ov-file#model-architectures
url = "https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt"
# url = "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt"
output = os.path.basename(url)


if "omniASR-LLM-7B.pt" in url:
    tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model"
elif "W2V" in url:
    # No tokenizer for W2V models: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models.yaml#L19-L45
    tokenizer_url = None
else:
    tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model"


# ---  Download the model weights and tokenizer---
print(f"⬇️ Downloading model weights from {url}...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

if tokenizer_url:
    tokenizer_output = os.path.basename(tokenizer_url)
    print(f"⬇️ Downloading tokenizer from {tokenizer_url}...")
    with requests.get(tokenizer_url, stream=True) as r:
        r.raise_for_status()
        with open(tokenizer_output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# ---  Create (or ensure) the repo exists ---
repo_id = f"{hf_account}/{os.path.splitext(output)[0]}"

# Set `private=True` if you don’t want it public
create_repo(repo_id, exist_ok=True)

# --- Upload model and tokenizer ---
api = HfApi()

api.upload_file(
    path_or_fileobj=output,  # local path to your file
    path_in_repo=output,     # how it will appear in the repo
    repo_id=repo_id,
    repo_type="model"
)
if tokenizer_url:
    api.upload_file(
    path_or_fileobj=tokenizer_output,  # local path to your file
    path_in_repo=tokenizer_output,     # how it will appear in the repo
    repo_id=repo_id,
    repo_type="model",
)
print(f"✅ Uploaded to https://huggingface.co/{repo_id}")

# ---  Clean up the local file if desired ---
os.remove(output)
if tokenizer_url:
    os.remove(tokenizer_output)