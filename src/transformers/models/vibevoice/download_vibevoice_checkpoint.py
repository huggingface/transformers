import os

import requests
from safetensors.torch import load_file, save_file


# Target folder
folder_path = "/raid/eric/vibevoice"
os.makedirs(folder_path, exist_ok=True)

# Hugging Face shard URLs
shard_urls = [
    "https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00001-of-00003.safetensors",
    "https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00002-of-00003.safetensors",
    "https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00003-of-00003.safetensors"
]

# Local paths for shards
shard_paths = [os.path.join(folder_path, os.path.basename(url)) for url in shard_urls]

# Download each shard if not already downloaded
for url, path in zip(shard_urls, shard_paths):
    if not os.path.exists(path):
        print(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"{path} already exists, skipping download.")

# Merge shards into one file
combined = {}
for shard_path in shard_paths:
    combined.update(load_file(shard_path))

combined_path = os.path.join(folder_path, "VibeVoice-1.5B-combined.safetensors")
save_file(combined, combined_path)

print(f"Combined model saved to {combined_path}")
