"""
Setup:
```
pip install huggingface_hub git-lfs
git lfs install
```

Create HF token
1. https://huggingface.co/settings/tokens/new?tokenType=write
2. from Terminal: `export HF_TOKEN=hf_xxxxx`
"""

import os
import shutil
import subprocess
from huggingface_hub import HfApi, create_repo

# --------------------------
# Update these settings!
# --------------------------
SRC_REPO = "bezzam/VibeVoice-1.5B"
DST_REPO = "bezzam/VibeVoice-1.5B-hf2"
LOCAL_DIR = "/raid/eric/VibeVoice_tmp"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set: export HF_TOKEN=hf_xxxxx
# --------------------------

if HF_TOKEN is None:
    raise ValueError("Please set HF_TOKEN environment variable with your Hugging Face write token.")

api = HfApi()

# -------------------------------------------------
# 1. Clone the source repo with Git LFS enabled
# -------------------------------------------------
print("Cloning source repo using LFS...")
if os.path.exists(LOCAL_DIR):
    shutil.rmtree(LOCAL_DIR)

subprocess.run(["git", "lfs", "install"], check=True)

subprocess.run([
    "git", "clone",
    f"https://huggingface.co/{SRC_REPO}",
    LOCAL_DIR
], check=True)

# -------------------------------------------------
# 2. Remove git history (make a clean, non-git repo)
# -------------------------------------------------
print("Removing git history...")

git_dir = os.path.join(LOCAL_DIR, ".git")
if os.path.exists(git_dir):
    shutil.rmtree(git_dir)

# repo is now a clean folder with no git history

# -------------------------------------------------
# 3. Create the destination repo (or skip if exists)
# -------------------------------------------------
print(f"Ensuring destination repo exists: {DST_REPO}")
create_repo(DST_REPO, token=HF_TOKEN, exist_ok=True)

# -------------------------------------------------
# 4. Upload the cleaned folder
# -------------------------------------------------
print("Uploading cleaned repo to Hugging Face Hub...")

api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=DST_REPO,
    token=HF_TOKEN,
    commit_message="Initial import with cleaned history",
    repo_type="model"  # Change if needed
)

# delete tmp folder


print("\nDone! Uploaded to:", DST_REPO)
