import os
from datasets import load_dataset

token = os.getenv("HF_HUB_READ_TOKEN", True)
ds = load_dataset("mozilla-foundation/common_voice_6_1", "ja", split="test", streaming=True, token=token)
