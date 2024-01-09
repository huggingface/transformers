from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_6_1", "ja", split="test", streaming=True, token=True)
