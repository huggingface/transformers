from datasets import load_dataset


splits = ["earnings22", "earnings21", "tedlium"]
# splits = ["earnings21", "tedlium"]

for _split in splits:
    print("\nLoading split:", _split)
    dataset = load_dataset("hf-audio/asr-leaderboard-longform", _split, split="test")
    print(dataset[0].keys())
    sample = dataset[0]["audio"]

    if _split == "tedlium":
        import pudb; pudb.set_trace()

    print(dataset[0]["audio"]["array"].shape)
    print(dataset[0]["audio"]["sampling_rate"])
    print(dataset[0]["audio"].keys())