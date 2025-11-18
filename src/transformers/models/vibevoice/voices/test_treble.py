from datasets import Audio, load_dataset


base_url = "https://huggingface.co/datasets/treble-technologies/current-Treble10-RIR/resolve/main/"


# 1. Load one Treble10 RIR
rir_ds = load_dataset("treble-technologies/current-Treble10-RIR", split="rir_mono")

# -- add base URL to filenames
rir_ds = rir_ds.map(lambda x: {"Filename": base_url + x["Filename"]})
rir_ds = rir_ds.cast_column("Filename", Audio())

rir_rec = next(iter(rir_ds))

import pudb; pudb.set_trace()

rir = rir_rec["audio"]["array"]
rir_sr = rir_rec["audio"]["sampling_rate"]

import pudb; pudb.set_trace()
