# python test_file/convert_asr.py \
#   --src zai-org/GLM-ASR-Nano-2512/model.safetensors \
#   --dst GLM-ASR-Nano-2512-change/model.safetensors

import argparse
import os

from safetensors.torch import load_file, save_file


def rename_key(k: str):
    if k.endswith("audio_bos_eos_token.weight"):
        return None

    if k.startswith("audio_encoder.proj."):
        return None

    if k == "model.norm.weighth":
        return "language_model.model.norm.weight"

    if k.startswith("model."):
        return "language_model.model." + k[len("model.") :]

    if k.startswith("lm_head."):
        return "language_model.lm_head." + k[len("lm_head.") :]

    if k.startswith("audio_encoder.adapting.0."):
        return "multi_modal_projector.linear_1." + k.split(".")[-1]

    if k.startswith("audio_encoder.adapting.2."):
        return "multi_modal_projector.linear_2." + k.split(".")[-1]

    if k.startswith("audio_encoder.whisper."):
        return "audio_tower." + k[len("audio_encoder.whisper.") :]

    if k.startswith("audio_encoder.layer_norm."):
        return "audio_tower.layer_norm." + k.split(".")[-1]

    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    state = load_file(args.src, device="cpu")
    new_state = {}
    collisions = []

    for old_k, v in state.items():
        new_k = rename_key(old_k)
        if new_k is None:
            print(f"[DROP]   {old_k}")
            continue

        tag = "KEEP" if old_k == new_k else "RENAME"
        print(f"[{tag}] {old_k} -> {new_k}")

        if new_k in new_state:
            collisions.append((old_k, new_k))
        new_state[new_k] = v

    if collisions:
        for o, n in collisions[:20]:
            print(f"COLLISION {o} -> {n}")
        raise RuntimeError(f"key collision: {len(collisions)}")

    save_file(new_state, args.dst)

    print(f"saved: {args.dst}")
    print(f"keys: {len(state)} -> {len(new_state)}")


if __name__ == "__main__":
    main()
